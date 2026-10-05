"""Scheduler loop: profitability, SLA guard, capacity, lost-race, health gate."""

from __future__ import annotations

import base64
import json

import asyncio
import dataclasses

import httpx
import pytest

from vorqd._crypto import BoxCipher, open_dek
from vorqd.backend import Normalized
from vorqd.blob import BlobError, BlobResolver, MemoryBlobSource
from .conftest import fake_cid, job_id_of, seal_container
from vorqd.config import (
    FAIL_GRACE_SECONDS,
    MAX_INPUT_BYTES_PER_UNIT,
    BackendConfig,
    LoadProbeConfig,
    ModelConfig,
    PricingConfig,
    ProviderConfig,
    SlaRate,
    VorqdConfig,
)
from vorqd.backend import ENVELOPE_SLACK_BYTES, input_shortfall
from vorqd.container import MIN_CONTAINER_BYTES, derive_dek, sealed_plaintext_bytes
from vorqd.errors import (
    BackendError,
    BackendGone,
    ChainConflict,
    ConfigError,
    OpRefused,
    OpRejected,
    UploadInvalid,
)
from vorqd.escrow import ReleaseRefused
from vorqd.node import CatalogModel, ClaimSimulation, OpResult
from vorqd.scheduler import Scheduler
from vorqd.state import InflightStore
from vorqd.types import INLINE_MAX_BYTES, ChainContext, EvmJob


# --- fakes ------------------------------------------------------------------


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


PROVIDER_ID = 7   # admin-issued; ambient from the session
MODEL = "deepseek-ai/deepseek-v4-pro:fp8"

#: An embedding model, because it is the modality where a client-declared
#: ``units_in`` is the *entire* bill: settlement reports ``completion_tok = 0``,
#: so the output leg prices to nothing and ``rate_in · units_in`` is all the
#: provider is ever paid.
EMBED_MODEL = "baai/bge-m3:fp16"



CHAIN = ChainContext(
    chain_id=31337,
    job_registry="0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
    provider_registry="0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
    ask_registry="0x5FbDB2315678afecb367f032d93F642f64180aa3",
    usdc="0xDc64a140Aa3E981100a9behA0000000000000000",
    decimals=6,
)


#: The curated catalog, as the node serves it: an id, a name, an enabled flag —
#: and a modality only where something actually names one. A coordinator node's
#: model record carries no modality at all, so the media model's is ``None`` here
#: and the operator's own ``modality:`` declaration is its only source.
MODEL_IDS = {
    MODEL: 7,
    "black-forest-labs/flux-2-dev:fp8": 9,
    "model-a:fp8": 11,
    "model-b:fp8": 12,
    "org/e2ee-model:fp8": 13,
    EMBED_MODEL: 14,
}
CATALOG = [
    CatalogModel(model_id=MODEL_IDS[MODEL], name=MODEL, enabled=True, modality="text"),
    CatalogModel(model_id=9, name="black-forest-labs/flux-2-dev:fp8", enabled=True),
    CatalogModel(model_id=11, name="model-a:fp8", enabled=True, modality="text"),
    CatalogModel(model_id=12, name="model-b:fp8", enabled=True, modality="text"),
    CatalogModel(model_id=13, name="org/e2ee-model:fp8", enabled=True, modality="text"),
    CatalogModel(model_id=14, name=EMBED_MODEL, enabled=True, modality="embedding"),
]


class FakeNode:
    """The coordinator node: the daemon's one chain collaborator.

    Everything that costs money arrives as a **signed artifact** and is recorded
    verbatim before it acts — ops through :meth:`push_op`, the whole ask book
    through :meth:`push_asks` — so a test can assert on the exact wire payload
    and on the signature that authorised it.

    It also mirrors the two registry rules a daemon cannot see any other way:
    ``set_identity`` and ``request_capacity`` each have their own **strictly**
    monotonic floor, and a repeated ``issued_at`` is refused with the contracts'
    own ``StaleOp``.
    """

    def __init__(self, jobs, clock, *, claim_conflict=False, provider_rec=None,
                 simulate=None, refuse_claim=None, revert_claim=False, revert_settle=False,
                 revert_fail=False, catalog=None):
        self._jobs = {j.job_id: j for j in jobs}
        self._clock = clock
        # A claim the chain refuses: another provider won the race, or the
        # client's escrow could not be funded. Either way a skip, never an error.
        self.claim_conflict = claim_conflict
        self.refuse_claim = refuse_claim
        # An op the node relayed and the chain then reverted — the state moved
        # after the pre-relay simulate had passed. The op is recorded and the
        # receipt is answered, but no row moved.
        self.revert_claim = revert_claim
        self.revert_settle = revert_settle
        self.revert_fail = revert_fail
        # The advisory gate's answer. None means "ok".
        self.simulate = simulate
        self.catalog = list(CATALOG if catalog is None else catalog)
        # What the node answers about the chain and about its own book: the block
        # every listing was answered at (the envelope's `as_of_block`) and the
        # chain's block time.
        self.as_of_block = 5_000
        self.block_time = 2_000
        self.bound = None
        self.ops: list[tuple[str, dict, str]] = []
        # Every open-book read, with the exact filters it carried.
        self.job_queries: list[dict] = []
        self.simulations: list[tuple[str, str]] = []
        self.settled = []
        self.failed = []
        self.snapshots: list[tuple[dict, str]] = []
        self.minted: list[str] = []
        # Every upload this daemon made through the files door: (purpose, bytes).
        self.uploaded: list[tuple[str, bytes]] = []
        # Default record: matching box key filled in by tests that call startup().
        self.provider_rec = provider_rec or {"box_key": None, "allow_all_models": True,
                                             "allowed_models": []}
        # The two ProviderRegistry floors, per op, exactly as the contract holds
        # them: `issuedAt <= lastIdentityAt[id]` reverts StaleOp, and identity and
        # capacity do not share a floor.
        self.floors: dict[str, int] = {}
        # How many further get_provider polls still answer the STALE record before
        # the identity op is visible — the indexing lag a confidential boot waits out.
        self.delay_update_rounds = 0
        self.provider_polls = 0
        self._pending_update: dict | None = None

    # -- reads ------------------------------------------------------------- #

    async def list_open_jobs(self, model_id, *, free=None, min_rate_in=None, min_rate_out=None):
        """``GET /evm/jobs?state=Open&free=…`` — the sweep's poll.

        The real node leases this provider the oldest open rows that clear its
        floors, at most ``free`` of them, and never a row designated to somebody
        else. Insertion order stands in for the book's own order.
        """
        assert isinstance(model_id, int), "the book is addressed by model id, never by name"
        name = self._name(model_id)
        self.job_queries.append({"model_id": model_id, "free": free,
                                 "min_rate_in": min_rate_in, "min_rate_out": min_rate_out})
        rows = [j for j in self._jobs.values() if j.state == "Open" and j.model == name]
        rows = [j for j in rows if (getattr(j, "designated", 0) or 0) in (0, PROVIDER_ID)]
        if min_rate_out is not None:
            rows = [j for j in rows if int(j.rate_out or 0) >= min_rate_out]
        if min_rate_in is not None:
            rows = [j for j in rows if int(j.rate_in or 0) >= min_rate_in]
        return rows if free is None else rows[:free]

    async def list_claimed_jobs(self, provider):
        self.claimed_lists = getattr(self, "claimed_lists", 0) + 1
        return [j for j in self._jobs.values() if j.state == "Claimed" and j.provider == provider]

    async def get_models(self):
        return list(self.catalog)

    def bind_models(self, resolver):
        self.bound = resolver

    async def upload_file(self, purpose, content, filename="result"):
        """``POST /v1/files``: the daemon's upload-first path for a result over
        the inline bound. Recorded like every other call that costs something,
        and answers a fresh cid each time — the node mints one per upload."""
        self.uploaded.append((purpose, content))
        return f"cid-uploaded-{len(self.uploaded)}"

    def _name(self, model_id):
        return next(m.name for m in self.catalog if m.model_id == model_id)

    async def chain_context(self):
        return CHAIN

    async def block_time_ms(self):
        return self.block_time

    async def simulate_claim(self, job_id, address):
        self.simulations.append((job_id, address))
        if self.simulate is not None:
            return self.simulate
        return ClaimSimulation(ok=True)

    # -- the ask book ------------------------------------------------------ #

    async def push_asks(self, snapshot, signature):
        """``PUT /evm/asks``: one signed snapshot, recorded before it lands."""
        assert isinstance(signature, str) and signature.startswith("0x")
        assert set(snapshot) == {"provider_id", "signed_at", "quotes"}
        floor = self.floors.get("asks", 0)
        if int(snapshot["signed_at"]) <= floor:
            raise ChainConflict("invalid_request", "not newer than this provider's floor",
                                code="stale_snapshot")
        self.floors["asks"] = int(snapshot["signed_at"])
        self.snapshots.append((json.loads(json.dumps(snapshot)), signature))
        return {"published": True, "tx_hash": "0x" + "44" * 32}

    @property
    def published(self):
        """The live models in the last snapshot — a withdrawal is not one."""
        if not self.snapshots:
            return None
        return sorted({self._name(int(q["model_id"]))
                       for q in self.snapshots[-1][0]["quotes"] if q["rate_out"] != "0"})

    async def push_op(self, op, payload, signature):
        """The one door that costs money. Every op is recorded before it acts."""
        assert isinstance(signature, str) and signature.startswith("0x")
        self.ops.append((op, dict(payload), signature))
        if op in ("set_identity", "request_capacity"):
            return self._registry_op(op, payload)
        job_id = payload["job_id"]
        if op == "claim":
            if self.claim_conflict or self.refuse_claim:
                raise OpRefused(self.refuse_claim or "NotOpen")
            if self.revert_claim:
                # A mined revert rolls the whole claim back, so nothing on the
                # row moves — the job is still Open and still anyone's.
                return OpResult(tx_hash="0x" + "11" * 32, status="reverted", block_number=1)
            j = self._jobs[job_id]
            j.state = "Claimed"
            j.provider = PROVIDER_ID
            j.claimed_at = int(payload["issued_at"])
            return OpResult(tx_hash="0x" + "11" * 32, status="success", block_number=1)
        if op == "settle":
            if self.revert_settle:
                # Recorded as pushed — the daemon did sign and send it — but the
                # row is untouched: a reverted settle leaves the job Claimed.
                return OpResult(tx_hash="0x" + "22" * 32, status="reverted", block_number=2)
            self.settled.append((job_id, payload.get("result"), payload.get("completion_tok")))
            self._jobs[job_id].state = "Settled"
            # The node pins the delivered bytes and mints the name; the claimant
            # learns it from this answer and from nowhere else, because nothing
            # it holds could have computed it.
            cid = f"bafkrei-minted-{len(self.settled)}"
            self.minted.append(cid)
            return OpResult(tx_hash="0x" + "22" * 32, status="success", block_number=2, result_cid=cid)
        if op == "fail":
            # A Fail op carries the job id and issued_at and nothing else: the
            # registry refunds either way and cannot verify a provider's account
            # of why, so no reason travels.
            if self.revert_fail:
                # Nobody was refunded: the row stays Claimed until the reclaim.
                return OpResult(tx_hash="0x" + "33" * 32, status="reverted", block_number=3)
            self.failed.append(job_id)
            self._jobs[job_id].state = "Cancelled"
            return OpResult(tx_hash="0x" + "33" * 32, status="success", block_number=3)
        raise AssertionError(f"unexpected op {op!r}")

    def _registry_op(self, op, payload):
        """A ProviderRegistry op, under the floor the contract actually keeps.

        ``issuedAt <= lastIdentityAt[id]`` (or ``lastCapacityAt[id]``) reverts
        with ``StaleOp``, and the node answers that revert as a 409 carrying the
        contract's own error name. The two floors are separate, which is why
        identity and capacity may share a second and a *restart* may not.
        """
        issued_at = int(payload["issued_at"])
        if issued_at <= self.floors.get(op, 0):
            raise OpRefused("StaleOp")
        self.floors[op] = issued_at
        if op == "set_identity":
            self._pending_update = {"box_key": payload["box_key"], "evidence": payload["evidence"]}
        return OpResult(tx_hash="0x" + "55" * 32, status="success", block_number=4)

    @property
    def capacity_requested(self):
        payloads = self.pushed("request_capacity")
        return payloads[-1]["n"] if payloads else None

    def pushed(self, op: str) -> list[dict]:
        return [payload for name, payload, _ in self.ops if name == op]

    async def get_provider(self, provider_id):
        self.provider_polls += 1
        if self._pending_update is not None:
            if self.delay_update_rounds > 0:
                self.delay_update_rounds -= 1   # not indexed yet
            else:
                self.provider_rec.update(self._pending_update)
                self._pending_update = None
        # A copy: the caller must re-poll to observe a change, as over HTTP.
        return dict(self.provider_rec)


class FakeCoord:
    """The coordinator seam the daemon actually uses: a session, nothing more.

    It carries no file surface, so any attempt to route job bytes through it
    fails the test that tried.
    """

    provider_id = PROVIDER_ID

    async def token(self):
        return "vorq_sess_test"

    def invalidate(self):  # pragma: no cover
        pass


class FakeDriver:
    def __init__(self, *, healthy=True, on_run=None):
        self._healthy = healthy
        self._on_run = on_run
        self.last_timeout_s = None
        self.calls = 0

    async def run(self, job, input, *, timeout_s=None):
        self.calls += 1
        self.last_timeout_s = timeout_s
        if self._on_run:
            self._on_run(job)
        return Normalized(kind="text", text="answer", completion_tokens=42, raw={"choices": []})

    async def healthy(self):
        return self._healthy


class FailingDriver:
    """A healthy driver whose backend call raises BackendError (e.g. it cannot
    serve the job) — until ``fail`` is cleared, after which it answers."""

    def __init__(self, *, retryable=False):
        self._retryable = retryable
        self._healthy = True
        self.fail = True
        self.calls = 0

    async def run(self, job, input, *, timeout_s=None):
        self.calls += 1
        if self.fail:
            raise BackendError("backend could not serve this job", retryable=self._retryable)
        return Normalized(kind="text", text="answer", completion_tokens=42, raw={"choices": []})

    async def healthy(self):
        return self._healthy


class FlakyDriver:
    """A driver that fails with the scripted errors first, then answers.

    ``on_run`` runs before every attempt, so a test can spend clock time on
    each — a slow backend that fails is the shape the deadline guard exists for.
    """

    def __init__(self, errors, *, on_run=None):
        self._errors = list(errors)
        self._on_run = on_run
        self.calls = 0
        self.timeouts = []

    async def run(self, job, input, *, timeout_s=None):
        self.calls += 1
        self.timeouts.append(timeout_s)
        if self._on_run:
            self._on_run(job)
        if self._errors:
            raise self._errors.pop(0)
        return Normalized(kind="text", text="answer", completion_tokens=42, raw={"choices": []})

    async def healthy(self):
        return True


class ResumableDriver:
    """An async backend: the submit yields a handle, and a later life may hand
    it back as ``resume``. ``runs`` records the ``resume`` value of every call."""

    resumable = True

    def __init__(self, *, fail=None, hang=False, on_run=None):
        self._fail = fail
        self._hang = hang
        self._on_run = on_run
        self.runs = []

    async def run(self, job, input, *, timeout_s=None, resume=None, on_handle=None):
        self.runs.append(resume)
        if resume is None:
            on_handle("resp_abc")
        if self._on_run:
            self._on_run(job)
        if self._hang:
            await asyncio.Event().wait()
        if self._fail is not None:
            raise self._fail
        return Normalized(kind="text", text="answer", completion_tokens=42, raw={"choices": []})

    async def healthy(self):
        return True


class FakeSleep:
    """The scheduler's sleep seam: records every wait and spends it on the clock,
    so a deadline test sees time pass exactly as the daemon would."""

    def __init__(self, clock):
        self._clock = clock
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)
        self._clock.advance(seconds)


class FakeMetrics:
    def __init__(self):
        self.claims = 0
        self.settles = 0
        self.fails = []
        self.loads = {}
        self.floor_discounts = {}
        self.probe_failures = []
        self.retries = []
        self.model_free = {}
        self.granted = None

    def on_claim(self):
        self.claims += 1

    def on_settle(self, duration):
        self.settles += 1

    def on_fail(self, reason):
        self.fails.append(reason)

    def on_backend_latency(self, model, seconds):
        pass

    def set_capacity_free(self, n):
        pass

    def set_asks_published(self, n):
        pass

    def set_backend_load(self, model, value):
        self.loads[model] = value

    def set_floor_discount(self, model, pct):
        self.floor_discounts[model] = pct

    def on_probe_failure(self, model):
        self.probe_failures.append(model)

    def on_retry(self, model):
        self.retries.append(model)

    def set_model_free(self, model, n):
        self.model_free[model] = n

    def set_capacity_granted(self, n):
        self.granted = n


class FakeLoad:
    """A load monitor whose readings the test sets directly."""

    def __init__(self, loads=None):
        self.loads = dict(loads or {})

    async def refresh(self):
        pass

    def load(self, model):
        return self.loads.get(model)


# --- helpers ----------------------------------------------------------------


def text_model(model="deepseek-ai/deepseek-v4-pro:fp8", *, sla="1h", **limits):
    """A served text model; ``limits`` are the entry's `backend:` limit keys
    (`retries`, `retry_backoff_s`, `timeout_s`, `concurrency`, `rate_limit`, ...)."""
    return ModelConfig(
        model=model,
        slas={sla: SlaRate(rate_in="0.2", rate_out="0.6")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"},
                              **limits),
    )


# A valid 32-byte Curve25519 private key (hex) so the scheduler can build its box
# cipher; the box key is a required config field.
_TEST_BOX_KEY = "aa" * 32

# A real secp256k1 key: the ops the scheduler pushes are signed for real, so a
# placeholder would only prove that nothing was signed.
_TEST_WALLET_KEY = "0x" + "4a" * 32


def make_config(models, capacity=4, *, box_key=_TEST_BOX_KEY, poll_interval_s=5,
                pricing=None, bid_filter=None, max_input_bytes_per_unit=MAX_INPUT_BYTES_PER_UNIT):
    return VorqdConfig(
        provider=ProviderConfig(wallet_key=_TEST_WALLET_KEY, box_key=box_key, api_url="http://x",
                                capacity=capacity, safety_margin_s=60,
                                poll_interval_s=poll_interval_s,
                                pricing=pricing or PricingConfig(),
                                bid_filter=bid_filter or {},
                                max_input_bytes_per_unit=max_input_bytes_per_unit),
        models=models,
    )


# One shared, content-addressed pin store: a CID names its bytes, so tests can
# never collide and none of them has to thread a source around.
SOURCE = MemoryBlobSource()

OWNER = "0x" + "11" * 20
# Every envelope names a result key — there is no in-the-clear result path, so a
# fixture without one is not a valid task.
CLIENT = BoxCipher.generate()

# The two keys a container's seed_wrap can be sealed to, and the whole of the
# difference between the two bid shapes: this daemon's own published box key
# (designated), or the coordinator's attested escrow key (open).
_UNSET = object()
DAEMON_BOX = BoxCipher(_TEST_BOX_KEY)
ESCROW = BoxCipher.generate()


def pin_task(payload, *, owner=OWNER, result_key=_UNSET, recipient=None, seed=None,
             sealed_to_owner=None, custom_id=None) -> tuple[str, str]:
    """Pin a container and return the ``(job_id, task_cid)`` it names.

    The envelope is encrypted under ``derive_dek(seed, owner)`` and the **seed**
    is sealed to ``recipient`` — the escrow by default, this daemon's box key for
    a designated bid. The commitment covers the container that results, so the
    job id is derived from it last.

    ``sealed_to_owner`` overrides the owner the key is derived under while
    leaving the job's own owner alone: that is the wrap-lifting attack, and a
    fixture is the only way to build one.
    """
    env = {"v": "vorq-env-v1", "owner": owner,
           "result_key": CLIENT.public_key if result_key is _UNSET else result_key,
           "input": payload}
    # Absent rather than null when unset — the client canonicalizes this dict into
    # the commitment preimage, so a key that carries no meaning must not be there.
    if custom_id is not None:
        env["custom_id"] = custom_id
    plaintext = json.dumps(env, sort_keys=True, separators=(",", ":")).encode()
    container = seal_container(plaintext, recipient=recipient or ESCROW.public_key,
                               owner=sealed_to_owner or owner, seed=seed)
    SOURCE.put(container, cid=fake_cid(container))
    return job_id_of(owner, container), fake_cid(container)


def open_text_job(tag="job_1", rate_out=600_000, rate_in=200_000, sla="1h",
                  model=MODEL, *, owner=OWNER, result_key=_UNSET):
    # `tag` only varies the payload so distinct jobs get distinct content ids —
    # a job id is now a fact about its bytes, never a chosen label.
    job_id, task_cid = pin_task({"input": f"hi {tag}"}, owner=owner, result_key=result_key)
    return EvmJob(job_id=job_id, model=model, state="Open", sla=sla, created_at=1000,
                  owner=owner, rate_in=rate_in, rate_out=rate_out, units_in=10, units_out=128,
                  task_cid=task_cid)


def designated_text_job(tag="designated", *, box=None, provider=PROVIDER_ID, model=MODEL):
    """A bid pinned to this provider: the DEK is sealed to its box key, so the
    daemon opens the container in-process and the escrow is never in the path."""
    job_id, task_cid = pin_task({"input": f"hi {tag}"}, recipient=(box or DAEMON_BOX).public_key)
    return EvmJob(job_id=job_id, model=model, state="Open", sla="1h", created_at=1000,
                  owner=OWNER, rate_in=200_000, rate_out=600_000, units_in=10, units_out=128,
                  task_cid=task_cid, designated=provider)


class FakeEscrow:
    """The coordinator's attested escrow: it holds the key an open bid's wrap is
    sealed to and releases the DEK against a claim already on record.

    Deliberately not a claim response — nothing this daemon claims hands it a
    key; it asks for one, with the wrap and the ciphertext digest, and can be
    refused by code.

    It answers the DEK **already derived**, exactly as the real one does: the
    coordinator unseals the seed and derives against the owner it read from
    chain, so the provider must not derive again. A fake that handed back the
    raw seed would make the open path pass under a scheduler that derived twice.
    """

    def __init__(self, cipher=None, *, refuse=None, owner=None):
        self._cipher = cipher or ESCROW
        self._refuse = refuse
        # The owner the escrow derives under. Defaults to the job's own, which
        # is what reading it off chain gives you.
        self._owner = owner
        self.calls: list[tuple] = []

    async def release(self, job, seed_wrap, ct_hash):
        self.calls.append((job.job_id, seed_wrap, ct_hash))
        if self._refuse:
            raise ReleaseRefused(self._refuse, "the escrow said no")
        return derive_dek(self._cipher.decrypt(seed_wrap), self._owner or job.owner)


_DEFAULT_ESCROW = object()


def make_scheduler(config, chain, driver, clock, metrics=None, *, coord=None,
                   escrow=_DEFAULT_ESCROW, load=None, store=None):
    drivers = {m.model: driver for m in config.models}
    sleep = FakeSleep(clock)
    sched = Scheduler(config, chain, coord or FakeCoord(), drivers, metrics or FakeMetrics(),
                      clock=clock, blobs=BlobResolver([SOURCE]), load=load,
                      escrow=FakeEscrow() if escrow is _DEFAULT_ESCROW else escrow, sleep=sleep,
                      store=store)
    sched.sleeps = sleep.calls
    return sched


# --- tests ------------------------------------------------------------------


async def test_happy_path_claims_and_settles_with_tokens():
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert [(s[0], s[2]) for s in chain.settled] == [(job.job_id, 42)]
    assert job.state == "Settled"


async def test_a_labelled_job_settles_with_its_label_still_on_it():
    """The label survives the **claim path**, not just the sealing helper.

    `_build_result` has been able to stamp a `custom_id` since the correlation
    block landed, and two tests above prove it does — by calling it directly. The
    path a real job takes is `_ingest` → `_open_task` → `_spawn` → `_run_job` →
    `_build_result`, and it dropped the value on the floor between the first and
    the third: unpacked from the envelope, never passed on, and every caller
    defaulted it to `None` on the way down. Nothing was red.

    So this asserts the whole trip, on the bytes that actually settle: seal a
    label into the container, run the scheduler, open what it sent. A batch line
    correlates by `custom_id` and this is the only thing that carries it — the
    coordinator holds no copy, by design.
    """
    clock = Clock()
    job_id, task_cid = pin_task({"input": "hi labelled"}, custom_id="req-7")
    job = EvmJob(job_id=job_id, model=MODEL, state="Open", sla="1h", created_at=1000,
                 owner=OWNER, rate_in=200_000, rate_out=600_000, units_in=10,
                 units_out=128, task_cid=task_cid)
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)

    await sched.run_once()
    await sched.join()

    (_, result, _), = chain.settled
    payload = json.loads(_open_sealed(base64.b64decode(result), CLIENT))
    assert payload["vorq"] == {"job_id": job.job_id, "custom_id": "req-7"}


class LeakyNode(FakeNode):
    """A node whose poll answer includes rows designated to another provider."""

    async def list_open_jobs(self, model_id, *, free, **filters):
        name = self._name(model_id)
        rows = [j for j in self._jobs.values() if j.state == "Open" and j.model == name]
        return rows[:free]


async def test_skips_jobs_designated_to_another_provider():
    # A job designated to another provider is sealed to that provider's key and
    # unclaimable by anyone else: the scheduler must skip it without attempting
    # the claim, while still claiming open jobs and jobs designated to itself.
    clock = Clock()
    theirs = open_text_job("theirs")
    theirs.designated = 999   # another provider's id
    mine = designated_text_job("mine")   # pinned to FakeCoord.provider_id
    open_job = open_text_job("open")
    # A coordinator that leased somebody else's job to this daemon anyway: the
    # skip is defensive now, and it still has to hold.
    chain = LeakyNode([theirs, mine, open_job], clock)
    sched = make_scheduler(make_config([text_model()], capacity=3), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert theirs.state == "Open"  # never claimed, not even attempted
    assert {s[0] for s in chain.settled} == {mine.job_id, open_job.job_id}


class TokenDriver:
    """A healthy driver returning a text result with a chosen completion count."""

    def __init__(self, completion_tokens):
        self._ct = completion_tokens

    async def run(self, job, input, *, timeout_s=None):
        return Normalized(kind="text", text="answer", completion_tokens=self._ct, raw={"choices": []})

    async def healthy(self):
        return True


async def test_missing_completion_count_abandons_without_settling():
    # A text backend that reports no usable count has no billable quantity; the job
    # must be abandoned (backend_error), never settled — so the escrow's full-cap
    # fallback can never fire.
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, TokenDriver(None), clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert "backend_error" in metrics.fails
    assert job.state == "Cancelled"  # never settled; a fail op goes out for an instant refund


async def test_completion_count_clamped_to_units_out():
    # units_out is the charged cap; a backend counter above it settles at the cap.
    clock = Clock()
    job = open_text_job()  # units_out=128
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, TokenDriver(200), clock)
    await sched.run_once()
    await sched.join()
    assert [(s[0], s[2]) for s in chain.settled] == [(job.job_id, 128)]  # 200 clamped to cap


async def test_completion_count_below_cap_settles_verbatim():
    clock = Clock()
    job = open_text_job()  # units_out=128
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, TokenDriver(30), clock)
    await sched.run_once()
    await sched.join()
    assert [(s[0], s[2]) for s in chain.settled] == [(job.job_id, 30)]  # unchanged, under the cap


async def test_backend_request_timeout_is_the_remaining_sla_budget():
    # An attempt may run until `deadline - safety_margin_s` and no longer: a
    # result that lands after that could not settle, so the request is better
    # turned over (and retried, where the entry allows) than left hanging.
    clock = Clock()
    driver = FakeDriver()
    for sla, expected in (("1h", 3600 - 60), ("24h", 86_400 - 60)):
        job = open_text_job(sla=sla)
        chain = FakeNode([job], clock)
        model = ModelConfig(
            model="deepseek-ai/deepseek-v4-pro:fp8",
            slas={sla: SlaRate(rate_in="0.2", rate_out="0.6")},
            backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
        )
        sched = make_scheduler(make_config([model]), chain, driver, clock)
        await sched.run_once()
        await sched.join()
        assert driver.last_timeout_s == expected


async def test_unprofitable_job_is_skipped():
    clock = Clock()
    job = open_text_job(rate_out=100_000)  # below the configured 600000
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert job.state == "Open"  # never claimed


# --- units_in: the payload a bid pays for, against the one it carries --------
#
# `units_in` is the client's own count and the chain bills the input leg at
# whatever it says, with nothing anywhere holding it to the payload. Unlike
# `units_out` there is no clamp to reach for — a provider cannot deliver less
# input than it was sent — so the whole remedy is to decline, and the only
# question is where. These pin that it is *before* the claim, where declining is
# free, and that the check after decryption can never be the one that fires.


def underdeclared_text_job(tag="greedy", *, units_in=1, size=200_000, model=MODEL,
                           rate_in=200_000, rate_out=600_000, sla="1h"):
    """A bid whose declared input units come nowhere near the payload it names."""
    job_id, task_cid = pin_task({"input": "x" * size})
    return EvmJob(job_id=job_id, model=model, state="Open", sla=sla, created_at=1000,
                  owner=OWNER, rate_in=rate_in, rate_out=rate_out,
                  units_in=units_in, units_out=128, task_cid=task_cid, expires_at=99_000)


def embedding_model(model=EMBED_MODEL):
    """An embedding entry: an input rate and **no output rate at all**, which is
    what makes `units_in` the whole bill for anything claimed here."""
    return ModelConfig(
        model=model,
        modality="embedding",
        slas={"1h": SlaRate(rate_in="0.2", rate_out="0")},
        backend=BackendConfig(preset="openai-embeddings",
                              params={"base_url": "http://r/v1", "model": "e"}),
    )


async def test_a_bid_declaring_less_input_than_it_carries_is_not_claimed():
    clock = Clock()
    job = underdeclared_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert job.state == "Open"           # still on the book for a looser provider
    assert chain.pushed("claim") == []   # the op was never even signed
    assert metrics.claims == 0
    # The point of refusing *here*: nothing was claimed, so nothing was failed.
    # A fail would refund the client in full and leave the relayer's claim gas
    # unrecovered — under-declaring must not become a way to spend someone else's.
    assert metrics.fails == []


async def test_an_embedding_bid_that_declares_one_unit_for_a_long_prompt_is_not_claimed():
    """The motivating case. `rate_out` is zero, so `rate_in · units_in` is the
    entire settlement — one declared unit buys 200 KB of prompt for one atomic unit.
    """
    clock = Clock()
    job = underdeclared_text_job(model=EMBED_MODEL, rate_out=0)
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([embedding_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()

    assert job.state == "Open"
    assert chain.settled == []


async def test_an_honestly_declared_payload_is_claimed_and_settles():
    """The guard's other half: a bid that declares what it carries is untouched.
    Sized at 200 KB so it is the declaration, not the payload, doing the work.
    """
    clock = Clock()
    job = underdeclared_text_job(units_in=200_000 // 4)
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_a_media_bid_is_never_judged_on_its_input_units():
    """Media quotes an output rate and no input one, so the input leg bills nothing
    whatever `units_in` says. Judging it would decline honest work over a number
    nobody is charged for — and a media payload is exactly the shape that would
    trip a byte floor, since a prompt carrying an asset is bytes the pixel unit
    already prices.

    (`media_model`, `MediaDriver` and `BlindChain` live with the modality tests
    further down; this belongs here, with the rule it is about.)
    """
    clock = Clock()
    # A 200 KB prompt on a bid declaring no input units at all.
    job_id, task_cid = pin_task({"prompt": "a cat " + "x" * 200_000,
                                 "width": 1024, "height": 1024, "num_images": 1})
    job = EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                 owner=OWNER, rate_in=None, rate_out=20_000, units_in=None,
                 units_out=ONE_MEGAPIXEL, task_cid=task_cid)
    chain = BlindChain([job], clock)          # catalog down; modality comes from config
    sched = make_scheduler(make_config([media_model(modality="image")]), chain,
                           MediaDriver(), clock)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_a_media_bid_that_prices_its_input_is_still_never_judged_by_weight():
    """The trap in the pre-claim gate: `EvmJob.modality` is a *placeholder* until
    `_ingest` stamps it, so a media job reads as `"text"` for the whole of
    `_poll_model`. The gate therefore cannot ask the job what it is — it has to be
    told, from the model the sweep is polling for.

    Nothing on the shipped catalog prices a media input side today, so `rate_in`
    alone happens to exempt every media bid. This is the case that stops being
    true the moment one does: pixel-seconds of reference against a fat payload,
    which is honest and must claim.
    """
    clock = Clock()
    # A 64×64 reference — 4096 pixel-seconds — carried as 200 KB of payload. The
    # two numbers are unrelated by construction: pixels are what the work costs,
    # bytes are what the encoder happened to spend, and a byte floor reads the
    # honest declaration as a 30-fold under-declaration.
    job_id, task_cid = pin_task({"prompt": "a cat " + "x" * 200_000,
                                 "width": 1024, "height": 1024, "num_images": 1})
    job = EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                 owner=OWNER, rate_in=10, rate_out=20_000, units_in=64 * 64,
                 units_out=ONE_MEGAPIXEL, task_cid=task_cid)
    priced = ModelConfig(
        model=MEDIA_MODEL,
        slas={"24h": SlaRate(rate_in="0.00001", rate_out="0.02")},
        backend=BackendConfig(preset="openai-chat",
                              params={"base_url": "http://r/v1", "model": "flux"}),
        modality="image",
    )
    chain = BlindChain([job], clock)
    sched = make_scheduler(make_config([priced]), chain, MediaDriver(), clock)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_an_operator_who_turns_the_guard_off_claims_the_bid_anyway():
    clock = Clock()
    job = underdeclared_text_job()
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()], max_input_bytes_per_unit=0),
                           chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_the_shipped_clients_declaration_clears_the_floor_at_every_size():
    """The drift test, and the reason the daemon does not reproduce the formula.

    It builds the envelope exactly as a client does, declares `units_in` exactly
    as both SDKs do — `max(1, len(canonical(input)) // 4)` — and asserts the floor
    forgives it at every size and with or without a label. This is the one test
    that fails if either SDK changes its divisor, if the envelope grows a field,
    or if the container's own framing drifts.
    """
    for size in (1, 1024, 64 * 1024, 1024 * 1024):
        for custom_id in (None, "x" * 256):
            payload = {"input": "y" * size}
            declared = max(1, len(json.dumps(payload, sort_keys=True,
                                             separators=(",", ":")).encode()) // 4)
            _, task_cid = pin_task(payload, custom_id=custom_id)
            container = await SOURCE.get(task_cid)
            job = EvmJob(job_id="j", model=MODEL, state="Open", sla="1h", created_at=0,
                         owner=OWNER, rate_in=200_000, rate_out=600_000,
                         units_in=declared, units_out=128)

            assert input_shortfall(job, sealed_plaintext_bytes(len(container)),
                                   bytes_per_unit=MAX_INPUT_BYTES_PER_UNIT,
                                   modality="text") == 0, (size, custom_id)


async def test_unknown_sla_window_skipped():
    clock = Clock()
    job = open_text_job(sla="7d")  # window the provider does not serve
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert job.state == "Open"


async def test_lost_race_is_handled():
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock, claim_conflict=True)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []  # no crash, nothing settled


async def test_escrow_unfunded_claim_is_treated_as_lost_race():
    # The client's escrow is funded at claim; if the wallet was drained the op is
    # refused by the chain itself. The daemon must treat it exactly as a lost
    # race and never run the job — the payout guarantee at the provider edge.
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock, refuse_claim="EscrowUnfunded")
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []            # nothing settled
    assert driver.last_timeout_s is None  # backend never ran → no unpaid work


async def test_sla_guard_abandons_without_settling():
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()

    # backend "takes" 3600s, pushing past the 1h deadline minus 60s margin
    driver = FakeDriver(on_run=lambda j: clock.advance(3600))
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert "sla_abandon" in metrics.fails
    assert chain.failed == []  # SLA-abandon never pushes a fail op; only BackendError does


async def test_backend_error_reports_fail_and_never_settles():
    """A BackendError abandons the job AND pushes a fail op, so the client is refunded now."""
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FailingDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert chain.failed == [job.job_id]
    # A Fail op carries the job id and issued_at and nothing else. The registry
    # refunds the client either way and could not verify a provider's account of
    # why, so the reason is metered here, where an operator can act on it.
    (payload,) = chain.pushed("fail")
    assert set(payload) == {"job_id", "issued_at"}
    assert metrics.fails == ["backend_error"]


async def test_fail_report_error_does_not_crash_job_path():
    """chain.fail blowing up must not escalate — refund degrades to SLA-expiry reclaim."""
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()

    async def boom(job_id, reason):
        raise RuntimeError("network down")
    chain.fail = boom

    sched = make_scheduler(make_config([text_model()]), chain, FailingDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert metrics.fails == ["backend_error"]


async def test_capacity_limits_concurrent_claims():
    clock = Clock()
    jobs = [open_text_job("cap_1"), open_text_job("cap_2")]
    chain = FakeNode(jobs, clock)
    sched = make_scheduler(make_config([text_model()], capacity=1), chain, FakeDriver(), clock)

    await sched.run_once()   # only one slot -> one claim
    await sched.join()
    assert len(chain.settled) == 1

    await sched.run_once()   # slot freed -> second job now
    await sched.join()
    assert len(chain.settled) == 2


async def test_health_gate_withdraws_and_republishes():
    clock = Clock()
    chain = FakeNode([], clock)
    driver = FakeDriver(healthy=True)
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)

    await sched.sync_asks()
    assert chain.published == ["deepseek-ai/deepseek-v4-pro:fp8"]

    driver._healthy = False
    await sched.sync_asks()
    assert chain.published == []  # withdrawn

    driver._healthy = True
    await sched.sync_asks()
    assert chain.published == ["deepseek-ai/deepseek-v4-pro:fp8"]  # republished


def _windowed_model(sla_backends):
    """One model served at two windows, with per-window backends."""
    return ModelConfig(
        model=MODEL,
        slas={"1h": SlaRate(rate_in="0.2", rate_out="0.6"),
              "24h": SlaRate(rate_in="0.1", rate_out="0.3")},
        backend=BackendConfig(preset="openai-chat",
                              params={"base_url": "http://r/v1", "model": "rt"}),
        sla_backends=sla_backends,
    )


_BATCH_BACKEND = BackendConfig(preset="openai-batch",
                               params={"base_url": "http://r/v1", "model": "rt"})


@pytest.mark.parametrize("sla, on_override", [("24h", True), ("1h", False)])
async def test_a_job_runs_on_the_backend_of_its_sla_window(sla, on_override):
    clock = Clock()
    override, default = FakeDriver(), FakeDriver()
    config = make_config([_windowed_model({"24h": _BATCH_BACKEND})])
    job = open_text_job(sla=sla)
    chain = FakeNode([job], clock)
    sched = make_scheduler(config, chain, default, clock)
    sched._sla_drivers[(MODEL, "24h")] = override

    await sched.run_once()
    await sched.join()

    assert job.state == "Settled"
    assert (override.calls, default.calls) == ((1, 0) if on_override else (0, 1))


async def test_the_book_needs_every_backend_of_a_model_healthy():
    """A window whose backend is down takes the whole model off the book.

    Withdrawing that window alone would be perfectly possible — the book is
    keyed `(model_id, sla_secs)`, so one slot can be zeroed while its siblings
    stand. The whole-model rule is a deliberate simplification instead: the
    health gate answers per model, and a book that publishes a model only while
    every backend of it is up is one partial state fewer to reason about."""
    clock = Clock()
    chain = FakeNode([], clock)
    default, override = FakeDriver(), FakeDriver()
    sched = make_scheduler(make_config([_windowed_model({"24h": _BATCH_BACKEND})]),
                           chain, default, clock)
    sched._sla_drivers[(MODEL, "24h")] = override

    await sched.sync_asks()
    assert chain.published == [MODEL]

    override._healthy = False          # the default backend is still healthy
    await sched.sync_asks()
    assert chain.published == []       # withdrawn all the same


# --- the breaker: a backend failing job after job leaves the book ---------------


def _tripping(n_jobs, *, trip_after=3, capacity=4, driver=None, **limits):
    """``n_jobs`` open bids against a backend that fails every attempt; with
    ``retries`` 0 each job is one attempt, so each is one fault."""
    clock = Clock()
    jobs = [open_text_job(f"trip_{i}") for i in range(n_jobs)]
    chain = FakeNode(jobs, clock)
    metrics = FakeMetrics()
    driver = driver or FailingDriver(retryable=True)
    sched = make_scheduler(make_config([text_model(trip_after=trip_after, **limits)],
                                       capacity=capacity), chain, driver, clock, metrics)
    return clock, chain, metrics, driver, sched


async def test_consecutive_backend_faults_withdraw_the_model_at_once():
    """The withdrawal is pushed from the job that tripped it — no second sweep."""
    clock, chain, metrics, driver, sched = _tripping(3)
    await sched.run_once()
    assert chain.published == [MODEL]
    await sched.join()
    assert metrics.fails == ["backend_exhausted"] * 3
    assert chain.published == []
    assert len(chain.snapshots) == 2
    assert _quotes(chain) == [{"model_id": 7, "sla": 3600, "rate_in": "0", "rate_out": "0"}]
    assert sched._breakers[MODEL].tripped()


async def test_concurrent_faults_push_one_withdrawal():
    clock, chain, metrics, driver, sched = _tripping(4, trip_after=2, capacity=4)
    await sched.run_once()
    await sched.join()
    assert len(metrics.fails) == 4
    assert len(chain.snapshots) == 2   # the book, then one withdrawal
    assert chain.published == []


async def test_a_tripped_model_polls_free_zero_and_claims_nothing():
    clock, chain, metrics, driver, sched = _tripping(3)
    await sched.run_once()
    await sched.join()
    fresh = open_text_job("after_trip")
    chain._jobs[fresh.job_id] = fresh
    chain.job_queries.clear()
    await sched.run_once()
    await sched.join()
    assert [q["free"] for q in chain.job_queries] == [0]   # still polled: the heartbeat
    assert metrics.model_free[MODEL] == 0
    assert fresh.state == "Open"
    assert driver.calls == 3


async def test_the_cooldown_relists_only_when_the_probe_passes():
    clock, chain, metrics, driver, sched = _tripping(3, trip_cooldown_s=60)
    await sched.run_once()
    await sched.join()
    assert chain.published == []

    driver._healthy = False
    clock.advance(61)
    await sched.sync_asks()
    assert chain.published == []            # cooldown over, probe still failing

    driver._healthy = True
    await sched.sync_asks()
    assert chain.published == [MODEL]


async def test_one_fault_after_relist_retrips_and_a_success_closes_it():
    clock, chain, metrics, driver, sched = _tripping(3, trip_cooldown_s=60)
    await sched.run_once()
    await sched.join()
    clock.advance(61)

    again = open_text_job("again")
    chain._jobs[again.job_id] = again
    await sched.run_once()                  # re-listed on trust, one more fault
    await sched.join()
    assert chain.published == []
    assert sched._breakers[MODEL].open_until == clock() + 60

    clock.advance(61)
    driver.fail = False
    ok = open_text_job("ok")
    chain._jobs[ok.job_id] = ok
    await sched.run_once()
    await sched.join()
    assert ok.state == "Settled"
    assert sched._breakers[MODEL].streak == 0
    assert chain.published == [MODEL]


class GoneDriver(FailingDriver):
    """A backend whose endpoint answers 404: every job fails at once, unretried."""

    async def run(self, job, input, *, timeout_s=None):
        self.calls += 1
        raise BackendGone("the backend answered HTTP 404: the endpoint is not there")


async def test_an_endpoint_that_is_gone_trips_the_breaker():
    """Each job is refused once and never retried, and three of them still take
    the model off the book — a 404 is the backend's fault, not the job's."""
    clock, chain, metrics, driver, sched = _tripping(3, driver=GoneDriver(), retries=4)
    await sched.run_once()
    await sched.join()
    assert driver.calls == 3
    assert metrics.fails == ["backend_gone"] * 3
    assert chain.published == []
    assert sched._breakers[MODEL].tripped()


async def test_every_failed_job_is_reported_under_its_model(monkeypatch):
    """The Sentry report rides the same call that counts the failure, so the two
    cannot disagree: one report per failed job, named by model and reason."""
    reported = []
    monkeypatch.setattr("vorqd.scheduler.report_job_failed",
                        lambda model, reason, job_id, detail="": reported.append((model, reason, job_id, detail)))
    clock, chain, metrics, driver, sched = _tripping(2, trip_after=0, driver=GoneDriver())
    await sched.run_once()
    await sched.join()
    assert metrics.fails == ["backend_gone"] * 2
    assert sorted(r[:2] for r in reported) == [(MODEL, "backend_gone")] * 2
    assert {r[2] for r in reported} == {j.job_id for j in chain._jobs.values()}
    assert all("HTTP 404" in r[3] for r in reported)


async def test_a_refusal_and_a_throttle_deadline_are_not_backend_faults():
    # A non-retryable refusal is the job's input, not the backend.
    clock, chain, metrics, driver, sched = _tripping(1, trip_after=1,
                                                     driver=FailingDriver(retryable=False))
    await sched.run_once()
    await sched.join()
    assert metrics.fails == ["backend_error"]
    assert chain.published == [MODEL]

    # A deadline spent waiting on this daemon's own quota is not the backend's fault
    # either, even though the one attempt it did make failed.
    clock = Clock()
    job = open_text_job(sla="2m")
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(
        make_config([text_model(sla="2m", retries=1, retry_backoff_s=10, rate_limit={"1h": 1},
                                trip_after=1)]),
        chain, FailingDriver(retryable=True), clock, metrics)
    await sched.run_once()
    await sched.join()
    assert metrics.fails == ["deadline_wait"]
    assert chain.published == [MODEL]


async def test_a_disabled_breaker_keeps_the_model_listed():
    clock, chain, metrics, driver, sched = _tripping(3, trip_after=0)
    await sched.run_once()
    await sched.join()
    assert len(metrics.fails) == 3
    assert chain.published == [MODEL]
    assert len(chain.snapshots) == 1


async def test_a_failed_withdrawal_push_is_retried_by_the_next_sweep():
    clock, chain, metrics, driver, sched = _tripping(3)
    real_push = chain.push_asks
    calls = {"n": 0}

    async def flaky_push(snapshot, signature):
        calls["n"] += 1
        if calls["n"] == 2:                 # the withdrawal, not the initial book
            raise httpx.ConnectError("node down")
        return await real_push(snapshot, signature)
    chain.push_asks = flaky_push

    await sched.run_once()
    await sched.join()
    assert len(chain.failed) == 3           # the job path was not derailed by the push
    assert chain.published == [MODEL]       # nothing landed, so the book stands
    await sched.run_once()
    assert chain.published == []


async def test_a_trip_landing_after_shutdown_does_not_republish_the_book():
    clock, chain, metrics, driver, sched = _tripping(3)
    await sched.sync_asks()
    await sched.stop()
    await sched.shutdown()
    assert chain.published == []
    n = len(chain.snapshots)
    for _ in range(3):
        sched._breakers[MODEL].record_failure()
    await sched._withdraw_tripped(MODEL)
    assert len(chain.snapshots) == n


# --- the ask book: one signed snapshot, pushed only when it changes -----------


def _quotes(chain):
    return chain.snapshots[-1][0]["quotes"]


async def test_the_snapshot_is_the_whole_book_in_ids_and_seconds():
    """Nothing on this wire is a name, and money is a USD decimal string.

    The registry's ask row is ``(modelId, sla, rateIn, rateOut)`` — four
    integers — so the model name and the ``"1h"`` window are resolved to numbers
    here. The rates cross as canonical USD strings and every other member as a
    JSON integer.
    """
    clock = Clock()
    chain = FakeNode([], clock)
    model = ModelConfig(
        model=MODEL,
        slas={"1h": SlaRate(rate_in="0.2", rate_out="0.6"),
              "24h": SlaRate(rate_in=None, rate_out="0.09")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
    )
    sched = make_scheduler(make_config([model]), chain, FakeDriver(), clock)
    await sched.sync_asks()

    (snapshot, _), = chain.snapshots
    assert set(snapshot) == {"provider_id", "signed_at", "quotes"}
    # The snapshot names its own publisher: `setAsks` compares it against
    # `idOf(signer)` and skips a mismatch, which is what stops a real operator
    # publishing prices under somebody else's id.
    assert snapshot["provider_id"] == PROVIDER_ID
    assert snapshot["quotes"] == [
        {"model_id": 7, "sla": 3600, "rate_in": "0.2", "rate_out": "0.6"},
        # A side the model does not meter is quoted at zero, not omitted.
        {"model_id": 7, "sla": 86400, "rate_in": "0", "rate_out": "0.09"},
    ]
    assert isinstance(snapshot["signed_at"], int)
    assert all(isinstance(q[k], int) for q in snapshot["quotes"] for k in ("model_id", "sla"))


async def test_the_snapshot_signature_recovers_to_the_operator_over_the_ask_registry():
    """The only authority the push has. The node holds no key that could price."""
    from vorqd import opsig

    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.sync_asks()

    snapshot, signature = chain.snapshots[-1]
    message = opsig.ask_snapshot_message(snapshot, CHAIN.decimals)
    # Pushed as USD strings, signed as the atomic members the registry verifies.
    assert (message["quotes"][0]["rateIn"], message["quotes"][0]["rateOut"]) == (200_000, 600_000)
    data = opsig.typed_data("AskSnapshot", message, CHAIN)
    assert opsig.recover(data, signature) == sched.ops.address
    assert data["domain"]["verifyingContract"] == CHAIN.ask_registry


async def test_a_withdrawal_rides_in_the_snapshot_as_rate_out_zero():
    """On chain the write is an upsert: a slot the snapshot omits keeps its price
    forever, so going quiet does not withdraw anything. Only ``rateOut == 0``
    deletes a slot, which means the withdrawal has to be inside the signed book."""
    clock = Clock()
    chain = FakeNode([], clock)
    driver = FakeDriver(healthy=True)
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)

    await sched.sync_asks()
    assert [q["rate_out"] for q in _quotes(chain)] == ["0.6"]

    driver._healthy = False
    await sched.sync_asks()
    withdrawn = _quotes(chain)
    assert withdrawn == [{"model_id": 7, "sla": 3600, "rate_in": "0", "rate_out": "0"}]
    # Not an empty snapshot: an empty book is silence, and silence leaves the
    # price standing.
    assert withdrawn != []


async def test_a_withdrawn_slot_is_not_withdrawn_again():
    """It is gone from chain state after one push, so re-stating it every sweep
    would pay gas to delete nothing."""
    clock = Clock()
    chain = FakeNode([], clock)
    driver = FakeDriver(healthy=True)
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)
    await sched.sync_asks()
    driver._healthy = False
    await sched.sync_asks()
    assert len(chain.snapshots) == 2

    driver._healthy = True
    await sched.sync_asks()
    assert len(chain.snapshots) == 3
    assert _quotes(chain) == [
        {"model_id": 7, "sla": 3600, "rate_in": "0.2", "rate_out": "0.6"}
    ]   # the republish carries the price alone; nothing lingers as a zero


async def test_an_unchanged_book_is_not_re_pushed():
    """Every push spends the node's gas and burns a strictly-increasing floor
    slot, so republishing an identical book pays to change nothing.

    And it is the *book* that is compared, not the health set: a rate edited in
    the config under an unchanged set of healthy models must still go out.
    """
    clock = Clock()
    chain = FakeNode([], clock)
    config = make_config([text_model()])
    sched = make_scheduler(config, chain, FakeDriver(), clock)

    await sched.sync_asks()
    await sched.sync_asks()
    await sched.sync_asks()
    assert len(chain.snapshots) == 1

    config.models[0].slas["1h"] = SlaRate(rate_in="0.2", rate_out="0.7")
    await sched.sync_asks()
    assert len(chain.snapshots) == 2
    assert _quotes(chain)[0]["rate_out"] == "0.7"


async def test_free_capacity_is_not_in_the_book():
    """The registry's ask row is a price and nothing else, and concurrency is a
    different contract's op. A book that carried it would need a signed
    transaction every time a slot filled."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.sync_asks()
    for quote in _quotes(chain):
        assert set(quote) == {"model_id", "sla", "rate_in", "rate_out"}


async def test_shutdown_pushes_an_all_withdrawn_book():
    """The ask registry has no TTL, so a daemon that simply stopped would keep
    being matched with work it is no longer running."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.sync_asks()

    await sched.shutdown()
    assert _quotes(chain) == [{"model_id": 7, "sla": 3600, "rate_in": "0", "rate_out": "0"}]
    assert chain.published == []


async def test_shutdown_before_anything_was_published_pushes_nothing():
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.shutdown()
    assert chain.snapshots == []


@pytest.mark.parametrize("rate, needle", [
    ("0.0000001", "more than 6 fraction digits"),   # finer than the token's decimals
    (str(2**128), "wider than uint128"),             # atomic past the signed member
])
async def test_a_rate_the_token_cannot_hold_is_refused_not_rounded(rate, needle):
    """The token's ``decimals`` bound a configured rate once the chain context is
    read. A finer rate is refused, never rounded: a rounded ask is a price the
    operator did not write."""
    clock = Clock()
    chain = FakeNode([], clock)
    model = ModelConfig(
        model=MODEL,
        slas={"1h": SlaRate(rate_in=None, rate_out=rate)},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
    )
    sched = make_scheduler(make_config([model]), chain, FakeDriver(), clock)
    with pytest.raises(ConfigError, match=needle):
        await sched.sync_asks()
    assert chain.snapshots == []


async def test_a_book_over_the_registry_bound_is_refused_rather_than_dropped():
    """``AskRegistry.MAX_QUOTES`` is 64 and the contract **skips** a snapshot over
    it — silently, so the whole book would vanish with no error anywhere."""
    from vorqd.scheduler import MAX_QUOTES

    clock = Clock()
    windows = {f"{n}m": SlaRate(rate_in=None, rate_out="0.6") for n in range(1, MAX_QUOTES + 2)}
    model = ModelConfig(
        model=MODEL, slas=windows,
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
    )
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([model]), chain, FakeDriver(), clock)
    with pytest.raises(ConfigError, match=str(MAX_QUOTES)):
        await sched.sync_asks()
    assert chain.snapshots == []


# --- startup: identity discovery, box-key check, allowed-models gate ----------

import logging  # noqa: E402

from vorqd._crypto import BoxCipher as _BoxCipher  # noqa: E402

_LOCAL_BOX_PUB = _BoxCipher(_TEST_BOX_KEY).public_key


def _on_record(box_key: str | None = _LOCAL_BOX_PUB, **rest):
    """A provider row as the node serves it: ``box_key`` is 0x hex of a bytes32,
    and ``allow_all_models`` — not an empty list — is what "no restriction" means."""
    rec = {"box_key": None if box_key is None else "0x" + box_key,
           "allow_all_models": True, "allowed_models": []}
    rec.update(rest)
    return rec


def _startup_scheduler(chain, config, clock):
    drivers = {m.model: FakeDriver() for m in config.models}
    return Scheduler(config, chain, FakeCoord(), drivers, FakeMetrics(), clock=clock,
                     blobs=BlobResolver([SOURCE]))


async def test_startup_requests_capacity_and_publishes_asks():
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record())
    sched = _startup_scheduler(chain, make_config([text_model()]), clock)
    await sched.startup()
    assert chain.capacity_requested == 4
    assert chain.published == ["deepseek-ai/deepseek-v4-pro:fp8"]


async def test_startup_binds_the_catalog_before_the_first_poll():
    """Q15: nothing in a sweep can be spelled without it.

    The book is polled by model id and the ask book quotes ids and seconds, so a
    daemon that reached a poll with an unbound catalog could name neither. The
    binding is handed to the node itself, which is what makes ``list_open_jobs``
    able to translate a row back into the operator's own model name.
    """
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record())
    sched = _startup_scheduler(chain, make_config([text_model()]), clock)
    await sched.startup()

    assert chain.bound is not None
    assert chain.bound.model_id(MODEL) == MODEL_IDS[MODEL]
    assert chain.bound.sla_secs("1h") == 3600


async def test_a_configured_model_the_catalog_does_not_carry_stops_the_daemon():
    """Never a silent skip: the operator would run at a capacity they did not choose."""
    from vorqd.errors import UnknownModel

    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record())
    sched = _startup_scheduler(chain, make_config([text_model("org/never-listed:fp8")]), clock)
    with pytest.raises(UnknownModel, match="does not carry it"):
        await sched.startup()
    assert chain.ops == [] and chain.snapshots == []   # nothing was signed


async def test_startup_box_key_mismatch_is_fatal():
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record("00" * 32))   # not our key
    sched = _startup_scheduler(chain, make_config([text_model()]), clock)
    with pytest.raises(ConfigError, match="box public key"):
        await sched.startup()
    assert chain.capacity_requested is None   # never got past the check


async def test_the_record_key_is_compared_as_bytes_and_not_as_text():
    """The chain holds a bytes32 and the node serves ``0x…``; nacl hands out bare
    hex. One key, two spellings — comparing the strings would make every
    confidential boot wait forever for a record that already matched."""
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record())
    chain.provider_rec["box_key"] = "0x" + _LOCAL_BOX_PUB.upper()
    await _startup_scheduler(chain, make_config([text_model()]), clock).startup()
    assert chain.capacity_requested == 4     # got past the check


async def test_startup_excludes_models_not_in_allowed_set(caplog):
    clock = Clock()
    allowed_model = text_model("model-a:fp8")
    forbidden_model = text_model("model-b:fp8")
    # The permitted set is model **ids**, and the flag is what makes it a
    # restriction at all.
    chain = FakeNode([], clock, provider_rec=_on_record(
        allow_all_models=False, allowed_models=[MODEL_IDS["model-a:fp8"]]))
    sched = _startup_scheduler(chain, make_config([allowed_model, forbidden_model]), clock)
    with caplog.at_level(logging.WARNING):
        await sched.startup()
    assert chain.published == ["model-a:fp8"]   # model-b excluded from asks
    assert any("model-b:fp8" in r.message for r in caplog.records)   # warning logged


async def test_an_empty_allowed_list_with_the_flag_set_is_no_restriction():
    """The flag is the authority, and reading the list alone inverts the meaning.

    ``allow_all_models`` true beside an empty ``allowed_models`` is how the
    registry spells "everything" — a daemon that read the list would withdraw
    every ask it has and look perfectly healthy doing it.
    """
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record())
    assert chain.provider_rec["allowed_models"] == []
    sched = _startup_scheduler(chain, make_config([text_model()]), clock)
    await sched.startup()
    assert sched._allowed is None
    assert chain.published == ["deepseek-ai/deepseek-v4-pro:fp8"]


# --- content-addressed payload path (resolve, open the envelope, seal result) --

async def test_task_bytes_are_resolved_and_verified_before_the_claim():
    # The order the whole claim path exists to get right. The blob surface is
    # public, so the fetch costs nothing but a round trip — and doing it first is
    # what makes it possible to refuse bytes that do not name this job while
    # refusing is still free. A claim spends the client's order; it must never be
    # spent on a payload that can never be opened.
    clock = Clock()
    job = open_text_job("gated")
    chain = FakeNode([job], clock)
    seen = {}

    class RecordingSource:
        async def get(self, cid):
            seen["state"] = job.state
            seen["ops"] = list(chain.ops)
            return await SOURCE.get(cid)

    config = make_config([text_model()])
    sched = Scheduler(config, chain, FakeCoord(), {m.model: FakeDriver() for m in config.models},
                      FakeMetrics(), clock=clock, blobs=BlobResolver([RecordingSource()]),
                      escrow=FakeEscrow())
    await sched.run_once()
    await sched.join()
    assert seen["state"] == "Open"     # fetched before the claim...
    assert seen["ops"] == []           # ...and nothing had been signed yet
    assert len(chain.settled) == 1     # and the honest job still ran end to end


async def test_a_designated_bid_opens_with_this_daemons_own_box_key():
    # The wrap is sealed to the key this provider published, so the container
    # opens in-process: no escrow, no network, nobody else holding the DEK.
    clock = Clock()
    box = BoxCipher.generate()          # the daemon's box key
    client = BoxCipher.generate()       # the key the client put in its envelope
    job_id, cid = pin_task({"input": "hi"}, result_key=client.public_key, recipient=box.public_key)
    job = EvmJob(job_id=job_id, model=MODEL, state="Open", sla="1h", created_at=1000,
                 owner=OWNER, rate_in=200_000, rate_out=600_000, units_in=10, units_out=128,
                 task_cid=cid, designated=PROVIDER_ID)

    chain = FakeNode([job], clock)
    config = make_config([text_model()])
    drivers = {m.model: FakeDriver() for m in config.models}
    escrow = FakeEscrow()
    sched = Scheduler(config, chain, FakeCoord(), drivers, FakeMetrics(), clock=clock,
                      cipher=box, blobs=BlobResolver([SOURCE]), escrow=escrow)
    await sched.run_once()
    await sched.join()

    assert len(chain.settled) == 1
    assert escrow.calls == []   # the escrow was never asked, and never could be
    env = json.loads(base64.b64decode(chain.settled[0][1]))
    assert env["enc"] == "vorq-sealed-v1"
    # The result opens with the envelope's key — no result_key ever rode the order.
    assert json.loads(client.decrypt(base64.b64decode(env["ciphertext"])))


async def test_an_open_bid_opens_with_the_dek_the_escrow_releases():
    # An open bid's wrap is sealed to the escrow, so the daemon asks for the DEK
    # once its claim is on record. Nothing about the claim itself delivers a key.
    clock = Clock()
    job = open_text_job("broad")
    chain = FakeNode([job], clock)
    escrow = FakeEscrow()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, escrow=escrow)
    await sched.run_once()
    await sched.join()

    assert len(chain.settled) == 1
    (job_id, seed_wrap, ct_hash), = escrow.calls
    assert job_id == job.job_id
    assert len(seed_wrap) == 80 and len(ct_hash) == 32   # the wrap and the digest, never the payload


@pytest.mark.parametrize("code", ["escrow_key_lost", "not_claimed", "unseal_failed"])
async def test_an_escrow_refusal_gives_the_claim_back_under_its_own_code(code):
    # The escrow's refusal vocabulary is frozen and attributable: it is reported
    # verbatim so the client is refunded now and an operator can page on the
    # difference between a lost key and a wrong wallet.
    clock = Clock()
    job = open_text_job(f"refused_{code}")
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics,
                           escrow=FakeEscrow(refuse=code))
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert driver.last_timeout_s is None      # nothing ran
    assert chain.failed == [job.job_id]      # the reason does not ride the wire...
    assert metrics.fails == [code]           # ...it is metered under the escrow's own code


async def test_an_open_bid_without_a_release_client_is_refused_not_guessed():
    # No configured escrow means no way to obtain the DEK — and no fallback to a
    # key delivered by anything else. The claim goes back immediately.
    clock = Clock()
    job = open_text_job("no_escrow")
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics,
                           escrow=None)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert chain.failed == [job.job_id]
    # Refused under the escrow's OWN code, so an operator reading the metric sees
    # the same word the escrow would have said.
    assert metrics.fails == ["escrow_unavailable"]


async def test_an_open_order_is_designated_zero_and_is_claimed_not_skipped():
    """Q22, and the bug it exists to prevent.

    An open order's ``designated`` is ``0``, never null. A guard written as
    ``designated is not None and designated != mine`` treats every open job on
    the book as somebody else's pin and claims nothing at all — a daemon that
    polls, logs nothing, publishes asks and earns zero.
    """
    clock = Clock()
    job = open_text_job("open_sentinel")
    assert job.designated == 0 and EvmJob("i", "m", "Open", "1h", 0).designated == 0
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert [payload["job_id"] for payload in chain.pushed("claim")] == [job.job_id]
    assert len(chain.settled) == 1


async def test_escrow_key_lost_hands_the_job_back_at_once_inside_the_grace_window():
    """The one refusal the grace window was written for.

    The key is gone; no amount of waiting brings it back, and a provider that
    says so immediately pays no penalty. So the fail op goes out on the spot,
    with an ``issued_at`` inside ``FAIL_GRACE`` of the claim.
    """
    clock = Clock()
    job = open_text_job("key_lost")
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics,
                           escrow=FakeEscrow(refuse="escrow_key_lost"))
    await sched.run_once()
    await sched.join()

    (claim,) = chain.pushed("claim")
    (fail,) = chain.pushed("fail")
    assert chain.failed == [job.job_id] and metrics.fails == ["escrow_key_lost"]
    grace = make_config([text_model()]).provider.fail_grace_s
    assert grace == FAIL_GRACE_SECONDS == 300   # mirrors JobRegistry.FAIL_GRACE
    assert fail["issued_at"] - claim["issued_at"] < grace
    assert sched._within_fail_grace(job)


async def test_the_advisory_gate_stops_a_claim_that_would_not_land():
    """`simulate_claim` reads chain state at `latest`, so its answer is
    meaningful in exactly the window a daemon is deciding in. A `no` costs one
    read; signing anyway would cost a refused op and the round trip under it."""
    clock = Clock()
    job = open_text_job("at_capacity")
    chain = FakeNode([job], clock, simulate=ClaimSimulation(ok=False, reason="AtCapacity"))
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.simulations == [(job.job_id, sched.ops.address)]
    assert chain.ops == []          # nothing was signed
    assert job.state == "Open"
    assert metrics.claims == 0


async def test_a_refused_claim_is_a_skip_and_never_a_failure():
    """The node runs the authoritative simulate of the exact signed op and
    relays only on ok, so a refusal here means the job was never ours: a lost
    race, or a client wallet drained between the gate and the op."""
    clock = Clock()
    job = open_text_job("refused_claim")
    chain = FakeNode([job], clock, refuse_claim="NotOpen")
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.pushed("claim") and chain.failed == []   # tried, refused, dropped
    assert metrics.claims == 0 and metrics.fails == []    # not an error, not metered as one
    assert driver.last_timeout_s is None


async def test_a_claim_that_mined_reverted_is_never_run():
    """The pre-relay simulate is authoritative only at the instant it runs.

    A client whose balance drops between that call and the broadcast leaves a
    claim that simulates clean and reverts on chain: the escrow pull is the last
    thing `claim` does and a failed pull rolls the whole transaction back. The
    node still answers a receipt, so the daemon has to read
    it — the job is still Open, the escrow was never funded, and serving it would
    be work given away.
    """
    clock = Clock()
    job = open_text_job("reverted_claim")
    chain = FakeNode([job], clock, revert_claim=True)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.pushed("claim")                           # signed, relayed, reverted
    assert job.state == "Open"                             # nothing was stamped locally
    assert metrics.claims == 0 and metrics.fails == []     # a skip, not a failure
    assert driver.last_timeout_s is None                   # the backend never ran
    assert chain.settled == [] and chain.failed == []      # and nothing was resolved


async def test_a_settle_that_mined_reverted_hands_the_job_back():
    """A reverted settle leaves the job Claimed and the escrow held.

    Nothing in the sweep will touch that job again — the work is done and the
    window is spent — so leaving it would strand the client's money until SLA
    expiry and hold a capacity slot the registry still counts as in use. It gets
    the same treatment as a settle that never reached the chain: reported, so the
    refund happens now.
    """
    clock = Clock()
    job = open_text_job("reverted_settle")
    chain = FakeNode([job], clock, revert_settle=True)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.pushed("settle")                    # it was signed and sent
    assert chain.settled == []                       # and it did not land
    assert chain.failed == [job.job_id]              # so the job was handed back
    assert metrics.fails == ["settle_reverted"]
    assert metrics.settles == 0                      # never metered as delivered


async def test_a_failure_report_that_mined_reverted_is_not_logged_as_a_refund(caplog):
    """A reverted `fail` refunded nobody, and the daemon has to say so.

    The refund degrades to the SLA-expiry reclaim exactly as it does when the
    report never reaches the chain, and the only thing separating the two is
    whether this daemon noticed.
    """
    clock = Clock()
    job = open_text_job("reverted_fail")
    chain = FakeNode([job], clock, revert_settle=True, revert_fail=True)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    with caplog.at_level(logging.WARNING):
        await sched.run_once()
        await sched.join()

    assert chain.pushed("fail")                      # the report was signed and sent
    assert chain.failed == []                        # and refunded nobody
    assert any("SLA-expiry reclaim" in r.getMessage() for r in caplog.records)


async def test_the_designated_path_derives_the_key_and_never_uses_the_seed(caplog):
    """Q3 on the path with no coordinator in it.

    The wrap seals a seed; the working key is HKDF of it under the job's owner.
    A daemon that used the sealed bytes directly would open nothing — and a
    container welded into a job with a different owner opens nothing either,
    which is the whole of the wrap-lifting defence on this path.
    """
    clock = Clock()
    box = BoxCipher.generate()
    seed = bytes(range(32))
    job_id, cid = pin_task({"input": "hi"}, recipient=box.public_key, seed=seed)
    job = EvmJob(job_id=job_id, model=MODEL, state="Open", sla="1h", created_at=1000,
                 owner=OWNER, rate_in=200_000, rate_out=600_000, units_in=10, units_out=128,
                 task_cid=cid, designated=PROVIDER_ID)
    chain = FakeNode([job], clock)
    config = make_config([text_model()])
    escrow = FakeEscrow()
    sched = Scheduler(config, chain, FakeCoord(), {m.model: FakeDriver() for m in config.models},
                      FakeMetrics(), clock=clock, cipher=box, blobs=BlobResolver([SOURCE]),
                      escrow=escrow)
    await sched.run_once()
    await sched.join()

    assert len(chain.settled) == 1
    assert escrow.calls == []                    # in-process, no third party
    # The key that opened it is the derived one, and it is not the seed.
    container = SOURCE._blobs[cid]
    dek = derive_dek(seed, OWNER)
    assert dek != seed
    assert open_dek(container[MIN_CONTAINER_BYTES:], dek)   # the derived key opens the payload
    with pytest.raises(Exception):
        open_dek(container[MIN_CONTAINER_BYTES:], seed)     # the sealed bytes do not


async def test_a_designated_container_sealed_under_another_owner_does_not_open():
    clock = Clock()
    box = BoxCipher.generate()
    stranger = "0x" + "44" * 20
    job_id, cid = pin_task({"input": "hi"}, recipient=box.public_key, sealed_to_owner=stranger)
    job = EvmJob(job_id=job_id, model=MODEL, state="Open", sla="1h", created_at=1000,
                 owner=OWNER, rate_in=200_000, rate_out=600_000, units_in=10, units_out=128,
                 task_cid=cid, designated=PROVIDER_ID)
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    config = make_config([text_model()])
    sched = Scheduler(config, chain, FakeCoord(), {m.model: FakeDriver() for m in config.models},
                      metrics, clock=clock, cipher=box, blobs=BlobResolver([SOURCE]),
                      escrow=FakeEscrow())
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert metrics.fails == ["undecryptable"]


async def test_the_open_path_does_not_derive_what_the_escrow_already_derived():
    """`POST /release` answers the DEK, not the seed: the coordinator derived it
    against the owner it read from chain. Deriving again here would produce a
    key of a key and decrypt nothing — and this is the assertion that catches
    it, because the fake escrow derives exactly as the real one does."""
    clock = Clock()
    seed = bytes(range(1, 33))
    job = open_text_job("no_double_derive")
    chain = FakeNode([job], clock)
    escrow = FakeEscrow()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, escrow=escrow)
    await sched.run_once()
    await sched.join()

    assert len(chain.settled) == 1
    (_, wrap, _), = escrow.calls
    released = derive_dek(ESCROW.decrypt(wrap), OWNER)
    assert released != derive_dek(released, OWNER)   # the mistake, spelled out
    assert seed != released


async def test_scheduler_refuses_mis_owned_envelope():
    # The envelope names a victim; the job is owned by the attacker who paid for
    # it. Running it would let the attacker buy an answer to somebody else's
    # prompt under their own escrow (D8) — so the claim is given back, not run.
    clock = Clock()
    victim = "0x" + "22" * 20
    attacker = "0x" + "33" * 20
    # The attacker's own commitment over the victim's envelope: a well-formed
    # job id, and bytes that open — `sealed_to_owner` derives the key under the
    # attacker so the container genuinely decrypts, and the refusal is the
    # envelope's owner and nothing else.
    job_id, cid = pin_task({"input": "secret"}, owner=victim, sealed_to_owner=attacker)
    job = EvmJob(job_id=job_id_of(attacker, SOURCE._blobs[cid]), model=MODEL, state="Open",
                 sla="1h", created_at=1000, owner=attacker, rate_in=200_000, rate_out=600_000,
                 units_in=10, units_out=128, task_cid=cid)

    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert driver.last_timeout_s is None            # the backend never ran
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["owner_mismatch"]


async def test_a_job_without_a_task_cid_is_never_claimed():
    # Nothing can serve bytes for a job that names none, and the fetch runs
    # before the claim — so this one is simply never claimed. There is no claim
    # to give back and no fail op to push: reporting a job we do not hold would
    # be a lie the registry refuses anyway.
    clock = Clock()
    job = open_text_job("no_cid")
    job.task_cid = None
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert driver.last_timeout_s is None
    assert chain.ops == []          # not one signed op
    assert job.state == "Open"      # still on the book for a provider that can serve it


async def test_a_container_sealed_under_another_owner_does_not_decrypt():
    """The wrap-lifting attack, end to end, on the path that carries the job.

    The attacker lifts a container whose key was derived under the victim,
    welds it into a commitment under their own address and posts it. The job id
    is honest, the container is well formed, the commitment reproduces — every
    check over public data passes, because every field really is the attacker's.
    What refuses it is the derivation: the DEK the daemon derives is bound to
    the job's owner, so the ciphertext does not open and no plaintext is ever
    produced.
    """
    clock = Clock()
    victim = "0x" + "22" * 20
    attacker = "0x" + "33" * 20
    _, cid = pin_task({"input": "the victim's prompt"}, owner=victim)
    lifted = SOURCE._blobs[cid]
    job = EvmJob(job_id=job_id_of(attacker, lifted), model=MODEL, state="Open", sla="1h",
                 created_at=1000, owner=attacker, rate_in=200_000, rate_out=600_000,
                 units_in=10, units_out=128, task_cid=cid)
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert driver.last_timeout_s is None
    assert metrics.fails == ["undecryptable"]
    assert chain.failed == [job.job_id]


async def test_settle_is_one_op_carrying_the_bytes_and_naming_no_cid(caplog):
    # One call: the sealed bytes ride inside the op's payload, the node pins them
    # and mints the name, and the name comes back on the answer. The daemon never
    # learns a CID before the node answers and never signs one.
    clock = Clock()
    client = BoxCipher.generate()
    job = open_text_job("settle_shape", result_key=client.public_key)
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    with caplog.at_level(logging.INFO):
        await sched.run_once()
        await sched.join()

    (payload,) = chain.pushed("settle")
    assert set(payload) == {"job_id", "completion_tok", "result", "issued_at"}
    assert payload["issued_at"] == int(clock())
    # Nothing in the op, and nothing in what its signature covers, is a name.
    assert not any("cid" in key.lower() for key in payload)
    assert isinstance(payload["result"], str)   # flat JSON: base64, not a file part
    assert chain.uploaded == []                 # small enough to ride inline
    sealed = json.loads(base64.b64decode(payload["result"]))
    assert sealed["enc"] == "vorq-sealed-v1"
    # The bytes the client will fetch and open with the key from its own envelope.
    # The backend's own object, plus the stamp naming the line it answers.
    opened = json.loads(client.decrypt(base64.b64decode(sealed["ciphertext"])))
    assert opened["vorq"] == {"job_id": job.job_id}
    assert {k: v for k, v in opened.items() if k != "vorq"} == {"choices": []}
    # The minted name is the node's, read off the answer and logged with the settle.
    assert chain.minted == ["bafkrei-minted-1"]
    settled = [r for r in caplog.records if r.getMessage() == "settled"]
    assert [r.result_cid for r in settled] == ["bafkrei-minted-1"]


async def test_a_result_at_the_inline_bound_settles_inline():
    """Exactly ``INLINE_MAX_BYTES`` still rides as base64 — the bound is inclusive."""
    clock = Clock()
    job = open_text_job("inline_bound")
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    big = b"x" * INLINE_MAX_BYTES

    async def fake_build_result(job, normalized, result_key, custom_id=None):
        return big, 1

    sched._build_result = fake_build_result
    await sched.run_once()
    await sched.join()

    (payload,) = chain.pushed("settle")
    assert payload["result"] == base64.b64encode(big).decode()
    assert "result_cid" not in payload
    assert chain.uploaded == []


async def test_a_result_one_byte_over_the_bound_is_uploaded_first():
    """One byte past the bound and the daemon uploads before it settles, then
    references the upload rather than sending the bytes twice."""
    clock = Clock()
    job = open_text_job("over_bound")
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    big = b"x" * (INLINE_MAX_BYTES + 1)

    async def fake_build_result(job, normalized, result_key, custom_id=None):
        return big, 1

    sched._build_result = fake_build_result
    await sched.run_once()
    await sched.join()

    assert chain.uploaded == [("result", big)]
    (payload,) = chain.pushed("settle")
    assert payload["result_cid"] == "cid-uploaded-1"
    assert "result" not in payload


async def test_the_settle_is_stamped_after_the_upload_not_before_it():
    """``issued_at`` is the op's freshness, and the node refuses one outside ±600 s.

    Stamped before the upload, the whole window is spent on an operation the
    daemon does *before* it starts the clock: a large result on a slow uplink
    arrives already stale, and the 409 lands in the ``OpRefused`` branch, which
    logs and meters but does not hand the job back — so the client's escrow idles
    to SLA expiry and the provider takes a missed-SLA hit for work it completed
    and delivered.

    The upload here takes eleven minutes, which is past the node's window and
    well inside what 150 MiB on a 2 Mbps link costs.
    """
    clock = Clock()
    job = open_text_job("slow_upload")

    class SlowUploadNode(FakeNode):
        async def upload_file(self, purpose, content, filename="result"):
            clock.advance(660)
            return await super().upload_file(purpose, content, filename)

    chain = SlowUploadNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    big = b"x" * (INLINE_MAX_BYTES + 1)

    async def fake_build_result(job, normalized, result_key, custom_id=None):
        return big, 1

    sched._build_result = fake_build_result
    await sched.run_once()
    await sched.join()

    assert chain.uploaded == [("result", big)]
    (payload,) = chain.pushed("settle")
    # The stamp the node judges, read against the clock as it stood when the op
    # was actually sent — not as it stood before the bytes started moving.
    assert payload["issued_at"] == int(clock.t)

    # And the *signature* binds that same stamp. The body and the signature are
    # built from one local in one expression, so they cannot diverge — but the
    # node checks the signature, not the field, so a change that reintroduced a
    # second clock read would be invisible to the assertion above alone.
    (_, _, signature) = [op for op in chain.ops if op[0] == "settle"][0]
    ctx = await chain.chain_context()
    assert signature == sched.ops.sign_settle(job.job_id, payload["completion_tok"],
                                              payload["issued_at"], ctx)
    assert signature != sched.ops.sign_settle(job.job_id, payload["completion_tok"],
                                              payload["issued_at"] - 660, ctx)


async def test_a_413_on_the_upload_is_reported_as_result_too_large():
    """The blob ceiling can refuse the upload itself, not just the op door — the
    same size condition gets the same label either way."""
    clock = Clock()
    job = open_text_job("upload_413")

    class OversizeUploadNode(FakeNode):
        async def upload_file(self, purpose, content, filename="result"):
            req = httpx.Request("POST", "http://x/v1/files")
            raise httpx.HTTPStatusError("too large", request=req,
                                        response=httpx.Response(413, request=req))

    chain = OversizeUploadNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    big = b"x" * (INLINE_MAX_BYTES + 1)

    async def fake_build_result(job, normalized, result_key, custom_id=None):
        return big, 1

    sched._build_result = fake_build_result
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["settle_result_too_large"]
    assert job.state == "Cancelled"     # settled by nobody, refunded now


async def test_an_upload_answer_without_a_cid_hands_the_job_back():
    """A 2xx upload with no ``vorq.cid`` leaves nothing to reference the result
    by, and there is no name to invent. It is a delivery failure like a settle
    that was refused with a status: the job is reported failed so the client is
    refunded now, rather than left claimed until the reclaim.

    The untyped form of this — ``resp.json()["vorq"]["cid"]`` raising
    ``KeyError`` — escaped every handler around the settle, which is the one
    outcome they exist to prevent.
    """
    clock = Clock()
    job = open_text_job("nameless_upload")

    class NamelessUploadNode(FakeNode):
        async def upload_file(self, purpose, content, filename="result"):
            raise UploadInvalid("POST /v1/files answered no vorq.cid")

    chain = NamelessUploadNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    big = b"x" * (INLINE_MAX_BYTES + 1)

    async def fake_build_result(job, normalized, result_key, custom_id=None):
        return big, 1

    sched._build_result = fake_build_result
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["settle_upload_invalid"]
    assert job.state == "Cancelled"     # settled by nobody, refunded now


async def test_a_settle_that_answers_without_a_cid_is_a_warning_not_a_failure(caplog):
    # The job did settle — the bytes are on chain and the client is charged — and
    # nothing here can reconstruct a name the node did not send. So it is said out
    # loud and never invented.
    clock = Clock()
    job = open_text_job("nameless")

    class NamelessNode(FakeNode):
        async def push_op(self, op, payload, signature):
            answer = await super().push_op(op, payload, signature)
            return OpResult(answer.tx_hash, answer.status, answer.block_number) \
                if op == "settle" else answer

    chain = NamelessNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    with caplog.at_level(logging.WARNING):
        await sched.run_once()
        await sched.join()

    assert len(chain.settled) == 1 and metrics.settles == 1   # it settled
    assert metrics.fails == []                                # and it did not fail
    assert any("without a result_cid" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("status, path, reason", [
    (413, "/evm/ops", "settle_result_too_large"),  # past the node's blob ceiling
    (500, "/evm/ops", "settle_http_500"),          # anything else the op door refuses
    (500, "/auth/nonce", "session_http_500"),      # the session broke; the settle never went
])
async def test_a_refused_settle_is_reported_rather_than_left_to_expire(status, path, reason):
    # A transport refusal is not the chain's verdict: it leaves a claimed job the
    # daemon has already paid to run and will never revisit. Reporting it refunds
    # the client's escrow now; staying silent would lock it until the SLA expires.
    clock = Clock()
    job = open_text_job(f"refused_{status}")

    class RefusingNode(FakeNode):
        async def push_op(self, op, payload, signature):
            if op != "settle":
                return await super().push_op(op, payload, signature)
            # `path` is the surface that actually refused: a settle needs a session,
            # so a handshake failure surfaces here without the settle being sent.
            req = httpx.Request("POST", f"http://x{path}")
            raise httpx.HTTPStatusError("refused", request=req,
                                        response=httpx.Response(status, request=req))

    chain = RefusingNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert chain.failed == [job.job_id]
    assert metrics.fails == [reason]
    assert job.state == "Cancelled"                   # settled by nobody, refunded now


async def test_a_settle_that_never_reaches_the_network_is_reported_too():
    # No status to read: connect refused, DNS, read timeout. The job is claimed and
    # this daemon will never revisit it, so the report goes out on the same reasoning
    # as a refused settle — if the blip was momentary the client is refunded now.
    clock = Clock()
    job = open_text_job("unreachable")

    class UnreachableNode(FakeNode):
        async def push_op(self, op, payload, signature):
            if op != "settle":
                return await super().push_op(op, payload, signature)
            raise httpx.ConnectError("connection refused",
                                     request=httpx.Request("POST", "http://x/evm/ops"))

    chain = UnreachableNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["settle_transport_error"]


# --- confidential boot: ephemeral key publication + provisioning wait ---------


def _confidential_config():
    """A confidential-shaped config: no operator box key — it is ephemeral, generated
    in guest memory each boot."""
    model = ModelConfig(
        model="org/e2ee-model:fp8",
        slas={"1h": SlaRate(rate_in="0.2", rate_out="0.6")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
        confidential=True,
    )
    return make_config([model], box_key=None, poll_interval_s=0.01)


def _confidential_scheduler(chain, clock, ident, config=None):
    config = config or _confidential_config()
    drivers = {m.model: FakeDriver() for m in config.models}
    return Scheduler(config, chain, FakeCoord(), drivers, FakeMetrics(), clock=clock,
                     cipher=ident.cipher, evidence=ident.evidence, blobs=BlobResolver([SOURCE]))


def _boot_identity():
    from vorqd.tee.agent import MockAttestationAgent, boot_identity
    return boot_identity("0x" + "ab" * 20, MockAttestationAgent())


async def test_confidential_startup_publishes_ephemeral_key_and_waits():
    """The boot key never matches the record (it was generated seconds ago): push
    the signed identity op, wait for the record to reflect it, then proceed — no
    mismatch-fatal."""
    clock = Clock()
    chain = FakeNode([], clock)
    chain.delay_update_rounds = 1   # the record reflects the op only after a poll
    ident = _boot_identity()
    sched = _confidential_scheduler(chain, clock, ident)

    await sched.startup()

    # Both fields in one op: setIdentity writes the key and the evidence together,
    # so a key published without its evidence would replace the stored blob.
    (identity,) = chain.pushed("set_identity")
    assert set(identity) == {"box_key", "evidence", "issued_at"}
    assert identity["box_key"] == "0x" + ident.cipher.public_key
    # Evidence is opaque BYTES on chain, so it travels as hex — never as a JSON
    # object beside the signature, which would be a second reading of the field
    # the signature covers.
    assert identity["evidence"].startswith("0x")
    assert json.loads(bytes.fromhex(identity["evidence"][2:])) == ident.evidence
    assert chain.provider_rec["box_key"] == "0x" + ident.cipher.public_key
    # Two polls happen without the loop running at all (once before the op, once
    # straight after it), so a third proves the wait loop's body executed.
    assert chain.provider_polls >= 3
    assert chain.capacity_requested == 4      # ...and startup then proceeded normally
    assert chain.published == ["org/e2ee-model:fp8"]


async def test_the_boot_pushes_identity_and_capacity_exactly_once():
    clock = Clock()
    chain = FakeNode([], clock)
    ident = _boot_identity()
    await _confidential_scheduler(chain, clock, ident).startup()

    assert [op for op, _, _ in chain.ops] == ["set_identity", "request_capacity"]
    assert len(chain.snapshots) == 1


async def test_issued_at_is_monotonic_across_a_reboot_inside_one_second():
    """The case a naive ``int(time.time())`` fails, and the reason it matters.

    Both ProviderRegistry floors are strict — ``issuedAt <= lastIdentityAt[id]``
    reverts ``StaleOp`` — and nothing off chain can read them: the daemon holds
    no RPC and the provider record does not carry them. So a process that
    restarts inside the same wall-clock second as its own last op signs the
    identical timestamp and is refused, and for a confidential boot that means a
    box key generated seconds ago that never reaches the record while every
    client still seals to the key it replaced. The chain's refusal is the only
    signal there is: bump, **re-sign**, push again.
    """
    clock = Clock()
    chain = FakeNode([], clock)
    await _confidential_scheduler(chain, clock, _boot_identity()).startup()
    first = {op: p["issued_at"] for op, p, _ in chain.ops}

    # A second boot, same node, same second on the clock, a new ephemeral key.
    reboot = _boot_identity()
    await _confidential_scheduler(chain, clock, reboot).startup()

    assert clock.t == 1000.0                     # the wall clock has not moved at all
    landed = [(op, p["issued_at"], sig) for op, p, sig in chain.ops]
    identity = [entry for entry in landed if entry[0] == "set_identity"]
    capacity = [entry for entry in landed if entry[0] == "request_capacity"]
    # The refused attempts are recorded too, so the retry is visible: each op was
    # attempted at the stale second and re-signed one second later.
    assert [t for _, t, _ in identity] == [first["set_identity"], first["set_identity"],
                                           first["set_identity"] + 1]
    assert [t for _, t, _ in capacity][-1] > first["request_capacity"]
    # Re-signed, not re-sent: issuedAt is inside the struct hash, so the same
    # signature under a new timestamp recovers a stranger.
    assert identity[-1][2] != identity[-2][2]
    # ...and the record ends up carrying the SECOND boot's key.
    assert chain.provider_rec["box_key"] == "0x" + reboot.cipher.public_key


async def test_a_stale_snapshot_is_re_signed_at_a_later_second():
    """The ask book has the identical floor (``AskRegistry.lastSignedAt``)."""
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record())
    config = make_config([text_model()])
    await _startup_scheduler(chain, config, clock).startup()
    first = int(chain.snapshots[-1][0]["signed_at"])

    # A reboot in the same second: the node's floor already stands at `first`.
    sched = _startup_scheduler(chain, config, clock)
    await sched.startup()
    assert int(chain.snapshots[-1][0]["signed_at"]) > first
    assert chain.published == ["deepseek-ai/deepseek-v4-pro:fp8"]


async def test_confidential_startup_reads_the_record_before_sleeping(monkeypatch):
    """A coordinator that applies the update synchronously (the emulator does) is
    already correct when the PUT returns, so the boot must not sleep a poll interval
    to find that out: read once more straight after the write, then loop only if the
    record is still stale."""
    import asyncio

    clock = Clock()
    chain = FakeNode([], clock)          # delay_update_rounds = 0: the PUT lands at once
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def counting_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)
    await _confidential_scheduler(chain, clock, _boot_identity()).startup()

    assert chain.provider_polls == 2   # one before the PUT, one straight after it
    assert sleeps == []                # ...and not a single wait-loop sleep


async def test_confidential_startup_still_applies_the_allowed_models_gate():
    # The gate reads the FINAL record — the one fetched after the wait, not the
    # stale pre-update copy.
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_on_record(None, allow_all_models=False))
    ident = _boot_identity()
    config = make_config(
        [ModelConfig(model="org/e2ee-model:fp8", slas={"1h": SlaRate(rate_in=None, rate_out="0.6")},
                     backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
                     confidential=True)],
        box_key=None, poll_interval_s=0.01,
    )

    async def gated(provider_id):
        rec = await FakeNode.get_provider(chain, provider_id)
        # The network permits the model only once the record carries the boot key.
        if rec["box_key"] == "0x" + ident.cipher.public_key:
            chain.provider_rec["allowed_models"] = [MODEL_IDS["org/e2ee-model:fp8"]]
            rec = dict(chain.provider_rec)
        return rec

    chain.get_provider = gated
    sched = _confidential_scheduler(chain, clock, ident, config)
    await sched.startup()
    assert sched._allowed == {"org/e2ee-model:fp8"}
    assert chain.published == ["org/e2ee-model:fp8"]


async def test_mismatch_stays_fatal_only_for_an_operator_keyed_daemon():
    """Same mismatched record, two daemons: the operator-keyed one dies (its sealed
    payloads would be undecryptable), the confidential one publishes and waits."""
    clock = Clock()
    rec = _on_record("00" * 32)

    operator_chain = FakeNode([], clock, provider_rec=dict(rec))
    with pytest.raises(ConfigError, match="does not match"):
        await _startup_scheduler(operator_chain, make_config([text_model()]), clock).startup()
    # Never publishes over the operator's key: an operator-keyed daemon holds a
    # stable identity and a mismatch means somebody else's record, not a stale one.
    assert operator_chain.pushed("set_identity") == []

    ident = _boot_identity()
    conf_chain = FakeNode([], clock, provider_rec=dict(rec))
    await _confidential_scheduler(conf_chain, clock, ident).startup()
    assert conf_chain.provider_rec["box_key"] == "0x" + ident.cipher.public_key


async def test_no_box_key_and_no_boot_identity_is_a_config_error():
    # An operator-keyed daemon with no box key has no way to open a sealed payload;
    # fail at construction, naming both remedies, rather than crashing on a job.
    clock = Clock()
    with pytest.raises(ConfigError, match="cipher"):
        Scheduler(_confidential_config(), FakeNode([], clock), FakeCoord(), {}, FakeMetrics(), clock=clock)


# --- modality is established, never assumed ----------------------------------
#
# Modality decides how a job is metered. Guessing "text" for a media model would
# skip plan_media_units entirely: no clamp of num_images/duration_secs down to the
# units the client paid for, no guard against a client under-declaring to underpay,
# a media driver returning completion_tokens=None, and therefore a settle at the
# full cap. So an unestablished modality is refused, not defaulted.

MEDIA_MODEL = "black-forest-labs/flux-2-dev:fp8"
ONE_MEGAPIXEL = 1024 * 1024


def media_model(*, modality=None):
    return ModelConfig(
        model=MEDIA_MODEL,
        slas={"24h": SlaRate(rate_in=None, rate_out="0.02")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "flux"}),
        modality=modality,
    )


def uncapped_media_job():
    """A media job whose cap is out of the way, for tests about what is *counted*."""
    job = media_job()
    job.units_out = 2**31
    return job


def media_job(tag="media"):
    # Asks for two 1-megapixel images but paid for one: a correctly-metered run
    # clamps to one and settles 1_048_576 pixels.
    job_id, task_cid = pin_task({"prompt": f"a cat {tag}", "width": 1024, "height": 1024,
                                 "num_images": 2})
    return EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                  owner=OWNER, rate_in=None, rate_out=20_000, units_in=None,
                  units_out=ONE_MEGAPIXEL, task_cid=task_cid)


class MediaDriver:
    """A media backend: it reports no token count, so a job mis-metered as text
    settles with completion_tokens=None — i.e. at the full cap."""

    def __init__(self):
        self.ran_with = None

    async def run(self, job, input, *, timeout_s=None):
        self.ran_with = input
        return Normalized(kind="media", media_blobs=[b"\x89PNG"], completion_tokens=None)

    async def healthy(self):
        return True


class BlindChain(FakeNode):
    """A node whose catalog names ids but **no modality** — which is what a
    coordinator node actually serves: the on-chain model record is an id, a name
    and an enabled flag, and nothing in it says how a job is metered."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("catalog", [CatalogModel(model_id=m.model_id, name=m.name,
                                                   enabled=m.enabled) for m in CATALOG])
        super().__init__(*args, **kwargs)


class UnreachableCatalog(FakeNode):
    """A node whose catalog does not answer at all."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog_calls = 0

    async def get_models(self):
        self.catalog_calls += 1
        raise httpx.ConnectError("catalog unreachable")


async def test_media_job_is_not_claimed_when_nothing_can_name_its_modality():
    clock = Clock()
    job = media_job()
    chain = BlindChain([job], clock)
    driver = MediaDriver()
    sched = make_scheduler(make_config([media_model()]), chain, driver, clock)
    await sched.run_once()
    await sched.join()

    assert job.state == "Open"        # never claimed — still available to a provider that can price it
    assert chain.settled == []        # and so never settled at the full cap
    assert driver.ran_with is None    # the backend never ran unclamped


async def test_an_unreadable_catalog_stops_the_sweep_rather_than_guessing_ids():
    """No catalog, no poll: the book is addressed by model id and the ask book
    quotes ids, and neither can be spelled from the config alone. Costs a sweep,
    never correctness — the next one retries."""
    clock = Clock()
    job = media_job()
    chain = UnreachableCatalog([job], clock)
    sched = make_scheduler(make_config([media_model(modality="image")]), chain, MediaDriver(), clock)
    await sched.run_once()
    await sched.join()

    assert job.state == "Open"
    assert chain.ops == [] and chain.snapshots == []
    assert chain.catalog_calls == 1   # tried once, and the sweep did not raise


async def test_a_config_declared_modality_keeps_serving_a_catalog_that_names_none():
    clock = Clock()
    job = media_job()
    chain = BlindChain([job], clock)
    driver = MediaDriver()
    sched = make_scheduler(make_config([media_model(modality="image")]), chain, driver, clock)
    await sched.run_once()
    await sched.join()

    # Metered as media: the two-image ask is clamped to the one image paid for,
    # and the settle carries the delivered pixels rather than None.
    assert driver.ran_with["num_images"] == 1
    assert [(s[0], s[2]) for s in chain.settled] == [(job.job_id, ONE_MEGAPIXEL)]


async def test_a_sweep_re_reads_a_catalog_that_named_no_modality_at_startup():
    clock = Clock()
    job = media_job()
    chain = BlindChain([job], clock)
    driver = MediaDriver()
    sched = make_scheduler(make_config([media_model()]), chain, driver, clock)

    await sched.run_once()          # the catalog names no modality: nothing claimed
    await sched.join()
    assert chain.settled == []

    # ...and then it does. The ids do not move — they are chain facts — so only
    # the modality is picked up.
    chain.catalog = [CatalogModel(model_id=m.model_id, name=m.name, enabled=m.enabled,
                                  modality="image" if m.name == MEDIA_MODEL else m.modality)
                     for m in CATALOG]
    await sched.run_once()
    await sched.join()
    assert driver.ran_with["num_images"] == 1
    assert [(s[0], s[2]) for s in chain.settled] == [(job.job_id, ONE_MEGAPIXEL)]


def four_image_job(tag="short_render"):
    # Asks for four 1-megapixel images and paid for all four, so nothing is clamped:
    # the request count and the delivered count are free to differ, which is the only
    # way a settled charge can drift away from the frames the client receives.
    job_id, task_cid = pin_task({"prompt": f"a cat {tag}", "width": 1024, "height": 1024,
                                 "num_images": 4})
    return EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                  owner=OWNER, rate_in=None, rate_out=20_000, units_in=None,
                  units_out=4 * ONE_MEGAPIXEL, task_cid=task_cid)


class ShortRenderDriver:
    """A media backend that comes back with fewer frames than were asked for."""

    def __init__(self, delivered=1):
        self._delivered = delivered
        self.ran_with = None

    async def run(self, job, input, *, timeout_s=None):
        self.ran_with = input
        return Normalized(kind="media",
                          media_blobs=[b"\x89PNG frame %d" % i for i in range(self._delivered)],
                          width=1024, height=1024, content_type="image/png")

    async def healthy(self):
        return True


async def test_a_short_render_settles_the_frames_it_delivered():
    """One image back out of four asked for settles one image, not four.

    The count that settles is read off the frames inside the sealed result, so it
    is the same number the client SDK derives from those same frames when it
    displays what the job cost. Billing the request instead would charge four
    megapixels for one delivered megapixel, and the client would never see it.
    """
    clock = Clock()
    job = four_image_job()
    chain = BlindChain([job], clock)          # catalog down; modality comes from config
    driver = ShortRenderDriver(delivered=1)
    sched = make_scheduler(make_config([media_model(modality="image")]), chain, driver, clock)
    await sched.run_once()
    await sched.join()

    assert driver.ran_with["num_images"] == 4          # the cap covers four: nothing clamped
    (settled_job_id, result, settled_units), = chain.settled
    assert settled_job_id == job.job_id
    assert settled_units == ONE_MEGAPIXEL              # one frame delivered, one frame charged

    # The client's arithmetic over the frames it opens (MediaResult.cost sums
    # width × height across the delivered images) lands on the same number.
    payload = json.loads(_open_sealed(base64.b64decode(result), CLIENT))
    assert len(payload["images"]) == 1
    assert sum(f["width"] * f["height"] for f in payload["images"]) == settled_units


async def test_a_full_render_still_settles_every_frame_it_delivered():
    # The counterpart: nothing short about this one, so all four megapixels settle.
    clock = Clock()
    job = four_image_job("full_render")
    chain = BlindChain([job], clock)
    driver = ShortRenderDriver(delivered=4)
    sched = make_scheduler(make_config([media_model(modality="image")]), chain, driver, clock)
    await sched.run_once()
    await sched.join()

    assert [(s[0], s[2]) for s in chain.settled] == [(job.job_id, 4 * ONE_MEGAPIXEL)]


async def test_a_known_catalog_is_not_re_read_every_sweep():
    clock = Clock()
    chain = FakeNode([open_text_job("catalog_cached")], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    calls = {"n": 0}
    inner = chain.get_models

    async def counting():
        calls["n"] += 1
        return await inner()

    chain.get_models = counting
    await sched.run_once()
    await sched.join()
    await sched.run_once()
    await sched.join()
    assert calls["n"] == 1   # read once; the second sweep had nothing left to name


async def test_envelope_without_a_result_key_is_refused():
    # There is no in-the-clear result path: settling unsealed bytes would hand the
    # answer to the coordinator and to anyone reading the blob surface.
    clock = Clock()
    job = open_text_job("no_result_key", result_key=None)

    chain = FakeNode([job], clock)
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert driver.last_timeout_s is None
    assert chain.failed == [job.job_id]


@pytest.mark.parametrize("result_key", [
    "not-hex-at-all-not-hex-at-all-not-hex-at-all-not-hex-at-all-nope",   # 64 chars, not hex
    "aa" * 16,                                                            # right alphabet, 16 bytes
    "aa" * 64,                                                            # right alphabet, 64 bytes
    "0x" + "aa" * 32,                                                     # 0x-prefixed: nacl's hex decoder rejects it
])
async def test_a_malformed_result_key_fails_the_job_before_the_backend_runs(result_key):
    # The key is checked for shape when the envelope is opened, not when the
    # answer is sealed: a job whose result could never be sealed must not consume
    # backend capacity first and then die with the work already done.
    clock = Clock()
    job = open_text_job("bad_key", result_key=result_key)
    chain = FakeNode([job], clock)
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)
    await sched.run_once()
    await sched.join()

    assert driver.last_timeout_s is None      # the backend was never invoked
    assert chain.settled == []
    assert chain.failed == [job.job_id]


# --- a media result is a result ----------------------------------------------
#
# Frames ride inside the sealed result, exactly as text does: one sealed object
# goes out with the settle call and the coordinator pins it. Nothing is uploaded,
# and no frame carries a name of its own.

RESULT_SECRET = BoxCipher.generate()
RESULT_KEY = RESULT_SECRET.public_key


def _normalized_media(*, media_blobs=None, media_urls=None, width=1024, height=768,
                      content_type="image/png", seed=42, duration_secs=None) -> Normalized:
    return Normalized(kind="media", media_blobs=media_blobs, media_urls=media_urls,
                      content_type=content_type, width=width, height=height,
                      duration_secs=duration_secs, seed=seed)


def _scheduler_with_backend_media(normalized: Normalized, *, modality="image"):
    """A scheduler whose backend renders exactly ``normalized``."""

    class StaticMediaDriver:
        async def run(self, job, input, *, timeout_s=None):
            return normalized

        async def healthy(self):
            return True

    clock = Clock()
    return make_scheduler(make_config([media_model(modality=modality)]),
                          FakeNode([], clock), StaticMediaDriver(), clock)


def _open_sealed(sealed: bytes, secret: BoxCipher) -> bytes:
    env = json.loads(sealed)
    assert env["enc"] == "vorq-sealed-v1"
    return secret.decrypt(base64.b64decode(env["ciphertext"]))


async def test_media_result_is_sealed_with_frames_inline():
    """Media takes the result path: one sealed object, frames base64 inside it.

    No upload, no per-frame CID — from _seal_result onward media and text are
    the same code path, and the coordinator pins the one object it is handed.
    """
    frame = b"\x89PNG\r\n\x1a\n fake pixels"
    normalized = _normalized_media(media_blobs=[frame], width=1024, height=768,
                                   content_type="image/png", seed=42)
    sched = _scheduler_with_backend_media(normalized)

    sealed, completion_tokens = await sched._build_result(media_job(), normalized, RESULT_KEY)

    # Media bills on pixels, not tokens: the count is the delivered frame's own
    # dimensions, the same product the client computes from the frame it opens.
    assert completion_tokens == 1024 * 768
    payload = json.loads(_open_sealed(sealed, RESULT_SECRET))
    assert payload["seed"] == 42
    assert len(payload["images"]) == 1
    image = payload["images"][0]
    assert base64.b64decode(image["b64"]) == frame
    assert image == {"b64": image["b64"], "content_type": "image/png", "width": 1024, "height": 768}


async def test_video_result_carries_one_frame_and_its_duration():
    frame = b"\x00\x00\x00 ftypisom fake video"
    normalized = _normalized_media(media_blobs=[frame], width=1280, height=720,
                                   content_type="video/mp4", seed=None, duration_secs=5)
    sched = _scheduler_with_backend_media(normalized, modality="video")

    sealed, completion_tokens = await sched._build_result(uncapped_media_job(), normalized, RESULT_KEY)

    assert completion_tokens == 1280 * 720 * 5            # pixel-seconds, from the sealed frame
    payload = json.loads(_open_sealed(sealed, RESULT_SECRET))
    assert "images" not in payload
    assert "seed" not in payload                          # a backend that reports none claims none
    assert base64.b64decode(payload["video"].pop("b64")) == frame
    assert payload["video"] == {"content_type": "video/mp4", "width": 1280, "height": 720,
                                "duration_secs": 5}
    assert payload["units"] == 1280 * 720 * 5


async def test_a_sealed_result_stamps_the_line_it_answers():
    """The correlation stamp: which job, and which of the caller's own labels.

    A claimant running thousands of lines of one batch concurrently can cross a settle with
    another line's answer. The client cannot see that from the outside — its result cipher is
    derived from its wallet, not per job, so another line's result decrypts perfectly. The
    stamp is what makes the crossing visible.

    It is a correctness check, not a security boundary: the provider authors these bytes, so a
    dishonest one stamps whatever it likes. What it catches is the provider's own bug.
    """
    normalized = _normalized_media(media_blobs=[b"\x89PNG\r\n\x1a\n x"], width=8, height=8)
    sched = _scheduler_with_backend_media(normalized)
    job = media_job()

    sealed, _ = await sched._build_result(job, normalized, RESULT_KEY, custom_id="req-42")

    payload = json.loads(_open_sealed(sealed, RESULT_SECRET))
    assert payload["vorq"] == {"job_id": job.job_id, "custom_id": "req-42"}


async def test_an_unlabelled_job_is_stamped_with_its_job_id_alone():
    """`custom_id` is absent, not null, when the caller named none — the client reads this
    back onto a result field and a null would be indistinguishable from a label of 'None'."""
    normalized = _normalized_media(media_blobs=[b"\x89PNG\r\n\x1a\n x"], width=8, height=8)
    sched = _scheduler_with_backend_media(normalized)
    job = media_job()

    sealed, _ = await sched._build_result(job, normalized, RESULT_KEY)

    payload = json.loads(_open_sealed(sealed, RESULT_SECRET))
    assert payload["vorq"] == {"job_id": job.job_id}


async def test_an_embedding_result_seals_the_response_and_bills_no_output_tokens():
    """The billing asymmetry reaching settle.

    A text result with no token count is a backend failure — the daemon abandons rather than
    let the escrow's full-cap fallback fire. An embedding result with no token count is
    *correct*: the backend reports `usage.prompt_tokens` and there is no completion count to
    report. So this path must settle at 0 rather than raise, and the charge collapses to
    `rate_in * units_in`.
    """
    response = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": "dmVjdG9yLWJ5dGVz"}],
        "model": "emb",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
    }
    normalized = Normalized(kind="embedding", completion_tokens=None, raw=response)
    sched = _scheduler_with_backend_media(normalized)
    job = media_job()

    sealed, completion_tokens = await sched._build_result(job, normalized, RESULT_KEY)

    assert completion_tokens == 0, "no output side to bill"
    # The client is handed the OpenAI EmbeddingResponse verbatim — same shape it would have
    # got from the backend directly, sealed — beside the stamp naming the line.
    opened = json.loads(_open_sealed(sealed, RESULT_SECRET))
    assert opened["vorq"] == {"job_id": job.job_id}
    assert {k: v for k, v in opened.items() if k != "vorq"} == response


async def test_a_media_result_with_no_frames_is_a_backend_failure():
    # The client paid for pixels: an empty render is a backend failure the daemon
    # reports for an immediate refund, never a settle with nothing sealed inside.
    normalized = _normalized_media(media_blobs=[])
    sched = _scheduler_with_backend_media(normalized)
    with pytest.raises(BackendError):
        await sched._build_result(media_job(), normalized, RESULT_KEY)


async def test_a_frame_the_backend_cannot_serve_is_reported_not_left_to_expire():
    # The render is named but undeliverable: the frame URL answers 502. That is a
    # backend failure, not an internal one — it is reported so the client's escrow
    # refunds now, instead of the job sitting claimed until the SLA expires.
    clock = Clock()
    job = media_job("unfetchable")
    chain = BlindChain([job], clock)      # catalog down; the config declares the modality
    metrics = FakeMetrics()

    class UrlMediaDriver:
        async def run(self, job, input, *, timeout_s=None):
            return Normalized(kind="media", media_urls=["https://cdn.test/a.png"],
                              width=1024, height=1024)

        async def healthy(self):
            return True

    config = make_config([media_model(modality="image")])
    cdn = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(502)))
    sched = Scheduler(config, chain, FakeCoord(), {m.model: UrlMediaDriver() for m in config.models},
                      metrics, clock=clock, blobs=BlobResolver([SOURCE]), http=cdn,
                      escrow=FakeEscrow())
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert metrics.fails == ["backend_error"]          # not internal_error
    assert chain.failed == [job.job_id]
    # The URL cannot ride the network, because nothing but the job id and a
    # timestamp does: a Fail op has no field an operator's runtime could leak
    # through, and the detail stays in this process's own log.
    (payload,) = chain.pushed("fail")
    assert set(payload) == {"job_id", "issued_at"}
    assert "cdn.test" not in json.dumps(payload)


# --- boot recovery of jobs claimed in a previous life -------------------------


def claimed_text_job(tag="orphan", *, model=MODEL, claimed_at=1000):
    """A job this provider claimed before the daemon died: still Claimed, still
    ours, its task bytes still pinned. ``claimed_at`` is always stamped — the
    coordinator sets it at the claim, so a Claimed row without one is not a row
    any daemon can ever be handed."""
    job = open_text_job(tag, model=model)
    job.state = "Claimed"
    job.provider = PROVIDER_ID
    job.claimed_at = claimed_at
    return job


async def test_a_recovered_claim_whose_input_outruns_its_units_is_failed_back():
    """The one path with no pre-claim gate in front of it.

    Boot recovery picks up a job a previous life claimed — a life that may have
    run a build without this check at all. The claim is already spent, so the
    remedy here is the other one: hand it back. Inside the grace window that
    refunds the client in full and costs the provider nothing.
    """
    clock = Clock()
    job = claimed_text_job()
    job.units_in, job.rate_in = 1, 200_000
    # Re-pin the same job id over a payload far larger than one unit buys.
    job.job_id, job.task_cid = pin_task({"input": "x" * 200_000})
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert metrics.fails == ["units_in_short"]
    assert [payload["job_id"] for payload in chain.pushed("fail")] == [job.job_id]
    assert chain.settled == []
    assert sched._inflight == 0          # refused before a capacity slot was taken


async def test_a_job_the_pre_claim_gate_admitted_is_never_failed_after_decryption():
    """Both checks measure the same integer — `len(container) - 121` is exactly
    `len(plaintext)` — so a bid the gate let through cannot trip the backstop.

    That identity is what closes the griefing vector: a fail refunds the client
    `cap + feeCap + gasFeeSnap` in full, leaving the relayer's claim gas unrecovered, so an
    under-declared bid must never be able to reach a claim in the first place.
    Sized one byte inside the floor, which is where the two would disagree if they
    ever could.
    """
    clock = Clock()
    payload_size = 200_000
    _, probe_cid = pin_task({"input": "x" * payload_size})
    carried = sealed_plaintext_bytes(len(await SOURCE.get(probe_cid)))
    # The smallest declaration the gate accepts for this payload, exactly.
    tightest = -(-(carried - ENVELOPE_SLACK_BYTES) // MAX_INPUT_BYTES_PER_UNIT)

    job = underdeclared_text_job(units_in=tightest, size=payload_size)
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]
    assert metrics.fails == []


async def test_boot_recovery_settles_a_claimed_job_from_a_previous_life():
    clock = Clock()
    job = claimed_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]
    assert chain.failed == []
    assert metrics.claims == 0   # never re-claimed: the previous life already paid for it


async def test_boot_recovery_fails_a_claimed_job_for_an_unserved_model():
    # The operator dropped the model between boots: the job cannot run here, so it
    # is failed back now for an immediate refund rather than idling to SLA expiry.
    clock = Clock()
    job = claimed_text_job("dropped", model="model-gone:fp8")
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert driver.last_timeout_s is None          # the backend never ran
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["unservable"]


async def test_boot_recovery_fails_a_claim_whose_window_closed_while_the_daemon_was_down():
    # The dominant crash case: the daemon was down longer than the SLA window.
    # Running the job would burn a capacity slot on a result the SLA guard drops
    # unreported, leaving the client's escrow locked until someone reclaims it.
    clock = Clock()
    job = claimed_text_job("expired")          # claimed at t=1000, 1h window
    clock.advance(2 * 3600)                    # ...and the daemon came back two hours later
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()

    assert driver.last_timeout_s is None       # the backend never ran
    assert chain.settled == []
    assert chain.failed == [job.job_id]      # refunded now, not at reclaim
    assert metrics.fails == ["claim_expired"]


async def test_boot_recovery_retries_after_an_unreadable_listing():
    # A listing body this build cannot read is not a fact about the jobs, and it
    # must not escape: an exception here would take the whole sweep's claim path
    # down with it.
    clock = Clock()
    job = claimed_text_job("garbled")

    class GarbledChain(FakeNode):
        fault = True

        async def list_claimed_jobs(self, provider):
            jobs = await super().list_claimed_jobs(provider)
            if self.fault:
                self.fault = False
                raise TypeError("unexpected field in the listing row")
            return jobs

    chain = GarbledChain([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()                     # does not raise
    await sched.join()
    assert chain.settled == []

    await sched.run_once()
    await sched.join()
    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_boot_recovery_runs_once():
    clock = Clock()
    chain = FakeNode([claimed_text_job("once")], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    await sched.run_once()
    await sched.join()

    assert chain.claimed_lists == 1   # boot recovery, not a per-sweep read


async def test_boot_recovery_retries_after_a_transport_fault():
    clock = Clock()
    job = claimed_text_job("flaky")

    class FlakyChain(FakeNode):
        """The recovery read fails once — the coordinator was not answering yet."""
        fault = True

        async def list_claimed_jobs(self, provider):
            jobs = await super().list_claimed_jobs(provider)
            if self.fault:
                self.fault = False
                raise httpx.ConnectError("coordinator unreachable")
            return jobs

    chain = FlakyChain([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert chain.settled == []        # nothing recovered yet

    await sched.run_once()
    await sched.join()
    assert chain.claimed_lists == 2   # the fault did not consume the one attempt
    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_recovery_leaves_alone_a_job_this_daemon_is_already_running():
    # The first sweep's recovery read failed, so the sweep after it lists the
    # Claimed rows again — and by then it is looking at the job its own claim
    # path started in between. Re-ingesting it would run the same claim twice.
    clock = Clock()
    job = open_text_job("already_running")

    class FlakyChain(FakeNode):
        fault = True

        async def list_claimed_jobs(self, provider):
            jobs = await super().list_claimed_jobs(provider)
            if self.fault:
                self.fault = False
                raise httpx.ConnectError("coordinator unreachable")
            return jobs

    class HangingDriver(FakeDriver):
        """Holds the backend call open until the test lets it answer."""

        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()
            self.calls = 0

        async def run(self, job, input, *, timeout_s=None):
            self.calls += 1
            await self.release.wait()
            return await super().run(job, input, timeout_s=timeout_s)

    chain = FlakyChain([job], clock)
    driver = HangingDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock)

    await sched.run_once()               # recovery fails; the claim path claims and spawns
    for _ in range(5):                   # let the task reach the backend call
        await asyncio.sleep(0)
    assert job.state == "Claimed" and driver.calls == 1

    await sched.run_once()               # recovery reads it back as Claimed by us
    for _ in range(5):
        await asyncio.sleep(0)
    assert driver.calls == 1             # not started a second time

    driver.release.set()
    await sched.join()
    assert [s[0] for s in chain.settled] == [job.job_id]


# --- the commitment is the security of the whole fetch path ------------------
#
# The gateway is untrusted by construction. What makes reading from it safe is
# that the bytes have to re-derive the job's own id: substituted ciphertext
# cannot, and a seed_wrap lifted from another order cannot either. Both must be
# refused BEFORE anything is decrypted — a cipher that runs on attacker-chosen
# bytes has already lost, whatever it answers.


class _PassThroughSource:
    """A source that serves whatever it is handed — the commitment does the work."""

    def __init__(self, blob: bytes):
        self._blob = blob

    async def get(self, cid):
        return self._blob


class _RecordingCipher:
    """Wraps the daemon's cipher and records every unseal attempt."""

    def __init__(self, inner):
        self._inner = inner
        self.attempts = 0

    @property
    def public_key(self):
        return self._inner.public_key

    def decrypt(self, data):
        self.attempts += 1
        return self._inner.decrypt(data)


def _tampered(container: bytes) -> bytes:
    """One flipped bit in the bulk — the alteration only the digest inside the
    commitment catches."""
    return container[:-1] + bytes([container[-1] ^ 0x01])


def _wrap_swapped(container: bytes, donor: bytes) -> bytes:
    """This order's ciphertext under another order's wrap: well formed, and it
    reproduces somebody else's commitment.

    The offsets come from the module under test, never from literals here: a
    hardcoded 5/85 survives the version-byte change by silently slicing four bytes
    off the wrap instead, which still misses the commitment and so still passes —
    a green test that has stopped testing what it names.
    """
    return container[:1] + donor[1:MIN_CONTAINER_BYTES] + container[MIN_CONTAINER_BYTES:]


@pytest.mark.parametrize("corrupt", ["tampered", "wrap_swapped"])
async def test_a_container_that_misses_the_commitment_is_never_claimed(corrupt):
    """Refused before the claim, so nothing is spent and nothing is opened.

    Two shapes, and the second is the one the wrap is hashed into ``c`` for: a
    ``seed_wrap`` lifted from another order over this order's own ciphertext —
    same length, well formed, and it commits to somebody else's job id.
    """
    clock = Clock()
    job = designated_text_job(f"corrupt_{corrupt}")
    good = SOURCE._blobs[job.task_cid]
    donor = SOURCE._blobs[designated_text_job("donor").task_cid]
    bad = _tampered(good) if corrupt == "tampered" else _wrap_swapped(good, donor)
    assert bad != good and len(bad) == len(good)

    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FakeDriver()
    cipher = _RecordingCipher(DAEMON_BOX)
    escrow = FakeEscrow()
    config = make_config([text_model()])
    sched = Scheduler(config, chain, FakeCoord(), {m.model: driver for m in config.models},
                      metrics, clock=clock, cipher=cipher,
                      blobs=BlobResolver([_PassThroughSource(bad)], attempts=1),
                      escrow=escrow)
    await sched.run_once()
    await sched.join()

    assert chain.ops == []             # not one signed op: no claim, so no fail either
    assert chain.settled == []
    assert chain.failed == []
    assert job.state == "Open"         # never taken off the book
    assert cipher.attempts == 0        # nothing was unsealed...
    assert escrow.calls == []          # ...and no key was even asked for
    assert driver.last_timeout_s is None
    assert metrics.claims == 0


# --- and refusing it must not become the daemon's whole sweep ----------------
#
# The book has no memory: an Open job it cannot serve bytes for comes back every
# five seconds, and the fetch that refuses it is inline, ahead of every other bid
# in the page. Refusing correctly but re-refusing forever is how one posted order
# becomes a provider that mostly sleeps.


class _CountingSource:
    """Serves a fixed body — or raises — and counts how often it was asked."""

    def __init__(self, blob: bytes | None):
        self._blob = blob
        self.calls = 0

    async def get(self, cid):
        self.calls += 1
        if self._blob is None:
            raise BlobError(f"no blob under {cid}")
        return self._blob


def _sched_over(job, source, clock, *, attempts=1):
    config = make_config([text_model()])
    return Scheduler(config, FakeNode([job], clock), FakeCoord(),
                     {m.model: FakeDriver() for m in config.models}, FakeMetrics(),
                     clock=clock, cipher=_RecordingCipher(DAEMON_BOX),
                     blobs=BlobResolver([source], attempts=attempts, backoff_s=0),
                     escrow=FakeEscrow())


async def test_a_job_whose_bytes_miss_the_commitment_is_fetched_once_not_every_sweep():
    # Permanent for this job id: the CID is content-addressed, so the bytes under
    # it are the bytes under it and no later sweep can be told anything new.
    clock = Clock()
    job = designated_text_job("stall_mismatch")
    source = _CountingSource(_tampered(SOURCE._blobs[job.task_cid]))
    sched = _sched_over(job, source, clock)

    for _ in range(5):
        await sched.run_once()
        await sched.join()

    assert source.calls == 1           # asked once across five sweeps
    assert sched._node.ops == []       # and still never claimed


async def test_a_bid_refused_for_its_input_units_is_not_re_fetched_every_sweep():
    """Permanent, for the same reason the commitment case is: the container is
    content-addressed and `units_in` is a *signed* order term, so neither half of
    the comparison can move. Without the shelf the daemon re-fetches a megabyte
    from the gateway every sweep until the order expires, and one greedy bid
    becomes the whole sweep's bandwidth.
    """
    clock = Clock()
    job = underdeclared_text_job("stall_units")
    source = _CountingSource(SOURCE._blobs[job.task_cid])
    sched = _sched_over(job, source, clock)

    for _ in range(5):
        await sched.run_once()
        await sched.join()

    assert source.calls == 1           # asked once across five sweeps
    assert sched._node.ops == []       # and still never claimed


async def test_an_unresolvable_name_backs_off_and_is_tried_again_later():
    # Not permanent: a name that resolves to nothing may simply be a pin still
    # propagating, so the wait is doubled rather than the job written off.
    from vorqd.scheduler import UNRESOLVED_RETRY_S

    clock = Clock()
    job = designated_text_job("stall_missing")
    source = _CountingSource(None)
    sched = _sched_over(job, source, clock)

    await sched.run_once()
    await sched.join()
    assert source.calls == 1

    await sched.run_once()             # same second: still shelved
    await sched.join()
    assert source.calls == 1

    clock.advance(UNRESOLVED_RETRY_S + 1)
    await sched.run_once()             # the wait is over, so it is tried again
    await sched.join()
    assert source.calls == 2

    clock.advance(UNRESOLVED_RETRY_S + 1)
    await sched.run_once()             # ...but the second miss doubled the wait
    await sched.join()
    assert source.calls == 2


async def test_a_job_that_resolves_is_never_left_shelved():
    # A job deferred while its pin propagated must not carry the entry once the
    # bytes arrive — the book is for jobs this daemon could not resolve.
    from vorqd.scheduler import UNRESOLVED_RETRY_S

    clock = Clock()
    job = designated_text_job("stall_recovers")
    good = SOURCE._blobs[job.task_cid]
    source = _CountingSource(None)
    sched = _sched_over(job, source, clock)

    await sched.run_once()
    await sched.join()
    assert str(job.job_id) in sched._deferred

    source._blob = good
    clock.advance(UNRESOLVED_RETRY_S + 1)
    await sched.run_once()
    await sched.join()
    assert str(job.job_id) not in sched._deferred


@pytest.mark.parametrize("corrupt", ["tampered", "wrap_swapped"])
async def test_the_open_path_re_checks_the_commitment_itself(corrupt):
    # Belt and braces, and deliberately so: the check at the point of USE does not
    # inherit the fetch's word for it. A resolver that skipped the check — an
    # in-process source, a future cache — must not be able to feed a cipher.
    clock = Clock()
    job = designated_text_job(f"direct_{corrupt}")
    good = SOURCE._blobs[job.task_cid]
    donor = SOURCE._blobs[designated_text_job("direct_donor").task_cid]
    bad = _tampered(good) if corrupt == "tampered" else _wrap_swapped(good, donor)

    cipher = _RecordingCipher(DAEMON_BOX)
    escrow = FakeEscrow()
    config = make_config([text_model()])
    sched = Scheduler(config, FakeNode([job], clock), FakeCoord(),
                      {m.model: FakeDriver() for m in config.models}, FakeMetrics(),
                      clock=clock, cipher=cipher, blobs=BlobResolver([SOURCE]), escrow=escrow)

    from vorqd.scheduler import PayloadError

    with pytest.raises(PayloadError) as exc:
        await sched._open_task(job, bad)
    assert exc.value.reason == "commitment_mismatch"
    assert cipher.attempts == 0 and escrow.calls == []
    # ...and the untouched container still opens, so the fixture proves something.
    await sched._open_task(job, good)
    assert cipher.attempts == 1


async def test_a_result_well_under_the_bound_settles_inline_unbounded_otherwise():
    # The daemon puts no bound of its own below INLINE_MAX_BYTES: a result well
    # past a megabyte still rides inline as base64, and the node's blob ceiling
    # — a 413, reported as `settle_result_too_large` above — is the only limit
    # there is.
    clock = Clock()
    job = open_text_job("big")
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock, metrics)
    big = b"x" * (2 * 1024 * 1024)

    async def huge(job, normalized, result_key, custom_id=None):
        return big, 1

    sched._build_result = huge
    await sched.run_once()
    await sched.join()

    (payload,) = chain.pushed("settle")
    assert base64.b64decode(payload["result"]) == big
    assert chain.uploaded == []
    assert chain.failed == [] and metrics.fails == []


# --- the private acceptance floor --------------------------------------------
#
# The published book is NOT part of this. Every test here asserts on what the
# daemon claims, and `test_the_published_book_never_moves_with_load` is the one
# that pins the other half of the bargain.


def _bid(rate_in=200_000, rate_out=600_000, tag="bid"):
    return open_text_job(tag, rate_in=rate_in, rate_out=rate_out)


async def test_the_published_book_never_moves_with_load():
    """The ask book advertises the list price and nothing else. A floor the
    network can see is a floor bids converge onto, so the discount stays private
    — and this feature must therefore cost no transactions at all."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20,
                                                          bid_tolerance_pct=10)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 0.0}),
    )
    await sched.sync_asks()
    assert _quotes(chain) == [
        {"model_id": 7, "sla": 3600, "rate_in": "0.2", "rate_out": "0.6"}
    ]

    sched._load.loads[MODEL] = 1.0
    await sched.sync_asks()
    assert len(chain.snapshots) == 1   # nothing to republish: the book never moved


async def test_a_bid_at_the_published_ask_is_always_claimed():
    """The floor only ever sits below the advertised price, so a bid that
    matches the book can never be refused by this daemon's own floor."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 1.0}),
    )
    assert sched.profitable(_bid(), sched._config.models[0], decimals=6) is True


async def test_an_idle_backend_claims_a_bid_the_whole_discount_under_the_book():
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 0.0}),
    )
    model = sched._config.models[0]
    assert sched.profitable(_bid(160_000, 480_000), model, decimals=6) is True
    assert sched.profitable(_bid(160_000, 479_999), model, decimals=6) is False


async def test_an_occupancy_load_source_reads_the_entrys_own_fullness():
    """A hosted backend exports no metrics; `load: {source: occupancy}` makes
    the entry's held-over-holdable its load, so an idle entry earns the whole
    allowance and a full one none."""
    clock = Clock()
    chain = FakeNode([], clock)
    metrics = FakeMetrics()
    model = dataclasses.replace(text_model(concurrency=1, rate_limit={"1h": 2}),
                                load=LoadProbeConfig(source="occupancy"))
    sched = make_scheduler(
        make_config([model], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, metrics,
    )
    model = sched._config.models[0]
    assert sched._floor_pct(model) == 20                     # idle: the whole allowance
    assert sched.profitable(_bid(160_000, 480_000), model, decimals=6) is True
    throttle = sched._throttles[MODEL]
    throttle.hold()
    assert sched._floor_pct(model) == 10                     # half full
    assert metrics.loads[MODEL] == 0.5
    throttle.hold()
    assert sched._floor_pct(model) == 0                      # full: the configured rates
    assert sched.profitable(_bid(200_000, 599_999), model, decimals=6) is False


async def test_a_quiet_queue_earns_the_flat_tolerance_and_a_half_full_one_does_not():
    """The operator's threshold: under `low_load_pct` of the entry's own
    capacity in use, `bid_tolerance_pct` applies flat; at it, nothing."""
    clock = Clock()
    chain = FakeNode([], clock)
    model = dataclasses.replace(text_model(concurrency=2, rate_limit={"1h": 4}),
                                load=LoadProbeConfig(source="occupancy"))
    sched = make_scheduler(
        make_config([model], pricing=PricingConfig(bid_tolerance_pct=10, low_load_pct=50)),
        chain, FakeDriver(), clock,
    )
    model = sched._config.models[0]
    throttle = sched._throttles[MODEL]
    throttle.hold()
    assert sched._floor_pct(model) == 10                     # 1 of 4 held: quiet
    throttle.hold()
    assert sched._floor_pct(model) == 0                      # 2 of 4: no longer under 50%


async def test_a_bounded_entry_without_a_load_source_earns_no_discount():
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model(concurrency=2)], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock,
    )
    assert sched._floor_pct(sched._config.models[0]) == 0


async def test_the_discount_shrinks_as_the_backend_fills():
    """Half-loaded earns half the allowance: 10% of 600000 is 540000."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 0.5}),
    )
    model = sched._config.models[0]
    assert sched.profitable(_bid(180_000, 540_000), model, decimals=6) is True
    assert sched.profitable(_bid(180_000, 539_999), model, decimals=6) is False


async def test_a_busy_backend_holds_the_configured_floor():
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 1.0}),
    )
    model = sched._config.models[0]
    assert sched.profitable(_bid(200_000, 599_999), model, decimals=6) is False


async def test_an_unknown_load_holds_the_configured_floor():
    """A dark probe, a stale one, or none at all. Unknown is not idle."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=FakeLoad({}),
    )
    assert sched.profitable(_bid(200_000, 599_999), sched._config.models[0], decimals=6) is False


async def test_the_default_config_accepts_exactly_what_it_did_before():
    """The whole feature is off unless the operator turns it on."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock,
                           load=FakeLoad({MODEL: 0.0}))
    model = sched._config.models[0]
    assert sched.profitable(_bid(), model, decimals=6) is True
    assert sched.profitable(_bid(200_000, 599_999), model, decimals=6) is False


async def test_the_tolerance_applies_only_below_the_load_gate():
    clock = Clock()
    chain = FakeNode([], clock)
    config = make_config([text_model()],
                         pricing=PricingConfig(bid_tolerance_pct=10, low_load_pct=30))
    load = FakeLoad({MODEL: 0.2})
    sched = make_scheduler(config, chain, FakeDriver(), clock, load=load)
    model = config.models[0]
    assert sched.profitable(_bid(180_000, 540_000), model, decimals=6) is True
    assert sched.profitable(_bid(180_000, 539_999), model, decimals=6) is False

    load.loads[MODEL] = 0.9   # above the gate: the tolerance closes
    assert sched.profitable(_bid(180_000, 540_000), model, decimals=6) is False


async def test_the_load_discount_and_the_tolerance_stack():
    """20% earned at an idle backend plus a 10% low-load tolerance is 30% off:
    600000 -> 420000."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()],
                    pricing=PricingConfig(max_discount_pct=20, bid_tolerance_pct=10,
                                          low_load_pct=30)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 0.0}),
    )
    model = sched._config.models[0]
    assert sched.profitable(_bid(140_000, 420_000), model, decimals=6) is True
    assert sched.profitable(_bid(140_000, 419_999), model, decimals=6) is False


async def test_the_floor_never_reaches_zero():
    """However the knobs are stacked, a floor that accepted a bid paying nothing
    would not be a floor."""
    clock = Clock()
    chain = FakeNode([], clock)
    model = ModelConfig(
        model=MODEL,
        slas={"1h": SlaRate(rate_in=None, rate_out="0.000001")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
    )
    sched = make_scheduler(
        make_config([model], pricing=PricingConfig(max_discount_pct=90, bid_tolerance_pct=90,
                                                   low_load_pct=100)),
        chain, FakeDriver(), clock, load=FakeLoad({MODEL: 0.0}),
    )
    assert sched.profitable(_bid(rate_in=None, rate_out=1), sched._config.models[0], decimals=6) is True
    assert sched.profitable(_bid(rate_in=None, rate_out=0), sched._config.models[0], decimals=6) is False


async def test_the_sweep_reports_the_current_floor_discount():
    clock = Clock()
    chain = FakeNode([], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20,
                                                          bid_tolerance_pct=10,
                                                          low_load_pct=30)),
        chain, FakeDriver(), clock, metrics, load=FakeLoad({MODEL: 0.0}),
    )
    await sched.run_once()
    assert metrics.floor_discounts[MODEL] == 30   # 20 earned + 10 low-load tolerance


# --- pushing the floor down to the book --------------------------------------
#
# The floor decides which bids are worth reading, not just which are worth
# claiming. `GET /evm/jobs` bounds every listing and orders it oldest-first, so a
# daemon that reads the book unfiltered reads the oldest page of it and nothing
# else. These tests assert on the exact query the node received: a test that
# only checked the jobs that came back would pass with no filter sent at all.


def _two_window_model():
    """One model, two windows, two prices — and one filter for both."""
    return ModelConfig(
        model=MODEL,
        slas={"1h": SlaRate(rate_in="0.2", rate_out="0.6"),
              "24h": SlaRate(rate_in="0.1", rate_out="0.3")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
    )


def _output_only_model(slas=None):
    return ModelConfig(
        model=MODEL,
        slas=slas or {"1h": SlaRate(rate_in=None, rate_out="0.6")},
        backend=BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "rt"}),
    )


class DisobedientNode(FakeNode):
    """A node that answers the whole book whatever it was asked for.

    A stale floor, an old build, or a node that simply lies. The daemon must not
    care: the filters are an optimisation and `profitable` is the boundary.
    """

    async def list_open_jobs(self, model_id, *, free, **_filters):
        return await FakeNode.list_open_jobs(self, model_id, free=free)


def test_a_configured_bid_age_is_accepted_and_reported_as_ignored(caplog):
    """`bid_filter.min_age_s` predates the matcher: the coordinator now decides
    which bids this daemon sees, so the key is still accepted (an operator's
    YAML must not stop loading) and named once as inert."""
    clock = Clock()
    with caplog.at_level("WARNING"):
        make_scheduler(make_config([text_model()], bid_filter={"min_age_s": 3600}),
                       FakeNode([], clock), FakeDriver(), clock)
    assert any("min_age_s" in r.getMessage() and "ignored" in r.getMessage()
               for r in caplog.records)


async def test_the_daemon_keeps_no_clock_of_its_own_for_resting_bids():
    """Bid age is a fact the book already carries, not daemon state.

    The old first-seen clock reset on every restart, so a fresh process waited
    the whole period again while the chain's own answer had not moved.
    """
    clock = Clock()
    sched = make_scheduler(make_config([text_model()]), FakeNode([], clock), FakeDriver(), clock)
    assert not hasattr(sched, "_first_seen")
    assert not hasattr(sched, "_rested")


async def test_the_sweep_asks_only_for_bids_it_could_actually_take():
    clock = Clock()
    chain = FakeNode([], clock)
    load = FakeLoad({MODEL: 0.0})
    sched = make_scheduler(
        make_config([text_model()], capacity=4, pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=load,
    )

    await sched.run_once()
    # An idle backend: 20% off both configured rates, and four free slots.
    assert chain.job_queries == [
        {"model_id": MODEL_IDS[MODEL], "free": 4, "min_rate_out": 480_000, "min_rate_in": 160_000}
    ]

    load.loads[MODEL] = 1.0
    await sched.run_once()
    # A full backend earns nothing: the filter is the configured price exactly.
    assert chain.job_queries[-1]["min_rate_out"] == 600_000
    assert chain.job_queries[-1]["min_rate_in"] == 200_000


async def test_an_unknown_load_asks_at_the_configured_price():
    """The one safety property, on the query as well as on the claim: a dark
    probe is a busy backend, never an idle one."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(
        make_config([text_model()], pricing=PricingConfig(max_discount_pct=20)),
        chain, FakeDriver(), clock, load=FakeLoad({}),
    )
    await sched.run_once()
    assert chain.job_queries[-1]["min_rate_out"] == 600_000


async def test_the_sweep_asks_the_matcher_for_exactly_its_free_slots():
    """`free` is what the coordinator may lease to this daemon right now: the
    configured capacity less what is already in flight, never the capacity."""
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()], capacity=4), chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 4

    sched._inflight = 3
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 1


async def test_a_full_daemon_still_polls_and_reports_no_free_slot():
    """The poll is the heartbeat and `free` is what it reports.

    A full daemon that stopped polling would leave a stale, positive `free` on
    record for a whole liveness window, and the matcher would keep leasing it
    jobs it cannot start — each one parked for a lease window before it moves
    on. So it polls anyway, says 0, and the matcher leases it nothing.
    """
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([text_model()], capacity=2), chain, FakeDriver(), clock)
    await sched.run_once()          # boots the catalog, and polls once
    chain.job_queries.clear()

    sched._inflight = 2             # every slot taken
    await sched.run_once()
    assert [q["free"] for q in chain.job_queries] == [0]


class GreedyNode(FakeNode):
    """A node that leases every open row, whatever `free` asked for."""

    async def list_open_jobs(self, model_id, *, free, **filters):
        self.job_queries.append({"model_id": model_id, "free": free})
        name = self._name(model_id)
        return [j for j in self._jobs.values() if j.state == "Open" and j.model == name]


class BlinkingNode(FakeNode):
    """A node whose provider record is unreadable for one read."""

    fail_next = False

    async def get_provider(self, provider_id):
        if self.fail_next:
            self.fail_next = False
            raise httpx.ConnectError("down")
        return await super().get_provider(provider_id)


def _granted(n):
    # The record as the node serves it: every integer a JSON integer.
    return {"box_key": None, "allow_all_models": True, "allowed_models": [],
            "capacity": n, "active_jobs": 0}


async def test_free_is_capped_by_the_slots_the_network_grants():
    """The registry grants `min(requested, ceiling) × reputation / 1000` slots and
    refuses a claim past that. A daemon offering its configured capacity while
    the grant is smaller is leased rows it then skips at the simulate, and is
    named on client challenges it cannot serve — so the grant caps the poll."""
    clock = Clock()
    chain = FakeNode([], clock, provider_rec=_granted(1))
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()], capacity=4), chain, FakeDriver(), clock,
                           metrics)
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 1
    assert metrics.granted == 1

    chain.provider_rec["capacity"] = "3"     # reputation grew: the grant moves with it
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 3
    assert metrics.granted == 3

    chain.provider_rec["capacity"] = "9"     # a grant above capacity is capacity
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 4


async def test_the_grant_bounds_claims_not_only_the_poll():
    clock = Clock()
    chain = GreedyNode([open_text_job("a"), open_text_job("b")], clock, provider_rec=_granted(1))
    sched = make_scheduler(make_config([text_model()], capacity=4), chain, FakeDriver(), clock)
    sched._inflight = 1             # the one granted slot is taken
    await sched.run_once()
    await sched.join()
    assert chain.settled == []
    assert all(j.state == "Open" for j in chain._jobs.values())


async def test_a_record_without_a_grant_leaves_capacity_alone():
    clock = Clock()
    chain = FakeNode([], clock)     # the default record names no capacity
    sched = make_scheduler(make_config([text_model()], capacity=4), chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 4


async def test_an_unreadable_record_keeps_the_last_grant():
    clock = Clock()
    chain = BlinkingNode([], clock, provider_rec=_granted(2))
    sched = make_scheduler(make_config([text_model()], capacity=4), chain, FakeDriver(), clock)
    await sched.run_once()
    chain.fail_next = True
    await sched.run_once()
    assert [q["free"] for q in chain.job_queries][-2:] == [2, 2]


class HangingDriver:
    async def run(self, job, input, *, timeout_s=None):
        await asyncio.Event().wait()

    async def healthy(self):
        return True


async def test_an_attempt_never_outlives_the_sla_wall():
    """`timeout_s` is a read ceiling — one silence. A streamed answer that keeps
    producing has no other end, so the attempt as a whole ends at the deadline."""
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()]), chain, HangingDriver(), clock, metrics)
    await sched.run_once()                 # claims; the run starts on the next tick
    clock.advance(3600 - 60 - 1)           # one second of budget left
    await sched.join()
    assert len(chain.failed) == 1
    assert metrics.fails == ["backend_exhausted"]


async def test_the_filter_carries_the_loosest_floor_across_a_models_windows():
    """One request, several prices.

    A filter tighter than the loosest window's floor silently drops bids that
    window would have accepted — work lost, with no error anywhere.
    """
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([_two_window_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1]["min_rate_out"] == 300_000   # the 24h window, not the 1h one
    assert chain.job_queries[-1]["min_rate_in"] == 100_000


async def test_a_model_that_meters_no_input_side_sends_no_input_floor():
    clock = Clock()
    chain = FakeNode([], clock)
    sched = make_scheduler(make_config([_output_only_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1] == {
        "model_id": MODEL_IDS[MODEL], "free": sched._capacity,
        "min_rate_out": 600_000, "min_rate_in": None,
    }


async def test_one_unmetered_window_opens_the_input_filter_for_the_whole_model():
    """Still the loosest floor: a window that meters no input accepts a bid
    paying nothing for it, so no input filter may be sent at all."""
    clock = Clock()
    chain = FakeNode([], clock)
    model = _output_only_model({"1h": SlaRate(rate_in="0.2", rate_out="0.6"),
                                "24h": SlaRate(rate_in=None, rate_out="0.3")})
    sched = make_scheduler(make_config([model]), chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1]["min_rate_in"] is None
    assert chain.job_queries[-1]["min_rate_out"] == 300_000


async def test_an_underpriced_bid_the_node_returned_anyway_is_still_refused():
    """The filter is an optimisation and never a security boundary.

    A node that ignored it, or answered against a stale floor, costs one wasted
    row — never a bad claim.
    """
    clock = Clock()
    cheap = _bid(1, 1, tag="server_should_have_filtered_me")
    chain = DisobedientNode([cheap], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)

    await sched.run_once()
    await sched.join()

    assert chain.settled == []
    assert chain.ops == []          # not even a claim was attempted
    assert cheap.state == "Open"


# --- per-model limits: throttling and retries ---------------------------------


async def test_a_retryable_failure_is_retried_after_the_backoff_and_settles():
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FlakyDriver([BackendError("HTTP 503", retryable=True)])
    sched = make_scheduler(make_config([text_model(retries=2, retry_backoff_s=30)]),
                           chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    assert driver.calls == 2
    assert sched.sleeps == [30]
    assert metrics.retries == [MODEL]
    assert [s[0] for s in chain.settled] == [job.job_id]
    assert chain.failed == [] and metrics.fails == []


async def test_a_retry_after_longer_than_the_backoff_is_honoured():
    clock = Clock()
    chain = FakeNode([open_text_job()], clock)
    driver = FlakyDriver([BackendError("HTTP 429", retryable=True, retry_after_s=120)])
    sched = make_scheduler(make_config([text_model(retries=1, retry_backoff_s=30)]),
                           chain, driver, clock)
    await sched.run_once()
    await sched.join()
    assert sched.sleeps == [120]
    assert len(chain.settled) == 1


async def test_a_non_retryable_failure_is_not_retried_whatever_the_retries():
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FailingDriver(retryable=False)
    sched = make_scheduler(make_config([text_model(retries=5)]), chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    assert driver.calls == 1
    assert sched.sleeps == []
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["backend_error"]


async def test_exhausting_the_attempts_fails_the_job_back():
    clock = Clock()
    job = open_text_job()
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FailingDriver(retryable=True)
    sched = make_scheduler(make_config([text_model(retries=2, retry_backoff_s=10)]),
                           chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    assert driver.calls == 3
    assert sched.sleeps == [10, 20]
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["backend_exhausted"]
    assert metrics.retries == [MODEL, MODEL]


async def test_retries_stop_before_the_deadline_and_fail_the_job_back():
    """A 1h job with a slow, failing backend: attempts are spaced by the
    doubling backoff and the loop gives up the moment the next wait would end
    past `deadline - safety_margin_s`, never sleeping through it."""
    clock = Clock()
    job = open_text_job()            # claimed at t=1000 → deadline 4600, margin 60
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FailingDriver(retryable=True)

    # every attempt burns 300s of clock before it fails
    class SlowFailing(FailingDriver):
        async def run(self, job, input, *, timeout_s=None):
            clock.advance(300)
            return await super().run(job, input, timeout_s=timeout_s)
    driver = SlowFailing(retryable=True)
    sched = make_scheduler(make_config([text_model(retries=20, retry_backoff_s=30)]),
                           chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    assert sched.sleeps == [30, 60, 120, 240, 480]
    assert driver.calls == 6
    assert clock() <= 1000 + 3600 - 60
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["deadline_wait"]


async def test_a_throttle_wait_that_would_pass_the_deadline_fails_the_job_back():
    clock = Clock()
    job = open_text_job(sla="2m")    # deadline - margin = 60s after the claim
    chain = FakeNode([job], clock)
    metrics = FakeMetrics()
    driver = FailingDriver(retryable=True)
    sched = make_scheduler(
        make_config([text_model(sla="2m", retries=1, retry_backoff_s=10, rate_limit={"1h": 1})]),
        chain, driver, clock, metrics)
    await sched.run_once()
    await sched.join()
    # one attempt spent the hour's only slot; the retry would wait an hour
    assert driver.calls == 1
    assert sched.sleeps == [10]
    assert chain.failed == [job.job_id]
    assert metrics.fails == ["deadline_wait"]


async def test_the_attempt_timeout_is_the_entrys_ceiling_within_the_budget():
    clock = Clock()
    chain = FakeNode([open_text_job()], clock)
    driver = FlakyDriver([])
    sched = make_scheduler(make_config([text_model(timeout_s=310)]), chain, driver, clock)
    await sched.run_once()
    await sched.join()
    assert driver.timeouts == [310]


async def test_a_recovered_job_without_a_claim_stamp_is_treated_as_claimed_now():
    clock = Clock()
    sched = make_scheduler(make_config([text_model()]), FakeNode([], clock), FakeDriver(), clock)
    job = open_text_job()
    job.claimed_at = None
    assert sched._deadline(job) == clock() + 3600 - 60


async def test_free_is_capped_by_the_models_rate_window():
    """The poll's `free` is per model and never more than the tightest window
    admits now — so the coordinator is not offered work the daemon could not
    start inside the lease."""
    clock = Clock()
    jobs = [open_text_job("rl_1"), open_text_job("rl_2"), open_text_job("rl_3")]
    chain = FakeNode(jobs, clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model(rate_limit={"1m": 2})], capacity=4),
                           chain, FakeDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()
    assert chain.job_queries[-1]["free"] == 2
    assert len(chain.settled) == 2

    await sched.run_once()                       # the minute is spent
    assert chain.job_queries[-1]["free"] == 0
    assert metrics.model_free[MODEL] == 0
    assert len(chain.settled) == 2

    clock.advance(61)
    await sched.run_once()
    await sched.join()
    assert chain.job_queries[-1]["free"] == 2
    assert len(chain.settled) == 3


async def test_a_retry_spends_a_window_slot_the_next_sweep_no_longer_offers():
    clock = Clock()
    chain = FakeNode([open_text_job("w_1")], clock)
    driver = FlakyDriver([BackendError("HTTP 503", retryable=True)])
    sched = make_scheduler(make_config([text_model(retries=1, retry_backoff_s=1, rate_limit={"1m": 3})]),
                           chain, driver, clock)
    await sched.run_once()
    await sched.join()
    assert driver.calls == 2
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 1     # 3 - 2 attempts started


async def test_free_is_per_model_so_a_throttled_model_does_not_starve_a_sibling():
    clock = Clock()
    chain = FakeNode([], clock)
    throttled = text_model("model-a:fp8", rate_limit={"1m": 1})
    open_model = text_model("model-b:fp8")
    sched = make_scheduler(make_config([throttled, open_model], capacity=4), chain, FakeDriver(), clock)
    sched._throttles["model-a:fp8"].acquire()     # its minute is spent
    await sched.run_once()
    frees = {q["model_id"]: q["free"] for q in chain.job_queries}
    assert frees == {MODEL_IDS["model-a:fp8"]: 0, MODEL_IDS["model-b:fp8"]: 4}


async def test_concurrency_holds_the_daemon_to_one_job_at_a_time_for_the_entry():
    clock = Clock()
    jobs = [open_text_job("c_1"), open_text_job("c_2")]
    chain = FakeNode(jobs, clock)
    sched = make_scheduler(make_config([text_model(concurrency=1)], capacity=4),
                           chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 1
    assert sched._inflight == 1                 # the second row was not claimed
    await sched.join()
    await sched.run_once()
    await sched.join()
    assert len(chain.settled) == 2


async def test_a_window_at_the_sla_claims_beyond_concurrency_and_runs_the_jobs_in_turn():
    """A `rate_limit` window at the entry's SLA is what it may hold: the poll
    offers that window's budget, every claimed job waits for an in-flight slot,
    and the backend never sees more than `concurrency` at once."""
    class GatedDriver:
        def __init__(self):
            self.active = self.max_active = 0
            self.release = asyncio.Event()

        async def run(self, job, input, *, timeout_s=None):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await self.release.wait()
            self.active -= 1
            return Normalized(kind="text", text="answer", completion_tokens=42, raw={"choices": []})

        async def healthy(self):
            return True

    clock = Clock()
    chain = FakeNode([open_text_job("q_1"), open_text_job("q_2"), open_text_job("q_3")], clock)
    driver = GatedDriver()
    sched = make_scheduler(make_config([text_model(concurrency=1, rate_limit={"1h": 3})],
                                       capacity=4), chain, driver, clock)
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 3
    assert sched._inflight == 3                          # all three claimed
    for _ in range(5):
        await asyncio.sleep(0)
    assert driver.active == 1                            # one on the wire, two queued
    throttle = sched._throttles[MODEL]
    assert (throttle.held, throttle.inflight, throttle.claimable()) == (3, 1, 0)
    driver.release.set()
    await sched.join()
    assert driver.max_active == 1
    assert len(chain.settled) == 3


async def test_the_sla_window_bounds_free_after_the_waiting_jobs():
    """A waiting job will start inside the window at its SLA, so it reserves one
    of that window's attempts at the claim; the shorter windows only pace starts."""
    clock = Clock()
    jobs = [open_text_job(f"b_{i}", sla="24h") for i in range(4)]
    chain = FakeNode(jobs, clock)
    sched = make_scheduler(
        make_config([text_model(sla="24h", concurrency=1, rate_limit={"1m": 1, "24h": 3})],
                    capacity=8), chain, FakeDriver(), clock)
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 3            # the day's budget, not the minute's
    assert sched._inflight == 3
    await sched.join()                                   # paced one a minute by the sleeps
    assert len(chain.settled) == 3
    await sched.run_once()
    assert chain.job_queries[-1]["free"] == 0            # 3 started today; nothing left to claim
    assert len(chain.settled) == 3


async def test_a_job_waiting_to_retry_still_holds_its_concurrency_slot():
    clock = Clock()
    chain = FakeNode([open_text_job("h_1"), open_text_job("h_2")], clock)
    driver = FlakyDriver([BackendError("HTTP 503", retryable=True)])
    sched = make_scheduler(make_config([text_model(concurrency=1, retries=1, retry_backoff_s=30)],
                                       capacity=4), chain, driver, clock)
    await sched.run_once()
    throttle = sched._throttles[MODEL]
    assert throttle.held == 1 and throttle.claimable() == 0
    await sched.join()
    assert throttle.held == 0 and throttle.claimable() == 1


# --- in-flight handles survive a restart ---------------------------------------


async def test_a_submitted_handle_is_on_record_only_while_the_backend_runs():
    clock = Clock()
    job = open_text_job("handle")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    seen = {}
    driver = ResumableDriver(on_run=lambda j: seen.update(row=store.get(j.job_id)))
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, store=store)
    await sched.run_once()
    await sched.join()

    assert seen["row"].handle == "resp_abc" and seen["row"].model == MODEL
    assert store.all() == []                                   # gone once the job settled
    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_a_failed_job_leaves_no_handle_behind():
    clock = Clock()
    job = open_text_job("handle_fail")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    driver = ResumableDriver(fail=BackendError("backend reported failure status: 'failed'"))
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, store=store)
    await sched.run_once()
    await sched.join()

    assert store.all() == []
    assert chain.failed == [job.job_id]


async def test_boot_recovery_resumes_a_job_whose_handle_is_on_record():
    clock = Clock()
    job = claimed_text_job("resume")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    store.put(job.job_id, MODEL, "resp_abc")
    driver = ResumableDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, store=store)
    await sched.run_once()
    await sched.join()

    assert driver.runs == ["resp_abc"]                          # polled, never resubmitted
    assert [s[0] for s in chain.settled] == [job.job_id]
    assert store.all() == []


async def test_boot_recovery_reruns_a_job_whose_handle_belongs_to_another_model():
    # The operator moved the model to a different backend between boots: the
    # old handle means nothing there, so the job runs from scratch.
    clock = Clock()
    job = claimed_text_job("moved")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    store.put(job.job_id, "other-model:fp8", "resp_old")
    driver = ResumableDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, store=store)
    await sched.run_once()
    await sched.join()

    assert driver.runs == [None]
    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_boot_recovery_forgets_handles_whose_jobs_are_no_longer_claimed():
    clock = Clock()
    chain = FakeNode([], clock)                                 # nothing of ours is claimed
    store = InflightStore(None)
    store.put("0xgone", MODEL, "resp_stale")
    sched = make_scheduler(make_config([text_model()]), chain, ResumableDriver(), clock, store=store)
    await sched.run_once()
    await sched.join()

    assert store.all() == []


async def test_boot_recovery_keeps_handles_when_the_listing_cannot_be_read():
    clock = Clock()
    job = claimed_text_job("kept")

    class FlakyChain(FakeNode):
        fault = True

        async def list_claimed_jobs(self, provider):
            jobs = await super().list_claimed_jobs(provider)
            if self.fault:
                self.fault = False
                raise httpx.ConnectError("coordinator unreachable")
            return jobs

    chain = FlakyChain([job], clock)
    store = InflightStore(None)
    store.put(job.job_id, MODEL, "resp_abc")
    driver = ResumableDriver()
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, store=store)
    await sched.run_once()                                      # the read fails; nothing is forgotten
    await sched.join()
    assert store.get(job.job_id) is not None

    await sched.run_once()
    await sched.join()
    assert driver.runs == ["resp_abc"]


async def test_stop_suspends_a_job_with_a_handle_on_record_and_frees_its_slot():
    clock = Clock()
    job = open_text_job("suspend")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([text_model()], capacity=1), chain,
                           ResumableDriver(hang=True), clock, metrics, store=store)
    await sched.run_once()
    for _ in range(5):                                          # let the task reach its first await
        await asyncio.sleep(0)
    assert sched._inflight == 1
    assert store.get(job.job_id).handle == "resp_abc"

    await sched.stop()
    await asyncio.wait_for(sched.join(), timeout=1)             # returns at once

    assert store.get(job.job_id).handle == "resp_abc"           # kept for the next boot
    assert sched._inflight == 0
    assert sched._throttles[MODEL].held == 0
    assert chain.settled == [] and chain.failed == []           # still ours, still Claimed
    assert job.state == "Claimed"
    assert metrics.fails == []


async def test_stop_drains_a_failure_report_after_the_backend_phase():
    # A job whose backend just failed has left the backend phase: a stop that
    # lands during its failure report must let the report finish, not cancel it.
    # The node holds the fail op open so the stop lands inside the report every
    # run, instead of whenever the loop happens to interleave.
    class GatedNode(FakeNode):
        """Holds the `fail` op open until the test lets it through."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.reporting = asyncio.Event()
            self.release = asyncio.Event()

        async def push_op(self, op, payload, signature):
            if op == "fail":
                self.reporting.set()
                await self.release.wait()
            return await super().push_op(op, payload, signature)

    clock = Clock()
    job = open_text_job("report")
    chain = GatedNode([job], clock)
    store = InflightStore(None)
    driver = ResumableDriver(fail=BackendError("backend reported failure status: 'failed'"))
    sched = make_scheduler(make_config([text_model()]), chain, driver, clock, store=store)
    await sched.run_once()
    await asyncio.wait_for(chain.reporting.wait(), timeout=1)
    await sched.stop()
    chain.release.set()
    await sched.join()
    assert chain.failed == [job.job_id]
    assert store.all() == []


async def test_a_crash_between_backend_completion_and_settle_keeps_the_handle():
    # The backend is done and its handle still names a finished response. Until
    # the settle lands, that row is the only thing that would let a next boot
    # re-poll the answer instead of paying for it twice — so it survives the end
    # of the backend phase, and a stop landing here drains rather than cancels.
    class GatedNode(FakeNode):
        """Holds the `settle` op open until the test lets it through."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.settling = asyncio.Event()
            self.release = asyncio.Event()

        async def push_op(self, op, payload, signature):
            if op == "settle":
                self.settling.set()
                await self.release.wait()
            return await super().push_op(op, payload, signature)

    clock = Clock()
    job = open_text_job("mid_settle")
    chain = GatedNode([job], clock)
    store = InflightStore(None)
    sched = make_scheduler(make_config([text_model()]), chain, ResumableDriver(), clock, store=store)
    await sched.run_once()
    await asyncio.wait_for(chain.settling.wait(), timeout=1)

    assert sched._in_backend == set()                           # past the backend phase
    assert store.get(job.job_id).handle == "resp_abc"           # and still on record
    await sched.stop()
    assert not sched._running[job.job_id].cancelled()           # the settle is drained, not killed

    chain.release.set()
    await sched.join()
    assert [s[0] for s in chain.settled] == [job.job_id]
    assert store.all() == []                                    # dropped once the job exited


async def test_a_retry_after_a_resumed_attempt_submits_afresh_and_records_the_new_handle():
    clock = Clock()
    job = claimed_text_job("retry_resume")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    store.put(job.job_id, MODEL, "resp_old")

    class FirstPollFails(ResumableDriver):
        async def run(self, job, input, *, timeout_s=None, resume=None, on_handle=None):
            self.runs.append(resume)
            if resume is not None:
                raise BackendError("backend poll timed out", retryable=True)
            on_handle("resp_new")
            return Normalized(kind="text", text="answer", completion_tokens=42, raw={"choices": []})

    driver = FirstPollFails()
    sched = make_scheduler(make_config([text_model(retries=1)]), chain, driver, clock, store=store)
    await sched.run_once()
    await sched.join()
    assert driver.runs == ["resp_old", None]
    assert [s[0] for s in chain.settled] == [job.job_id]
    assert store.all() == []


async def test_stop_drains_a_job_without_a_handle():
    clock = Clock()
    job = open_text_job("drain")
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.stop()
    await sched.join()
    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_a_sync_driver_is_run_with_its_old_signature():
    # FakeDriver.run takes only timeout_s: a driver that cannot resume is never
    # handed the resume kwargs.
    clock = Clock()
    job = open_text_job("plain")
    chain = FakeNode([job], clock)
    sched = make_scheduler(make_config([text_model()]), chain, FakeDriver(), clock)
    await sched.run_once()
    await sched.join()
    assert [s[0] for s in chain.settled] == [job.job_id]


async def test_stop_suspends_a_job_running_on_the_window_s_own_backend():
    # The two features meet here: the window's override is the async backend, and
    # a stop mid-flight has to suspend the job running on *it*, not on the
    # model's default. The handle on record is what the next boot resumes from.
    clock = Clock()
    job = open_text_job("suspend_window", sla="24h")
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    override, default = ResumableDriver(hang=True), FakeDriver()
    sched = make_scheduler(make_config([_windowed_model({"24h": _BATCH_BACKEND})], capacity=1),
                           chain, default, clock, store=store)
    sched._sla_drivers[(MODEL, "24h")] = override

    await sched.run_once()
    for _ in range(5):                                          # let the task reach its first await
        await asyncio.sleep(0)
    assert override.runs == [None] and default.calls == 0       # the window's backend ran it
    assert sched._in_backend == {job.job_id}
    assert store.get(job.job_id).handle == "resp_abc"

    await sched.stop()
    await asyncio.wait_for(sched.join(), timeout=1)

    assert store.get(job.job_id).handle == "resp_abc"           # kept for the next boot
    assert sched._inflight == 0
    assert chain.settled == [] and chain.failed == []
    assert job.state == "Claimed"


async def test_boot_recovery_resumes_a_windowed_job_on_the_window_s_own_backend():
    # The restart half of the same pair: the row says nothing about which backend
    # ran the job, so the window is what has to pick the driver again — handing
    # a batch handle to the model's default backend would resubmit the work.
    clock = Clock()
    job = open_text_job("resume_window", sla="24h")
    job.state, job.provider, job.claimed_at = "Claimed", PROVIDER_ID, 1000
    chain = FakeNode([job], clock)
    store = InflightStore(None)
    store.put(job.job_id, MODEL, "resp_abc")
    override, default = ResumableDriver(), FakeDriver()
    sched = make_scheduler(make_config([_windowed_model({"24h": _BATCH_BACKEND})]),
                           chain, default, clock, store=store)
    sched._sla_drivers[(MODEL, "24h")] = override

    await sched.run_once()
    await sched.join()

    assert override.runs == ["resp_abc"]                        # polled, never resubmitted
    assert default.calls == 0
    assert [s[0] for s in chain.settled] == [job.job_id]
    assert store.all() == []


# --- a reference that outruns what the order paid for -------------------------


async def test_a_job_whose_reference_outruns_its_units_is_handed_back():
    """End to end: the client declared a thumbnail and sealed a photograph.

    The input leg is already priced at the declaration and cannot be re-billed, so
    the remedy is to hand the job back — which inside the grace window refunds the
    client in full and costs this provider nothing. It is metered under its own
    label because it is the caller's doing, not the backend's.
    """
    from tests.test_media_decode import png

    clock = Clock()
    big = base64.b64encode(png(1920, 1080)).decode()
    job_id, task_cid = pin_task({
        "prompt": "a cat",
        # 64x64 declared, 1920x1080 sealed.
        "image": {"b64": big, "media_type": "image/png", "width": 64, "height": 64},
        "width": 1024, "height": 1024, "num_images": 1,
    })
    job = EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                 owner=OWNER, rate_in=None, rate_out=20_000, units_in=64 * 64,
                 units_out=ONE_MEGAPIXEL, task_cid=task_cid)
    chain = BlindChain([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([media_model(modality="image")]), chain,
                           MediaDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert metrics.fails == ["media_input_refused"]
    assert [payload["job_id"] for payload in chain.pushed("fail")] == [job.job_id]
    assert chain.settled == []
    assert sched._inflight == 0          # the slot is released, not leaked


async def test_a_job_whose_reference_matches_its_declaration_runs():
    """The other half: an honest reference is not in anybody's way."""
    from tests.test_media_decode import png

    clock = Clock()
    raw = base64.b64encode(png(256, 256)).decode()
    job_id, task_cid = pin_task({
        "prompt": "a cat",
        "image": {"b64": raw, "media_type": "image/png", "width": 256, "height": 256},
        "width": 1024, "height": 1024, "num_images": 1,
    })
    job = EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                 owner=OWNER, rate_in=None, rate_out=20_000, units_in=256 * 256,
                 units_out=ONE_MEGAPIXEL, task_cid=task_cid)
    chain = BlindChain([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([media_model(modality="image")]), chain,
                           MediaDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert [s[0] for s in chain.settled] == [job.job_id]
    assert metrics.fails == []


async def test_a_request_that_cannot_be_priced_is_handed_back_not_crashed():
    """`width: "abc"` is a hand-built order — both SDKs refuse it before signing.

    Pricing it raises a plain ValueError, and anything that is not a BackendError
    lands in the crash handler, which reports no `fail`: the claim then sits until
    the SLA reclaims it, at this provider's expense. It is the client's input, so
    it is the client's refusal — and the message names the error's kind only,
    because its text quotes the input and this reason goes on the network.
    """
    clock = Clock()
    job_id, task_cid = pin_task({"prompt": "a cat", "width": "abc-secret", "height": 1024})
    job = EvmJob(job_id=job_id, model=MEDIA_MODEL, state="Open", sla="24h", created_at=1000,
                 owner=OWNER, rate_in=None, rate_out=20_000, units_in=0,
                 units_out=ONE_MEGAPIXEL, task_cid=task_cid)
    chain = BlindChain([job], clock)
    metrics = FakeMetrics()
    sched = make_scheduler(make_config([media_model(modality="image")]), chain,
                           MediaDriver(), clock, metrics)
    await sched.run_once()
    await sched.join()

    assert metrics.fails == ["media_input_refused"]
    failed = chain.pushed("fail")
    assert [payload["job_id"] for payload in failed] == [job.job_id]
    assert "abc-secret" not in json.dumps(failed)


# --- a result says what was delivered, not what was priced -----------------------


async def test_a_delivered_clip_is_labelled_and_settled_by_its_own_header():
    """Measured on a real backend: a job priced 640x480 for 4 s came back 752x560.

    The frame table is a pricing convention and a model renders what it renders, so
    the label is read off the clip. A result that repeated the priced numbers would
    tell the client a lie about the file in its hands, and a settle that used them
    would bill a size nobody delivered.
    """
    from tests.test_media_decode import media, movie, tkhd

    clip = movie(tkhd(752, 560) + media(timescale=12288, ticks=49664, coded=(752, 560)),
                 timescale=1000, ticks=4042)
    normalized = _normalized_media(media_blobs=[clip], width=640, height=480,
                                   content_type="video/mp4", seed=None, duration_secs=4)
    sched = _scheduler_with_backend_media(normalized, modality="video")

    sealed, units = await sched._build_result(uncapped_media_job(), normalized, RESULT_KEY)

    assert units == 752 * 560 * 4
    video = json.loads(_open_sealed(sealed, RESULT_SECRET))["video"]
    assert (video["width"], video["height"], video["duration_secs"]) == (752, 560, 4)


async def test_a_length_the_model_chose_is_read_off_the_clip():
    """The job was run with the backend's own "you choose" value. What settles is
    the clip's real length — there is no priced number to fall back on."""
    from tests.test_media_decode import movie, tkhd

    clip = movie(tkhd(854, 480), timescale=1000, ticks=9000)
    normalized = _normalized_media(media_blobs=[clip], width=854, height=480,
                                   content_type="video/mp4", seed=None, duration_secs=-1)
    sched = _scheduler_with_backend_media(normalized, modality="video")
    _, units = await sched._build_result(uncapped_media_job(), normalized, RESULT_KEY)
    assert units == 854 * 480 * 9

    unreadable = _normalized_media(media_blobs=[b"not a clip"], width=854, height=480,
                                   content_type="video/mp4", seed=None, duration_secs=-1)
    with pytest.raises(BackendError, match="length"):
        await sched._build_result(uncapped_media_job(), unreadable, RESULT_KEY)


async def test_a_delivered_image_is_labelled_by_its_own_header_whatever_it_was_served_as():
    from tests.test_media_decode import png

    normalized = _normalized_media(media_blobs=[png(1536, 1024)], width=1024, height=1024,
                                   content_type=None, seed=None)
    sched = _scheduler_with_backend_media(normalized)
    sealed, units = await sched._build_result(uncapped_media_job(), normalized, RESULT_KEY)
    assert units == 1536 * 1024
    image = json.loads(_open_sealed(sealed, RESULT_SECRET))["images"][0]
    assert (image["width"], image["height"], image["content_type"]) == (1536, 1024, "image/png")


async def test_a_render_larger_than_its_cap_states_the_units_that_settle():
    """The label says what was delivered; the stamp says what was charged. The
    client costs its result on the stamp, so its figure and the chain's are one."""
    from tests.test_media_decode import png

    normalized = _normalized_media(media_blobs=[png(2048, 2048)], width=1024, height=1024,
                                   content_type="image/png", seed=None)
    sched = _scheduler_with_backend_media(normalized)
    job = media_job()
    job.units_out = 1024 * 1024
    sealed, units = await sched._build_result(job, normalized, RESULT_KEY)
    payload = json.loads(_open_sealed(sealed, RESULT_SECRET))
    assert units == payload["units"] == 1024 * 1024
    assert payload["images"][0]["width"] == 2048
