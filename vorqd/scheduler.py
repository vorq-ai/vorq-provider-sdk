"""The daemon's core loop (architecture.md §Daemon loop).

Each poll sweep, per served model: poll Open jobs, apply the profitability +
capacity filter, **fetch the container by CID from the storage network's public
gateway and verify it against the job's commitment before claiming anything**,
simulate the claim, push a signed ``claim`` op (a refusal is a lost race and a
skip), recover the DEK from the container's ``seed_wrap`` — this daemon's own box
key on a designated bid, deriving the working key from the seed it unseals; the
coordinator's escrow on an open one, which answers with the key already derived —
decrypt, open the client's envelope and check it was addressed to this job's
owner, then run the job as a background coroutine holding a capacity slot:
execute, normalize, seal the result back to the envelope's key, and push a signed
``settle`` op **carrying the sealed bytes**, from which the node mints the result
CID it then names on chain. A job that overruns its SLA (minus the safety margin)
is abandoned, not settled. An unhealthy backend withdraws its asks and
republishes on recovery; a backend that fails ``trip_after`` jobs in a row is
withdrawn from the job that tripped it and re-listed after a cooldown. Every
failure path frees its slot.

**The daemon speaks to a coordinator node and to nothing else.** Startup is a
session handshake, the catalog (which binds every configured model name to the
``uint32`` the chain uses), the provider record, a signed ``set_identity`` on a
confidential boot, a signed ``request_capacity``, and the first ask snapshot.
Everything after that is a signed artifact the node relays and pays for: this
process holds no RPC, no nonce, no gas and no transaction.

**Asks are push-only.** A snapshot is the provider's *whole* book, signed once
and landed on chain immediately, with withdrawals riding inside it as
quotes with both rates at 0 — so there is no TTL, no heartbeat, and nothing to keep alive.

**Verification precedes the claim, and is repeated at the point of use.** Bytes
that do not re-derive the job's own id are never claimed, whatever CID named
them, and ``_open_task`` checks again rather than trusting the fetch that found
them — so a source added later, an in-process cache, or a resolver that skipped
the check can never feed a cipher.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any

import httpx

from ._crypto import BoxCipher, WalletSigner, open_dek, seal_to
from . import media
from .backend import DEFAULT_DIM, Normalized, input_shortfall, plan_media_units
from .blob import BlobError, BlobResolver, default_resolver
from .config import VorqdConfig
from .container import (ContainerError, ciphertext_hash, derive_dek, sealed_plaintext_bytes,
                        verify_container)
from .errors import (
    BackendError,
    BackendExhausted,
    BackendGone,
    ChainConflict,
    ConfigError,
    DeadlineExceeded,
    MediaInputRefused,
    NotRegisteredError,
    OpRefused,
    OpRejected,
    UnknownModel,
    UploadInvalid,
)
from .escrow import ESCROW_KEY_LOST, EscrowRelease, ReleaseRefused
from .limits import Breaker, RetryPolicy, Throttle, retry_delay
from .money import format_usd, parse_usd
from .node import ModelResolver
from .ops import report_job_failed
from .opsig import OpSigner
from .pricing import (
    MAX_FLOOR_DISCOUNT_PCT,
    NullLoadMonitor,
    discounted_units,
    raw_discount_pct,
)
from .state import InflightStore
from .types import INLINE_MAX_BYTES

log = logging.getLogger("vorqd")

# The one envelope version this daemon opens. The client seals
# ``{"v", "owner", "result_key", "input"}``; anything else is not a VORQ task.
ENVELOPE_VERSION = "vorq-env-v1"

# A ``result_key`` is a Curve25519 public key the way nacl's hex decoder wants it:
# 32 bytes as 64 bare hex digits. No ``0x`` — the decoder does not strip one.
_RESULT_KEY_RE = re.compile(r"[0-9a-fA-F]{64}")


def _same_box_key(on_record, local: str) -> bool:
    """Is the record's box key this daemon's own?

    The chain holds a ``bytes32`` and the node serves it as ``0x`` hex, while
    nacl hands out bare hex — so the two spellings of one key are compared after
    normalising, and never as strings. An absent key never matches: a record
    with no key is a provider no client can seal to.
    """
    if not on_record or not isinstance(on_record, str):
        return False
    return on_record.removeprefix("0x").lower() == local.removeprefix("0x").lower()


def _backend_at_fault(exc: BackendError) -> bool:
    """Did the backend fail this job, or was the job refused?

    Exhausted retries are the backend failing every attempt. A deadline spent
    on failed attempts chains the last one (``raise ... from exc``); one spent
    waiting on the throttle chains nothing and is this daemon's own quota. A
    404 or 410 is an endpoint that is not there, whatever the job asked. Any
    other non-retryable refusal is the job's input, not the backend.
    """
    if isinstance(exc, (BackendExhausted, BackendGone)):
        return True
    if isinstance(exc, DeadlineExceeded):
        return isinstance(exc.__cause__, BackendError)
    return False


#: The width of both rates on chain.
UINT128_MAX = 2**128 - 1


def _atomic_rate(value: str | None, decimals: int, model: str, window: str, field: str) -> int:
    """A configured USD rate as the atomic ``uint128`` the ask signs and floors compare.

    The loader checked the grammar; the token's ``decimals`` bound the fraction
    here, the first place they are known. More digits than the token holds is
    refused, never rounded: a rounded ask is a price the operator did not write.
    An unmetered side (``None``) is 0, the convention the order carries too.
    """
    if value is None:
        return 0
    try:
        units = parse_usd(value, decimals)
    except ValueError as exc:
        raise ConfigError(f"model {model!r} sla {window!r}: {field}: {exc}") from None
    if units > UINT128_MAX:
        raise ConfigError(f"model {model!r} sla {window!r}: {field}={value!r} is wider than uint128")
    return units


def _quote(slot: tuple[int, int], rates: tuple[int, int], decimals: int) -> dict:
    """One ask row on the wire: the rates are money, so USD decimal strings;
    every other member is a JSON integer."""
    model_id, sla_secs = slot
    rate_in, rate_out = rates
    return {"model_id": model_id, "sla": sla_secs,
            "rate_in": format_usd(rate_in, decimals), "rate_out": format_usd(rate_out, decimals)}


def _frame_pixels(frame: dict) -> int:
    """One sealed frame's billable pixels — width × height as the frame states them.

    A frame that names neither dimension is priced at the default the request was
    priced against, so this reads a sealed frame exactly as the client SDK reads
    it back: same dimensions, same default, same product.
    """
    return int(frame.get("width") or DEFAULT_DIM) * int(frame.get("height") or DEFAULT_DIM)


class PayloadError(Exception):
    """The delivered task bytes are not a task this daemon may run.

    ``reason`` is the short, loggable cause — it is what the failure is reported
    and metered under, so it must stay a stable token, not a sentence.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class NullMetrics:
    def on_claim(self): ...
    def on_settle(self, duration): ...
    def on_fail(self, reason): ...
    def on_backend_latency(self, model, seconds): ...
    def set_capacity_free(self, n): ...
    def set_asks_published(self, n): ...
    def set_backend_load(self, model, value): ...
    def set_floor_discount(self, model, pct): ...
    def on_probe_failure(self, model): ...
    def on_retry(self, model): ...
    def set_model_free(self, model, n): ...
    def set_capacity_granted(self, n): ...


def _error_code(resp: httpx.Response) -> str | None:
    """The node's ``error.code`` on a refusal, or ``None`` when the body carries none."""
    try:
        return resp.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None


def sla_seconds(window: str) -> int:
    unit = window[-1:]
    try:
        n = int(window[:-1])
    except ValueError:
        return 86_400
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86_400}.get(unit, 3600)


#: How many times a signed artifact is re-signed with a later issue time when the
#: chain answers that its floor already stands at or above ours.
#:
#: ``ProviderRegistry.lastIdentityAt`` / ``lastCapacityAt`` and
#: ``AskRegistry.lastSignedAt`` are **strictly** monotonic, and a daemon that
#: restarts inside the same wall-clock second as its own last op signs the
#: identical ``issuedAt`` and is refused — which for a confidential boot means a
#: freshly generated box key that never reaches the record, and every job sealed
#: to the key it replaced. Nothing off chain can know those floors (the daemon
#: holds no RPC and the provider record does not carry them), so the chain's own
#: refusal is the signal: bump, re-sign, push again.
MAX_MONOTONIC_BUMPS = 3

#: The wait before a settle the node could not relay is sent again: doubling
#: from the first value up to the second, until the job's SLA closes.
SETTLE_RETRY_S = 5.0
SETTLE_RETRY_MAX_S = 60.0

#: ``AskRegistry.MAX_QUOTES``. The contract **skips** a snapshot carrying more,
#: silently, so a provider that priced more slots than this would publish nothing
#: at all and see no error anywhere.
MAX_QUOTES = 64

#: How long a job whose bytes did not resolve is left alone, and the ceiling that
#: the doubling climbs to.
#:
#: **The book re-serves a job every sweep, so without this the daemon re-fetches
#: the same dead name forever.** A task whose ``task_cid`` names nothing costs a
#: full gateway budget (``GATEWAY_ATTEMPTS`` sweeps with a linear backoff, tens of
#: seconds) and that cost is paid *inline*, ahead of every other bid in the page —
#: so one unresolvable job, at a five-second poll interval, is a daemon that
#: mostly sleeps. The first miss is still worth retrying soon, because a fresh pin
#: takes seconds to propagate; the twentieth is not.
UNRESOLVED_RETRY_S = 60.0
UNRESOLVED_RETRY_MAX_S = 900.0

#: How many deferred jobs the skip book holds before the stalest are dropped.
#:
#: An entry is one job id and two numbers, and every entry retires on its own
#: order's ``expires_at`` — but the ids are chosen by whoever posts orders, so the
#: cap is what keeps a flood of unpostable bids from being a memory leak. Dropping
#: an entry only costs one re-fetch.
MAX_DEFERRED_JOBS = 4096



class Scheduler:
    def __init__(self, config: VorqdConfig, node, coordinator, drivers: dict, metrics=None, *,
                 ops: OpSigner | None = None,
                 clock=time.time, http: httpx.AsyncClient | None = None, cipher: BoxCipher | None = None,
                 evidence: dict | None = None, blobs: BlobResolver | None = None,
                 escrow: EscrowRelease | None = None,
                 load=None, sleep=asyncio.sleep, store: InflightStore | None = None,
                 sla_drivers: dict | None = None):
        self._config = config
        # The waits between a job's attempts and ahead of a throttled one, as a
        # seam: tests run them against a fake clock.
        self._sleep = sleep
        # The coordinator node, and the daemon's **only** chain collaborator. It
        # reads the book, the catalog and the provider directory, relays what
        # this daemon signs and pays the gas for it. Nothing here builds, funds,
        # prices or broadcasts a transaction.
        self._node = node
        # Signs the ops. Built lazily from the configured wallet when the wiring
        # did not inject one, so a scheduler can be constructed before a key is
        # needed and a daemon never signs with a second identity.
        self._ops = ops
        self._coord = coordinator
        self._drivers = drivers
        # (model, sla window) -> the backend that window is served through, for
        # the windows an entry gives one. Every other window uses `_drivers`.
        self._sla_drivers = dict(sla_drivers or {})
        self._metrics = metrics or NullMetrics()
        self._clock = clock
        self._http = http
        # Where containers come from: the storage network's public gateway,
        # resolved by the one helper the CLI wiring uses too — one way to choose a
        # gateway, not two. More sources only widen availability; the commitment
        # check, not the ordering, is the trust.
        self._blobs = blobs or (default_resolver(http) if http is not None else BlobResolver([]))
        # job_id -> (not before this wall-clock second, consecutive misses). The
        # book has no memory and re-serves an unclaimable job every sweep, so this
        # is the daemon's own. See ``UNRESOLVED_RETRY_S``.
        self._deferred: dict[str, tuple[float, int]] = {}
        # How an OPEN bid's DEK is obtained: its wrap is sealed to the
        # coordinator's attested escrow key, so the daemon asks the escrow to
        # release it (see vorqd/escrow.py). A designated bid never comes here —
        # its wrap is sealed to this daemon's own box key.
        self._escrow = escrow
        # model name -> modality, read from the curated catalog. The wire job
        # carries no modality: it is a catalog fact, never an order term. Empty
        # until the catalog is read, and a model missing from it is UNKNOWN, not
        # text — see ``_modality``.
        self._modalities: dict[str, str] = {}
        # Config model names and SLA windows ↔ the uint32 ids the chain uses.
        # None until the catalog is read: the job book is addressed by model id,
        # the ask book quotes ids and seconds, and none of that can be spelled
        # before the catalog answers. Bound once, on first need (``_ensure_models``).
        self._models: ModelResolver | None = None
        # The daemon's payload-decryption identity; its public key is published on
        # the registry record so clients can seal to it. Operator-keyed daemons build
        # it from the configured box key — a stable, operator-held identity, never
        # auto-generated. Confidential daemons instead have the ephemeral per-boot
        # cipher injected (with the evidence binding it to this boot), because in
        # that mode the key is generated in guest memory and never configured.
        self._cipher = cipher or (
            BoxCipher(config.provider.box_key) if config.provider.box_key else None
        )
        self._evidence = evidence
        if self._cipher is None:
            raise ConfigError(
                "no payload cipher: configure provider.box_key or inject a boot identity"
            )
        self._capacity = config.provider.capacity
        self._inflight = 0
        # The slots the network grants this provider on its record, read every
        # sweep; None until the record is read (or when it names none).
        self._granted: int | None = None
        # Per-model admission and retry policy, from each entry's `backend:`
        # limits. The throttle's answer is what the poll sends as `free`, so a
        # model whose quota is spent is leased nothing until it frees up.
        self._throttles = {m.model: Throttle.from_model(m, clock) for m in config.models}
        # A queued job has no instant to sleep until: an in-flight slot frees
        # when a running attempt returns, and that is signalled here per model.
        self._slot_freed = {m.model: asyncio.Event() for m in config.models}
        self._policies = {m.model: RetryPolicy.from_backend(m.backend) for m in config.models}
        # A backend that fails job after job is withdrawn from the book, from
        # the job that tripped it, and re-listed after a cooldown.
        self._breakers = {m.model: Breaker.from_backend(m.backend, clock) for m in config.models}
        # One snapshot on the wire at a time: the sweep and a tripped job's own
        # push would otherwise diff the same book and both pay to withdraw it.
        self._asks_lock = asyncio.Lock()
        if config.provider.bid_filter.get("min_age_s") is not None:
            # Accepted so an operator's YAML keeps loading; inert since the
            # coordinator's matcher decides which bids this daemon is offered.
            log.warning("provider.bid_filter.min_age_s is ignored: the coordinator's lease "
                        "matcher decides which bids this daemon sees")
        self._recovered = False
        self._tasks: set[asyncio.Task] = set()
        # job id -> its running task, so a stop can pick out the ones that
        # may be suspended (their backend attempt is still the live phase) from
        # the ones that must drain.
        self._running: dict[str, asyncio.Task] = {}
        # The jobs whose backend attempt is running or between attempts — the
        # ones a stop may suspend. Not the store: the row outlives the backend
        # phase so a crash during settle re-polls the finished response, and a
        # stop landing there must drain the settle rather than cancel it.
        self._in_backend: set[str] = set()
        # The handles of jobs in flight at an async backend. A daemon built by
        # the CLI persists them to `provider.state_db`; one built in code
        # without a store keeps them in memory, which is no resume at all.
        self._store = store or InflightStore(None)
        # The book as this daemon last pushed it: (model_id, sla_secs) -> (rate_in,
        # rate_out), the rates as the atomic integers they were signed as. It is
        # the live set and not a name set because a withdrawal has to be *pushed*:
        # on chain the write is an upsert, so a slot a snapshot omits keeps its
        # old price forever and only a quote with both rates at 0 deletes one.
        self._published: dict[tuple[int, int], tuple[int, int]] = {}
        # How busy each backend is. A daemon with no `load:` block anywhere gets
        # the null monitor, which answers "unknown" for every model — and unknown
        # prices exactly like busy, so the whole feature stays inert.
        #
        # Note what this does NOT touch: the published ask book. The book
        # advertises the operator's configured rates and never moves with load,
        # because a floor the network can see is a floor bids converge onto.
        # Everything below is one private decision, and ``JobRegistry.claim``
        # never consults the book — it charges the rates the client signed.
        self._load = load or NullLoadMonitor()
        # The last issue time this daemon signed anything under. See
        # ``MAX_MONOTONIC_BUMPS``: the registries' floors are strictly monotonic,
        # so two artifacts signed in one second must not carry one timestamp.
        self._issued_at = 0
        # Models the network permits this provider to serve; None until startup,
        # and None thereafter means "no restriction". Excluded models are filtered
        # out of ask publishing, composed with the health gate.
        self._allowed: set[str] | None = None
        self._shutdown = False

    @property
    def provider(self) -> int | None:
        # The admin-issued id, discovered at the session handshake; None before it.
        # Providers never transmit this — it is ambient from the session — but it is
        # still the value compared against a job's designated pin (int == int).
        return self._coord.provider_id

    @property
    def ops(self) -> OpSigner:
        """The op signer, built from the configured wallet on first use.

        Lazy because the wallet key is only needed once there is an op to sign,
        and eager construction would make every scheduler in a test that never
        claims anything depend on a key it never uses.
        """
        if self._ops is None:
            self._ops = OpSigner(WalletSigner(self._config.provider.wallet_key))
        return self._ops

    async def _push(self, op: str, payload: dict, signature: str):
        """One signed op to the node, which simulates it and relays it on ok.

        The daemon never assembles a transaction, never holds a nonce and never
        buys gas: it states what it wants in a typed message the registries
        verify, and the node pays to put it on chain.
        """
        return await self._node.push_op(op, payload, signature)

    @staticmethod
    def _landed(answer, op: str, job_id: str | None = None) -> bool:
        """Whether a relayed op actually mined — read this before believing one.

        The node waits for the receipt before it answers, so ``status`` is the
        **mined** outcome and not the broadcast's. An op can pass the node's
        pre-relay simulate, be broadcast, and still revert: the simulate is
        authoritative only at the instant it runs, and the state it ran against
        can move underneath it. A reverted transaction changed nothing on chain
        whatever the ``201`` said, so a daemon that reads only the absence of a
        refusal believes an op that never happened.

        ``!= "success"`` rather than ``== "reverted"``: an answer carrying no
        status at all reads as ``""`` here, and a receipt this daemon cannot read
        is not one it should act on.
        """
        if answer.status == "success":
            return True

        log.warning("%s mined %s (%s)", op, answer.status or "no status", answer.tx_hash,
                    extra={"job_id": job_id} if job_id else {})
        return False

    # -- monotonic issue times ------------------------------------------------

    def _next_issued_at(self) -> int:
        """The issue time to sign the next artifact under.

        Wall clock, except that it never repeats: two artifacts signed inside one
        second get consecutive values. The registries' floors are strict (``<=``
        reverts), so a repeated timestamp is not a duplicate — it is an op that
        never lands.
        """
        now = int(self._clock())
        self._issued_at = now if now > self._issued_at else self._issued_at + 1
        return self._issued_at

    def _bump_issued_at(self) -> int:
        """One second later than anything this process has signed."""
        self._issued_at += 1
        return self._issued_at

    async def _push_monotonic(self, op: str, build) -> None:
        """Push a ProviderRegistry op, re-signing later while the chain says stale.

        ``build(issued_at)`` returns ``(payload, signature)`` — it is a closure
        rather than a prepared body because the whole point is that the
        timestamp changes **and is re-signed**: ``issuedAt`` is inside the struct
        hash, so re-sending the old signature under a new timestamp recovers a
        stranger.

        The refusal is the only way to learn the floor. A daemon that restarted
        inside the same second as its own last op signs the same ``issuedAt``,
        and no local state survives to say so — the record does not carry the
        floor, and the daemon holds no RPC to read it.
        """
        issued_at = self._next_issued_at()
        for attempt in range(MAX_MONOTONIC_BUMPS):
            payload, signature = build(issued_at)
            try:
                answer = await self._push(op, payload, signature)
            except OpRefused as exc:
                if exc.reason != "StaleOp" or attempt == MAX_MONOTONIC_BUMPS - 1:
                    raise
                issued_at = self._bump_issued_at()
                log.info("%s was refused as stale; re-signing at %d", op, issued_at)
            else:
                # A registry op that reverted put neither the identity nor the
                # capacity on record, so what this daemon believes it published is
                # not what the chain holds. Logged and not raised: `startup` has no
                # handler for it and a boot that dies here serves nothing, while
                # the record it wanted is re-published on the next one.
                self._landed(answer, op)
                return

    # -- lifecycle -----------------------------------------------------------

    async def startup(self) -> None:
        # Wait out admin provisioning: until the operator wallet is registered as a
        # provider, the handshake 403s. Poll until it succeeds so e2e bootstrap
        # ordering (daemon vs. admin registration) is race-free.
        while True:
            try:
                await self._coord.token()
                break
            except NotRegisteredError:
                log.info("not registered with the coordinator; waiting for admin provisioning")
                await asyncio.sleep(self._config.provider.poll_interval_s)

        # The catalog, before anything that has to name a model by number: the
        # job book is polled by model id, the ask book quotes ids and seconds,
        # and a configured model the catalog does not carry is a startup failure
        # rather than a model quietly dropped (Q15).
        resolver = await self._bind_catalog()

        rec = await self._node.get_provider(self._coord.provider_id)
        local_box = self._cipher.public_key

        if self._evidence is not None:
            # Confidential boot: this key was generated seconds ago in guest
            # memory, so the record can never match yet. Publish it as a signed
            # `set_identity` op — the key and the evidence together, since the
            # contract writes both and a key sent alone would replace the stored
            # blob — then wait for the record to reflect it, the same discipline
            # as the not_registered provisioning wait above.
            await self._publish_identity(local_box)
            # Re-read straight after the push, before any sleep: a node that has
            # already indexed the receipt is correct now, and sleeping first
            # would burn a full poll interval on every confidential boot.
            rec = await self._node.get_provider(self._coord.provider_id)
            while not _same_box_key(rec.get("box_key"), local_box):
                log.info("waiting for the registry record to reflect this boot's box key")
                await asyncio.sleep(self._config.provider.poll_interval_s)
                rec = await self._node.get_provider(self._coord.provider_id)
        elif not _same_box_key(rec.get("box_key"), local_box):
            # Operator-keyed mode only: a sealed payload is encrypted to the box key
            # the network holds on record, so a mismatch means we could never open
            # our own jobs — fatal.
            raise ConfigError(
                "box public key on record with the coordinator does not match the local "
                "box key; sealed payloads would be undecryptable"
            )

        self._apply_allowed_models(rec, resolver)
        await self._request_capacity()
        await self.sync_asks()

    async def shutdown(self) -> None:
        """Withdraw the whole book on the way out.

        Every slot this daemon published, re-pushed with both rates at 0 — which
        is the only thing that deletes one. Going quiet is not withdrawal: the ask
        book is chain state with no TTL, so a daemon that simply stopped would
        leave its prices standing and keep being matched with work it is no
        longer running.
        """
        async with self._asks_lock:
            if not self._published:
                return   # nothing was ever published, so there is nothing to take down
            await self._push_snapshot(self._withdrawals(self._published))
            self._published = {}
            self._metrics.set_asks_published(0)

    async def _publish_identity(self, local_box: str) -> None:
        """Publish this boot's box key and its evidence as a signed op.

        ``evidence`` is opaque ``bytes`` on chain, so it travels as hex of its
        canonical JSON: the signature covers bytes, and a second spelling of the
        same object would be a second set of bytes the same signature does not
        cover. Sorted keys and no whitespace, so re-signing the same evidence
        produces the same bytes.
        """
        ctx = await self._node.chain_context()
        box_key = "0x" + local_box.removeprefix("0x")
        evidence = "0x" + json.dumps(
            self._evidence, sort_keys=True, separators=(",", ":")
        ).encode().hex()
        await self._push_monotonic(
            "set_identity",
            lambda at: (
                {"box_key": box_key, "evidence": evidence, "issued_at": at},
                self.ops.sign_set_identity(box_key, evidence, at, ctx),
            ),
        )

    async def _request_capacity(self) -> None:
        """Ask the registry for the slots this daemon may hold: `provider.capacity`,
        which the loader derives from the entries unless the operator set it."""
        ctx = await self._node.chain_context()
        n = int(self._capacity)
        await self._push_monotonic(
            "request_capacity",
            lambda at: ({"n": n, "issued_at": at}, self.ops.sign_request_capacity(n, at, ctx)),
        )

    def _apply_allowed_models(self, rec: dict, resolver: ModelResolver) -> None:
        """Restrict published asks to the models the network permits.

        The record answers ``allow_all_models`` beside a list of model **ids**;
        the flag is the authority and the list is only read when it is false. An
        empty list with the flag set is "no restriction", not "nothing allowed" —
        reading the list alone would silently withdraw every ask this provider
        has.
        """
        if rec.get("allow_all_models", True):
            self._allowed = None
            return
        allowed: set[str] = set()
        for model_id in rec.get("allowed_models") or []:
            try:
                allowed.add(resolver.model_name(model_id))
            except UnknownModel:   # permitted, but not a model this daemon serves
                continue
        self._allowed = allowed
        for model in self._config.models:
            if model.model not in allowed:
                log.warning("model %s is not permitted by the network; excluded from asks", model.model)

    async def _ensure_models(self) -> ModelResolver:
        """The bound catalog, read once. Every poll and every ask needs it."""
        if self._models is None:
            return await self._bind_catalog()
        return self._models

    async def _bind_catalog(self) -> ModelResolver:
        """Read the curated catalog, resolve the config against it, bind it.

        Fatal on a configured model the catalog does not carry: it has no model
        id, so it cannot be polled for, quoted or claimed, and skipping it would
        leave the operator running at a fraction of the capacity they chose with
        no error anywhere (Q15).
        """
        catalog = await self._node.get_models()
        resolver = ModelResolver.resolve(self._config, catalog)
        self._models = resolver
        self._node.bind_models(resolver)
        self._absorb_modalities(catalog)
        return resolver

    def _absorb_modalities(self, catalog) -> None:
        self._modalities = {m.name: m.modality for m in catalog if m.modality}

    async def _load_catalog(self) -> None:
        """Re-read the catalog for the modalities it may have gained.

        Never fatal, and never a default: a catalog this call could not read
        leaves the affected models UNKNOWN, and an unknown modality is refused
        rather than assumed (see :meth:`_modality`). Sweeps retry, so a transient
        outage costs polls, not correctness. The model **ids** are not re-read —
        they are chain facts that do not move under a running daemon, and
        re-resolving them here would make a transient outage fatal.
        """
        try:
            self._absorb_modalities(await self._node.get_models())
        except Exception as exc:  # noqa: BLE001 — availability, not correctness
            log.warning("could not read the model catalog (%s); modalities stay unknown", exc)

    def _modality(self, model_id: str) -> str | None:
        """This model's modality, or ``None`` when nothing can establish it.

        The curated catalog is the authority; a local ``modality:`` declaration in
        the model's config is the fallback that keeps a provider serving through a
        catalog outage. There is deliberately no default: guessing "text" would
        silently skip media unit planning — the clamp to the units the client
        actually paid for, and the guard against a client under-declaring to
        underpay — and settle a media job at the full cap.
        """
        catalogued = self._modalities.get(model_id)
        if catalogued:
            return catalogued
        for model in self._config.models:
            if model.model == model_id and model.modality:
                return model.modality
        return None

    def _unnamed_models(self) -> list[str]:
        return [m.model for m in self._config.models if self._modality(m.model) is None]

    async def run_forever(self) -> None:
        while not self._shutdown:
            try:
                await self.run_once()
            except Exception:  # a sweep-level failure must not kill the loop
                log.exception("poll sweep failed")
            await asyncio.sleep(self._config.provider.poll_interval_s)

    async def stop(self) -> None:
        """Stop claiming, and suspend what can be resumed.

        A job still in its **backend phase** with a handle on record is
        cancelled here: its work keeps running at the backend, and the next boot
        picks it up by the handle (``_recover_claimed``). Everything else — a
        sync attempt with a socket open, a job past its backend phase — is left
        to ``join()`` to drain, as before. The phase is what decides, not the
        row: the row outlives the backend phase so a crash during settle can
        re-poll the finished response, and cancelling on it would kill a settle
        that is already under way.
        """
        self._shutdown = True
        for job_id, task in list(self._running.items()):
            # Both, and in this order: the phase says the cancel is safe, the row
            # says the work survives it. A job past the backend phase keeps its
            # row so a crash re-polls the finished response, and a submit that
            # has not answered yet has no handle to resume from.
            if job_id in self._in_backend and self._store.get(job_id) is not None:
                log.info("suspended", extra={"job_id": job_id})
                task.cancel()

    def close(self) -> None:
        self._store.close()

    async def join(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- ask publishing (health gate) ----------------------------------------

    async def sync_asks(self) -> None:
        """Publish the whole book, once, when it changes.

        **Push-only.** The signed snapshot goes to the node, which lands it on
        chain immediately; there is no TTL and therefore no heartbeat to keep it
        alive. Pushing to this operator's own node *is* the self-publication
        path — and because the artifact is signed rather than authorised by a
        session, any key may land the same bytes through any relayer if this one
        will not, and a duplicate is a no-op under the monotonic ``signedAt``.

        Unchanged means unpushed. Every push spends the node's gas and consumes
        a strictly-increasing floor slot, so re-publishing an identical book
        would pay to change nothing.
        """
        async with self._asks_lock:
            # A trip that lands after `shutdown()` withdrew the book must not
            # publish it again on the way out.
            if self._shutdown:
                return
            resolver = await self._ensure_models()
            decimals = (await self._node.chain_context()).decimals
            healthy: set[str] = set()
            for model in self._config.models:
                if self._breakers[model.model].tripped():
                    continue   # off the book until its cooldown ends
                if await self._backends_healthy(model.model):
                    healthy.add(model.model)
            if self._allowed is not None:
                healthy &= self._allowed   # never offer a model the network forbids
            desired = self._desired_quotes(healthy, resolver, decimals)
            if desired == self._published:
                return
            # The complete current book, and then the slots it drops. A slot this
            # snapshot simply omits keeps its old price on chain — `setAsks` upserts
            # — so a withdrawal has to be an explicit quote with both rates at 0.
            quotes = [_quote(slot, rates, decimals) for slot, rates in sorted(desired.items())]
            quotes += self._withdrawals(
                {slot: rates for slot, rates in self._published.items() if slot not in desired}
            )
            if not await self._push_snapshot(quotes):
                return   # nothing landed, so nothing is recorded as published
            self._published = desired
            self._metrics.set_asks_published(len(desired))

    def _driver_for(self, model_name: str, sla: str | None):
        """The backend that serves this model at this window: the window's own
        override when the entry declares one, else the model's `backend`."""
        return self._sla_drivers.get((model_name, sla)) or self._drivers.get(model_name)

    async def _backends_healthy(self, model_name: str) -> bool:
        """Every backend of the model answers its health check — the default and
        each window's override — so the book never offers a window it cannot serve."""
        driver = self._drivers.get(model_name)
        if driver is None or not await driver.healthy():
            return False
        for (name, _window), extra in self._sla_drivers.items():
            if name == model_name and not await extra.healthy():
                return False
        return True

    def _desired_quotes(
        self, healthy: set[str], resolver: ModelResolver, decimals: int
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """This daemon's whole current book: ``(model_id, sla_secs) -> (in, out)``.

        Free capacity is deliberately **not** in it. The chain's ask row is a
        price and nothing else — the registry has no capacity member — and
        concurrency is the ``request_capacity`` op's business, on a different
        contract. Republishing the book every time a slot filled would also mean
        a signed transaction per claim.
        """
        quotes: dict[tuple[int, int], tuple[int, int]] = {}
        for model in self._config.models:
            if model.model not in healthy:
                continue
            model_id = resolver.model_id(model.model)
            for window, rate in model.slas.items():
                slot = (model_id, resolver.sla_secs(window))
                quotes[slot] = (
                    _atomic_rate(rate.rate_in, decimals, model.model, window, "rate_in"),
                    _atomic_rate(rate.rate_out, decimals, model.model, window, "rate_out"),
                )
        return quotes

    @staticmethod
    def _withdrawals(slots) -> list[dict]:
        return [{"model_id": m, "sla": s, "rate_in": "0", "rate_out": "0"}
                for m, s in sorted(slots)]

    async def _push_snapshot(self, quotes: list[dict]) -> bool:
        """Sign one whole book and hand it to the node.

        The snapshot names its own ``provider_id`` because the contract compares
        it against ``idOf(signer)`` and skips a mismatch: it is what stops a real
        operator publishing prices under somebody else's id, and it is why the
        provider id is in the body here when it is in no other request.

        A ``stale_snapshot`` is the same monotonic collision the ops path has —
        a restart inside the same second as the last push — and is answered the
        same way: bump ``signed_at``, **re-sign**, push again.
        """
        provider_id = self.provider
        if provider_id is None:
            log.warning("no provider id yet; the ask book cannot name its own publisher")
            return False
        if len(quotes) > MAX_QUOTES:
            raise ConfigError(
                f"this snapshot carries {len(quotes)} quotes and the ask registry skips any "
                f"snapshot over {MAX_QUOTES}; the whole book would be dropped silently"
            )
        ctx = await self._node.chain_context()
        signed_at = self._next_issued_at()
        for attempt in range(MAX_MONOTONIC_BUMPS):
            snapshot = {"provider_id": int(provider_id), "signed_at": signed_at, "quotes": quotes}
            try:
                await self._node.push_asks(snapshot, self.ops.sign_ask_snapshot(snapshot, ctx))
                return True
            except ChainConflict as exc:
                if exc.code != "stale_snapshot" or attempt == MAX_MONOTONIC_BUMPS - 1:
                    raise
                signed_at = self._bump_issued_at()
                log.info("the ask snapshot was refused as stale; re-signing at %d", signed_at)
        return False   # unreachable: the last attempt either returns or raises

    # -- polling sweep -------------------------------------------------------

    async def run_once(self) -> None:
        # Nothing in a sweep can be spelled without the catalog: the book is
        # polled by model id and quoted in ids and seconds. A catalog that does
        # not answer costs this sweep and is retried on the next one — a
        # configured model it does not *carry* is a different thing entirely and
        # propagates, because it is a config error and no amount of retrying
        # fixes it.
        try:
            await self._ensure_models()
        except httpx.HTTPError as exc:
            log.warning("the model catalog could not be read (%s); nothing is polled this sweep",
                        type(exc).__name__)
            return
        await self._refresh_load()
        await self._refresh_grant()
        await self.sync_asks()
        # Re-read the catalog while any served model is still unnamed: the
        # catalog need not name a modality at all, and a model whose modality is
        # unknown is one this daemon will not claim for. One GET per sweep, and
        # only while something is actually missing.
        if self._unnamed_models():
            await self._load_catalog()
        # First sweep after boot: pick up what a previous life claimed and never
        # settled. Runs before anything new is claimed, while nothing is in
        # flight — so every Claimed-by-me row is by definition orphaned.
        if not self._recovered:
            self._recovered = True
            await self._recover_claimed()
        # Every model is polled even when no slot is free: the poll is this
        # daemon's presence, and `free` is what it reports.
        for model in self._config.models:
            try:
                await self._poll_model(model)
            except httpx.HTTPError:
                log.warning("poll failed for model %s; will retry", model.model)

    async def _poll_model(self, model) -> None:
        # Fail closed on an unnamed model: without a modality the daemon cannot
        # tell a token-metered job from a pixel-metered one, so it would run media
        # work unclamped and settle it at the full cap. Skip BEFORE the claim — a
        # job never claimed is a job still available to a provider that can price
        # it, whereas claiming and then failing burns the client's order.
        modality = self._modality(model.model)
        if modality is None:
            log.warning("model %s has no known modality (catalog unread and none declared "
                        "in config); not claiming for it this sweep", model.model)
            return

        # Ask only for as many bids as this daemon could take for this model:
        # the tightest of the global slots (the configured capacity, or the
        # network's grant when smaller) and what the entry's throttle admits —
        # what it could start now, or, with a `rate_limit` window at its SLA,
        # what that window can still hold. The coordinator leases this provider
        # the oldest open rows that clear the floors below, at most `free` of
        # them, and records `free` as this daemon's presence for the model —
        # the poll is the heartbeat, which is why a full or throttled daemon
        # polls too: a `free` of 0 on record is what keeps it off a challenge's
        # candidates, and a poll it skipped would leave the last positive
        # `free` standing for a whole liveness window.
        throttle = self._throttles[model.model]
        breaker = self._breakers[model.model]
        free = 0 if breaker.tripped() else max(
            0, min(self._slots() - self._inflight, throttle.claimable())
        )
        self._metrics.set_model_free(model.model, free)

        # One floor per model per sweep. With no per-bid input left, the discount
        # is a function of load alone — so the number that goes out as the filter
        # is the same number that judges every row it brings back.
        pct = self._floor_pct(model)
        decimals = (await self._node.chain_context()).decimals
        # **The loosest floor across this model's windows.** The request carries
        # one rate but a model quotes several, and a filter tighter than the
        # cheapest window's floor silently drops bids that window would have
        # accepted — work lost, with no error anywhere. `profitable` still makes
        # the exact per-window decision on every row that comes back.
        floor_out = min(
            discounted_units(_atomic_rate(r.rate_out, decimals, model.model, w, "rate_out"), pct)
            for w, r in model.slas.items()
        )
        # A window metering no input side accepts a bid paying nothing for it, so
        # one such window opens the input filter for the whole model: an unpriced
        # side is `_atomic_rate` 0, which is the loosest floor there is, and 0
        # goes on the wire as no filter at all.
        floor_in = min(
            discounted_units(_atomic_rate(r.rate_in, decimals, model.model, w, "rate_in"), pct)
            for w, r in model.slas.items()
        )

        # The book is addressed by model **id**, never by name: the order carries
        # a uint32 and the daemon's config carries a string, and the catalog is
        # the only thing that maps one to the other.
        resolver = await self._ensure_models()
        jobs = await self._node.list_open_jobs(
            resolver.model_id(model.model),
            free=free,
            min_rate_out=floor_out,
            min_rate_in=floor_in or None,
        )
        for job in jobs:
            if self._inflight >= self._slots() or throttle.claimable() <= 0 or breaker.tripped():
                return  # no free slot; try again next sweep
            # A job this daemon has already failed to resolve is skipped without
            # touching the network. First, because it is the only free check in
            # this loop; and mainly because the fetch below is inline, so a job
            # that cannot be fetched delays every bid behind it in the page.
            if self._is_deferred(job):
                continue
            # A job designated to another provider is sealed to that provider's
            # key and unclaimable by anyone else — skip it without burning the
            # claim round trip. **An open order's `designated` is 0, not null**,
            # so a truth test is the guard: `designated is not None` would treat
            # every open job on the book as somebody else's and claim nothing.
            designated = getattr(job, "designated", 0) or 0
            if designated and designated != self.provider:
                continue
            # The price decision reads the cleartext order terms (model, units,
            # rates, SLA) and, with `pricing:` configured, this backend's own
            # load — never the payload. Nothing here decrypts anything to price
            # a job. It runs first because it costs nothing.
            if not self.profitable(job, model, decimals=decimals, floor_pct=pct):
                continue
            # Then, and still before the claim, the bytes. A CID is a locator
            # anyone may pin anything under, so bytes that do not re-derive this
            # job's own id are refused here — where refusing is free — rather
            # than after a claim has spent the client's order on a payload that
            # can never be opened. The cost of the trade is one gateway fetch on
            # a race we may lose.
            container = await self._verified_container(job)
            if container is None:
                continue
            # Then the size of what arrived, against the size the order paid for.
            # `units_in` is the client's own count and the chain bills the input
            # leg at whatever it says — on an embedding bid, where the output leg
            # settles at zero, it is the *whole* bill — so a job declaring one
            # unit for a megabyte is work this daemon would do for nothing. And
            # there is no clamp to reach for the way `units_out` has one: input
            # cannot be delivered short, so declining is the entire remedy.
            #
            # Refused HERE, where declining is free: nothing is claimed, no gas is
            # spent, no penalty is taken, and the bid stays on the book for a
            # provider whose floor is looser. The measurement is the container's
            # own length, so it costs no key and opens no box.
            # `modality` is the sweep's own, from the model being polled for:
            # `job.modality` is still the placeholder here and would read a media
            # bid as text.
            short = input_shortfall(
                job, sealed_plaintext_bytes(len(container)),
                bytes_per_unit=self._config.provider.max_input_bytes_per_unit,
                modality=modality,
            )
            if short:
                # Permanent: the bytes are content-addressed and `units_in` is a
                # signed order term, so no later sweep can answer differently —
                # and without the shelf this job is re-fetched from the gateway
                # every sweep until it expires.
                self._defer(job, permanent=True)
                log.info("not claiming %s: its declared %s input units do not cover the payload "
                         "it carries (%s bytes past what they buy at %s bytes/unit)",
                         job.job_id, job.units_in, short,
                         self._config.provider.max_input_bytes_per_unit)
                continue
            if not await self._claim(job):
                continue
            self._metrics.on_claim()
            log.info("claimed", extra={"job_id": job.job_id, "model": model.model})
            await self._ingest(job, model, container)

    async def _verified_container(self, job) -> bytes | None:
        """The task bytes this job names, or ``None`` — checked **before** claiming.

        ``None`` is not a failure of ours: nothing has been claimed, nothing has
        been spent, and the job stays on the book for a provider whose fetch
        answers differently. So it is logged and never reported — there is no
        claim to give back, and reporting a job we do not hold would be a lie
        the registry would refuse anyway.
        """
        try:
            raw = await self._blobs.fetch_task(job)
        except BlobError as exc:
            # Availability, so far as this daemon can tell: the name may still be
            # propagating. Backed off rather than written off — see ``_defer``.
            self._defer(job, permanent=False)
            log.info("not claiming %s: %s", job.job_id, exc)
            return None
        try:
            verify_container(job.owner or "", job.job_id, raw)
        except ContainerError as exc:
            # Substituted ciphertext, a wrap lifted from another order, or bytes
            # that are not a container at all. Never claimed, whatever CID named
            # them — and no key is touched, because nothing here can decrypt.
            # Permanent for this job: the name is content-addressed, so the bytes
            # under it are the bytes under it, and re-deriving the same refusal
            # every sweep is the whole cost of the attack this closes.
            self._defer(job, permanent=True)
            log.warning("not claiming %s: the bytes under %s do not name it (%s)",
                        job.job_id, job.task_cid, exc.fault)
            return None
        except (ValueError, AttributeError, TypeError) as exc:
            # The job's own fields are unusable, so no source and no amount of
            # patience can change the answer.
            self._defer(job, permanent=True)
            log.warning("not claiming %s: unusable job fields for the commitment check (%s)",
                        job.job_id, exc)
            return None
        self._deferred.pop(str(job.job_id), None)
        return raw

    def _is_deferred(self, job) -> bool:
        """Has this job been shelved, and is its wait still running?

        A finished wait is answered without dropping the row, deliberately: the
        miss count lives there, and forgetting it on the sweep that acts on it
        would reset the backoff to its first step every time — a doubling that
        never doubles. The row is cleared by the fetch that finally succeeds, and
        otherwise by ``MAX_DEFERRED_JOBS``.
        """
        entry = self._deferred.get(str(job.job_id))
        return entry is not None and self._clock() < entry[0]

    def _defer(self, job, *, permanent: bool) -> None:
        """Shelve a job this daemon could not resolve.

        ``permanent`` means *no later fetch can answer differently* — a
        commitment the bytes do not satisfy, or a job whose own fields cannot be
        read. Those wait out the order itself: an order past ``expires_at`` can
        never be claimed by anyone, which makes the expiry a self-pruning deadline
        rather than a timer of ours. A merely unresolvable name doubles its wait
        instead, because the first miss really can be a pin still propagating.
        """
        key = str(job.job_id)
        now = self._clock()
        if permanent:
            # No expiry on the row is not a licence to retry forever: fall back to
            # the longest ordinary wait.
            deadline = float(getattr(job, "expires_at", None) or now + UNRESOLVED_RETRY_MAX_S)
            self._deferred[key] = (deadline, 0)
        else:
            misses = self._deferred.get(key, (0.0, 0))[1] + 1
            wait = min(UNRESOLVED_RETRY_S * 2 ** (misses - 1), UNRESOLVED_RETRY_MAX_S)
            self._deferred[key] = (now + wait, misses)
        if len(self._deferred) > MAX_DEFERRED_JOBS:
            # Drop the entries closest to retiring: they are the ones a re-fetch
            # costs least, and the ones a sweep would have retried soonest anyway.
            for stale, _ in sorted(self._deferred.items(), key=lambda kv: kv[1][0])[
                : len(self._deferred) - MAX_DEFERRED_JOBS
            ]:
                del self._deferred[stale]

    async def _claim(self, job) -> bool:
        """Simulate, then push the signed claim. ``False`` means *skip*, never *fail*.

        The simulation is the advisory gate: it reads chain state at ``latest``
        rather than the index, so it is meaningful in exactly the window a
        daemon is deciding in. It is advisory because the answer can be stale by
        the time the op lands — the authority is the node's own simulate of the
        **signed** op, which is why a refusal here is ordinary. Losing a race
        and finding a wallet drained both arrive as a refusal, and neither is an
        error: the job was never ours.
        """
        try:
            sim = await self._node.simulate_claim(job.job_id, self.ops.address)
        except httpx.HTTPError as exc:
            log.warning("claim simulation unavailable for %s (%s); not claiming",
                        job.job_id, type(exc).__name__)
            return False
        if not sim.ok:
            log.info("claim would not land for %s (%s)", job.job_id, sim.reason)
            return False
        issued_at = int(self._clock())
        ctx = await self._node.chain_context()
        signature = self.ops.sign_claim(job.job_id, issued_at, ctx)
        try:
            answer = await self._push("claim", {"job_id": job.job_id, "issued_at": issued_at},
                                      signature)
        except OpRefused as exc:
            log.info("lost race for job %s (%s)", job.job_id, exc.reason)
            return False
        except OpRejected as exc:
            # The registries did not recognise the signer: an unregistered
            # wallet, or a key that is not the one on record. Not this job's
            # fault and not retryable by re-sending, so the sweep moves on.
            log.warning("claim rejected for %s: %s", job.job_id, exc)
            return False
        # A claim that reverted is one whose escrow pull refused: the client's
        # permit no longer covers the charge, because the allowance was revoked
        # or the wallet drained inside the window the simulate could not see. The
        # pull is the last thing `claim` does, so the state transition rolled
        # back with it — the job is still Open, still anyone's, and running it
        # would be work delivered into an escrow that was never funded.
        if not self._landed(answer, "claim", job.job_id):
            return False
        # What the daemon knows about the row it just claimed. Not re-read: the
        # ops door answers a receipt, not a job, and the three fields that moved
        # are the three we just caused.
        job.state = "Claimed"
        job.provider = self.provider
        job.claimed_at = issued_at
        return True

    async def _ingest(self, claimed, model, container: bytes | None = None, *,
                      resume: str | None = None) -> None:
        """Run one job this provider owns: stamp modality, fetch and verify the
        task bytes, open the envelope, spawn the backend run. Shared by the
        claim path and boot recovery — a recovered job re-enters exactly as a
        freshly claimed one, including every refusal path.

        ``container`` is the bytes the claim path already fetched and verified;
        boot recovery has none and fetches here. Either way ``_open_task``
        verifies again, at the point of use.

        ``resume`` is the backend handle a previous life recorded for this job;
        the run skips the submit and polls it.
        """
        # Modality is not on the wire: stamp the established answer onto the
        # job before anything downstream (unit planning, the confidential
        # driver's text-only guard) reads it. Re-checked against the CLAIMED
        # job's model — ``_poll_model``'s pre-claim gate read the config's model
        # name, and only this value is what actually gets metered.
        claimed_modality = self._modality(claimed.model)
        if claimed_modality is None:
            await self._abandon_claim(claimed, "modality_unknown",
                                      f"no modality established for {claimed.model}")
            return
        claimed.modality = claimed_modality
        # Content-addressed delivery: fetch the container the order named from the
        # gateway, verify it against the job's own commitment, recover the DEK and
        # open the envelope.
        if container is None:
            try:
                container = await self._blobs.fetch_task(claimed)
            except BlobError as exc:
                await self._abandon_claim(claimed, "task_unresolved", str(exc))
                return
        try:
            job_input, result_key, custom_id = await self._open_task(claimed, container)
        except PayloadError as exc:
            await self._abandon_claim(claimed, exc.reason, str(exc))
            return
        self._spawn(claimed, model, job_input, result_key, custom_id, resume=resume)

    async def _recover_claimed(self) -> None:
        """Finish or release the jobs this provider claimed in a previous life.

        A claim spent the client's order, so stranding it until the SLA reclaim
        punishes the client for our crash. What can still run, runs (capacity is
        deliberately not consulted: prior commitments outrank new claims); what
        cannot is failed back now, for an immediate refund.

        The network's ``allowed_models`` gate is deliberately not applied: it
        governs what this daemon may OFFER, and these jobs were already sold. A
        model revoked while the daemon was down still re-runs — the settle is the
        coordinator's call to refuse, and a refused settle is reported like any
        other, so nothing is stranded either way."""
        if self.provider is None:
            # A session that carries no provider id can own no claims: nothing to
            # recover, and retrying would only warn every sweep for the whole boot.
            return
        try:
            mine = await self._node.list_claimed_jobs(self.provider)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Transport fault or a listing this build cannot read. Neither says
            # anything about the jobs themselves, so recovery is retried rather
            # than skipped for the boot — and it never propagates: a sweep that
            # raised here would abandon the claim path for the sweep too.
            self._recovered = False
            log.warning("claimed-job recovery failed (%s); will retry next sweep", type(exc).__name__)
            return
        # What the previous life had in flight. A row whose job is no longer
        # ours is stale — settled, failed or reclaimed while we were down.
        rows = {row.job_id: row for row in self._store.all()}
        mine_ids = {job.job_id for job in mine}
        for job_id in sorted(set(rows) - mine_ids):
            log.info("forgot a handle for a job no longer claimed", extra={"job_id": job_id})
            self._store.delete(job_id)
        for job in mine:
            # Already running here. A listing retried after a failed one sees the
            # jobs the intervening sweep already claimed and spawned, and
            # re-ingesting one would run it a second time against the same claim.
            if job.job_id in self._running:
                continue
            # A window that closed while the daemon was down cannot be served: the
            # backend would run to completion, the SLA guard would drop the result
            # unreported, and the job would sit claimed until someone reclaims it.
            # Fail it back before spending anything, so the client is refunded now.
            # (A row another party already reclaimed answers 409 here; the report is
            # best-effort and swallows it — the client is refunded either way.)
            if not self._within_sla(job):
                await self._abandon_claim(job, "claim_expired",
                                          f"claimed at {job.claimed_at} with sla {job.sla}; "
                                          "window closed before recovery")
                continue
            model = next((m for m in self._config.models if m.model == job.model), None)
            if model is None:
                await self._abandon_claim(job, "unservable",
                                          f"model {job.model} is not served by this daemon")
                continue
            row = rows.get(job.job_id)
            resume = row.handle if row is not None and row.model == job.model else None
            log.info("recovered" if resume is None else "resumed",
                     extra={"job_id": job.job_id, "model": job.model})
            await self._ingest(job, model, resume=resume)

    async def _open_task(self, job, container: bytes) -> tuple[dict, str, str | None]:
        """Verify the container, recover the DEK, decrypt, open the envelope.

        The order is the point. Nothing is decrypted, and no key is touched,
        until ``keccak256(owner ‖ c)`` over the fetched bytes reproduces the job's
        own id: a gateway that substituted the ciphertext derives a different id,
        and a ``seed_wrap`` lifted from another order derives a different id too.
        The check is re-run here rather than inherited from the fetch, because
        this is the point where the bytes are used.
        """
        try:
            seed_wrap, ciphertext = verify_container(job.owner or "", job.job_id, container)
        except ContainerError as exc:
            raise PayloadError(exc.fault, str(exc)) from exc
        except (ValueError, AttributeError, TypeError) as exc:
            raise PayloadError("bad_container", str(exc)) from exc

        dek = await self._recover_dek(job, seed_wrap, ciphertext)
        try:
            plaintext = open_dek(ciphertext, dek)
        except Exception as exc:  # noqa: BLE001 — any crypto failure is one refusal
            raise PayloadError("undecryptable", str(exc)) from exc
        return self._open_envelope(job, plaintext)

    async def _recover_dek(self, job, seed_wrap: bytes, ciphertext: bytes) -> bytes:
        """The DEK this container's wrap seeds — from our own box key, or the escrow.

        Which one is a fact about the bid, not about the payload: a **designated**
        bid's wrap is sealed to this daemon's published box key and opens
        in-process, while an **open** bid's is sealed to the coordinator's
        attested escrow key and is released only against a claim on record. No
        key ever arrives with a claim response.

        **The wrap seals a seed, and the DEK is derived from it under the job's
        owner.** That derivation is what makes a lifted wrap worthless: a
        container is public, so an attacker can weld somebody else's wrap into a
        fresh commitment under their own address and every field of the
        resulting order is honest — nothing checkable over public data refuses
        it. What refuses it is the ``info`` string, which carries the owner this
        job was posted by. On the designated path there is no coordinator in the
        loop at all, so this is the only thing standing between the attack and
        the victim's plaintext.
        """
        designated = getattr(job, "designated", 0) or 0
        if designated:   # 0 is the open-order sentinel; the book, not a pin
            if designated != self.provider:
                # Pre-claim filtering should have skipped this; if it did not, the
                # wrap is sealed to somebody else and we must not spend a cipher on it.
                raise PayloadError("not_our_bid",
                                   f"job is designated to provider {designated}, not {self.provider}")
            if self._cipher is None:
                raise PayloadError("no_cipher", "no box cipher configured")
            try:
                seed = self._cipher.decrypt(seed_wrap)
            except Exception as exc:  # noqa: BLE001 — a wrap we cannot open is one refusal
                raise PayloadError("unseal_failed", str(exc)) from exc
            try:
                # The owner comes off the JOB — the row the chain minted this id
                # from — and never off a request field, which is the whole of the
                # binding. A wrap that opens to something that is not a 32-byte
                # seed yielded no usable key material, which is the same refusal.
                #
                # And the seed stops here. It is never returned, logged or put on a
                # wire, which is the same rule that keeps `/release` answering with
                # a derived DEK and never with the seed it unsealed (D5). A
                # container is public, so its wrap can be lifted verbatim into an
                # attacker's own order and every field of that order is honest —
                # nothing checkable over public data refuses it. Handing back the
                # seed would let the lifter compute HKDF(seed, victim_owner)
                # themselves, and the derivation, which is the only thing that
                # refuses the attack at all, would stop being a defence.
                return derive_dek(seed, job.owner or "")
            except ValueError as exc:
                raise PayloadError("unseal_failed", str(exc)) from exc

        if self._escrow is None:
            # Refused under the escrow's OWN code, and the claim goes straight
            # back, so the client is refunded now. There is deliberately no
            # other source of a key to fall back to.
            raise PayloadError(
                "escrow_unavailable",
                "this bid is open, so its DEK is held by the coordinator's escrow, and "
                "no release client is configured",
            )
        try:
            # What comes back is the DEK **already derived** — the coordinator
            # derived it against the owner it read from chain. Deriving again
            # here would produce a key of a key and decrypt nothing.
            #
            # It is a DEK and not the seed on purpose, and the asymmetry with the
            # designated path above is load-bearing (D5). One function returning
            # two kinds of thing looks like an obvious cleanup, and it is not: a
            # lifted wrap cannot be refused, because every field of the order
            # welded around it is honest, so an escrow that answered with the seed
            # would hand the lifter exactly the material to compute
            # HKDF(seed, victim_owner) for any owner they choose. Answering with a
            # key already derived under the owner the chain reports is what makes
            # the answer useless to anyone but this job's claimant.
            return await self._escrow.release(job, seed_wrap, ciphertext_hash(ciphertext))
        except ReleaseRefused as exc:
            # The escrow's own code is the reason, verbatim: it is a stable
            # token, it is what an operator pages on (escrow_key_lost is not
            # wrong_wallet), and a code this build does not know is reported as
            # itself rather than collapsed into a neighbour it did not happen.
            raise PayloadError(exc.code, str(exc)) from exc
        except httpx.HTTPError as exc:
            # The escrow did not answer at all. Not a verdict on this job, but
            # this loop has no way back to a claimed job before the next boot, so
            # sitting on it would idle the client's order to SLA expiry. Hand it
            # back now, under a code that is plainly not one of the escrow's.
            raise PayloadError("escrow_unreachable", type(exc).__name__) from exc

    def _open_envelope(self, job, plaintext: bytes) -> tuple[dict, str]:
        """Open the client's envelope inside the decrypted container.

        Returns ``(model input, result key)``; the key is always present, because
        an envelope that names none is refused below — there is no unsealed result
        path. The envelope names its own owner, and that name must be the job's
        owner — otherwise these are somebody else's terms replayed under this
        job's payment (spec D8), and the job is refused rather than run.
        """
        try:
            env = json.loads(plaintext)
        except ValueError as exc:
            raise PayloadError("malformed_envelope", str(exc)) from exc
        if not isinstance(env, dict) or env.get("v") != ENVELOPE_VERSION:
            raise PayloadError("malformed_envelope", f"version {env.get('v')!r}"
                               if isinstance(env, dict) else "not an object")
        if str(env.get("owner") or "").lower() != str(job.owner or "").lower():
            raise PayloadError("owner_mismatch",
                               f"envelope is addressed to {env.get('owner')!r}, job owner is {job.owner!r}")
        if "input" not in env:
            raise PayloadError("malformed_envelope", "envelope carries no input")
        # The same floor `_poll_model` applies before claiming, over the same
        # number: a container carries exactly `len(container) - 121` bytes of
        # envelope, so a bid that cleared the gate clears this too, by
        # construction. That is deliberate — a fail refunds the client in full and
        # leaves the relayer's claim gas unrecovered, so under-declaring must
        # never be a way to make somebody else pay for one.
        #
        # What it catches is the path with no gate in front of it: a claim this
        # daemon inherited at boot from a previous life, which may have run a
        # build that never checked. Re-checked at the point of use for the same
        # reason `_open_task` re-verifies the commitment the fetch already
        # verified.
        short = input_shortfall(job, len(plaintext),
                                bytes_per_unit=self._config.provider.max_input_bytes_per_unit,
                                modality=job.modality)
        if short:
            raise PayloadError(
                "units_in_short",
                f"the envelope runs {short} bytes past what its declared {job.units_in} input "
                f"units buy at {self._config.provider.max_input_bytes_per_unit} bytes/unit",
            )
        result_key = env.get("result_key")
        if not result_key or not isinstance(result_key, str):
            # There is no in-the-clear result path. Settling unsealed bytes would
            # hand the answer to the coordinator and anyone reading the blob
            # surface, which is the one thing end-to-end encryption is for — so an
            # envelope that names no key is malformed, not permission to skip it.
            raise PayloadError("malformed_envelope", "envelope names no result_key")
        # …and a key that is present but unusable is refused HERE, not at sealing
        # time. `_seal_result` hands the string straight to Curve25519's hex
        # decoder, so anything other than 32 bytes of bare hex raises after the
        # backend has already produced the answer — capacity spent on a job that
        # could never be settled. The shape is exactly what nacl accepts: 64 hex
        # digits, no `0x` (the decoder does not strip a prefix).
        if not _RESULT_KEY_RE.fullmatch(result_key):
            raise PayloadError("malformed_envelope",
                               "result_key is not a 32-byte Curve25519 public key in bare hex")
        # `custom_id` is the caller's own label for this line. It is optional and opaque —
        # never parsed, never forwarded to a backend, only stamped back onto the result so a
        # caller can correlate a line it did not submit in this process.
        custom_id = env.get("custom_id")
        return env["input"], result_key, (custom_id if isinstance(custom_id, str) else None)

    async def _abandon_claim(self, job, reason: str, detail: str) -> None:
        """Give a claimed job back: it cannot be run, so report it and let the
        client's escrow refund now instead of idling to SLA expiry.

        Immediately, and that is the point of doing it here rather than at the
        end of the sweep: inside ``FAIL_GRACE`` of the claim a fail is
        penalty-free, and every condition that reaches this method is known the
        moment it is discovered. ``escrow_key_lost`` is the case the window was
        written for — the key is gone, no amount of waiting brings it back, and
        the provider that says so at once is not punished for it.
        """
        log.warning("refusing claimed job (%s): %s", reason, detail,
                    extra={"job_id": job.job_id, "penalty_free": self._within_fail_grace(job)})
        self._failed(job, reason, detail)
        await self._report_fail(job, reason)

    def _failed(self, job, reason: str, detail: str = "") -> None:
        """Count a job this daemon gave back, and report it to Sentry under its model."""
        self._metrics.on_fail(reason)
        report_job_failed(job.model, reason, job.job_id, detail)

    def _within_fail_grace(self, job) -> bool:
        """Is a fail for this job still penalty-free?

        Mirrors ``JobRegistry.FAIL_GRACE``, which the daemon cannot read: it
        holds no RPC and ``GET /evm/chain`` does not serve it. Diagnostic only —
        nothing waits on it and nothing skips a report because of it, since a
        late fail still refunds the client.
        """
        claimed_at = getattr(job, "claimed_at", None)
        if claimed_at is None:
            return True
        return self._clock() < claimed_at + self._config.provider.fail_grace_s

    @staticmethod
    def _settle_failure(exc: httpx.HTTPError, result_size: int) -> tuple[str, str]:
        """The metric label and the operator's detail for a settle that did not land.

        The label names the surface that actually failed, because that is what an
        operator pages on. A settle needs a session, so a failed handshake arrives
        here without the chain surface ever having seen the settle; a transport
        failure means nothing answered at all; and 413 is the one condition an
        operator can act on — the sealed result, media frames and all, is past
        the node's blob ceiling, so the model's ``units_out`` cap admits jobs
        whose output cannot be delivered.
        """
        if not isinstance(exc, httpx.HTTPStatusError):
            return ("settle_transport_error",
                    f"the settle could not be delivered ({type(exc).__name__})")
        status = exc.response.status_code
        if exc.request.url.path.startswith("/auth/"):
            return (f"session_http_{status}",
                    f"the session handshake was refused with HTTP {status}, so the settle was "
                    "never sent (the failure report needs that session too)")
        if status == 413:
            return ("settle_result_too_large",
                    f"the sealed result is {result_size} bytes, past the node's ceiling")
        return f"settle_http_{status}", f"the settle was refused with HTTP {status}"

    async def _report_fail(self, job, reason: str) -> None:
        """Report a claimed job the daemon will not settle, so the client's escrow
        refunds now. Best-effort: if the report itself fails, the refund degrades to
        the SLA-expiry reclaim.

        **The reason does not travel.** A ``Fail`` op carries the job id and
        ``issued_at`` and nothing else — the registry refunds the client either
        way and has no use for a provider's account of why, which it could not
        verify. So ``reason`` is metered and logged here, where an operator can
        act on it, and the chain records only that this claimant gave the job
        back.
        """
        try:
            issued_at = int(self._clock())
            ctx = await self._node.chain_context()
            answer = await self._push("fail", {"job_id": job.job_id, "issued_at": issued_at},
                                      self.ops.sign_fail(job.job_id, issued_at, ctx))
        except Exception:  # noqa: BLE001 — refund degrades to SLA-expiry reclaim
            log.warning("could not report failure", extra={"job_id": job.job_id, "reason": reason})
            return
        # A fail that mined reverted refunded nobody, so the client's money is
        # still held and this is the same outcome as the report never landing —
        # said out loud, because the alternative is a daemon that logs a refund
        # it did not cause.
        if not self._landed(answer, "fail", job.job_id):
            log.warning("failure report did not land; the refund falls back to the SLA-expiry "
                        "reclaim", extra={"job_id": job.job_id, "reason": reason})

    # `custom_id` takes no default, deliberately. It had one, and the one caller
    # below unpacked the label out of the envelope and then never passed it — the
    # default filled in `None`, every downstream signature accepted it, and the
    # label vanished with nothing red. A required argument is the cheapest thing
    # that would have caught it.
    def _spawn(self, job, model, job_input, result_key, custom_id: str | None,
               resume: str | None = None) -> None:
        self._inflight += 1
        self._throttles[model.model].hold()
        self._metrics.set_capacity_free(max(0, self._slots() - self._inflight))
        task = asyncio.create_task(self._run_job(job, model, job_input, result_key, custom_id, resume))
        self._tasks.add(task)
        self._running[job.job_id] = task
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda _t: self._running.pop(job.job_id, None))

    # -- job execution -------------------------------------------------------

    async def _run_job(self, job, model, job_input, result_key: str,
                       custom_id: str | None, resume: str | None = None) -> None:
        driver = self._driver_for(model.model, job.sla)
        started = self._clock()
        suspended = False
        try:
            # units_out is a cap for every modality: clamp the media cost driver
            # (num_images / duration_secs) to what the client paid for before rendering.
            # A minimum unit over the cap raises here (BackendError → abandoned), which
            # stops a client under-declaring to underpay. The plan is a ceiling, not a
            # bill: what settles is counted from the frames actually sealed, so a render
            # that comes back short is charged short.
            if job.modality in ("image", "video"):
                # param_caps additionally bounds the compute drivers (steps/fps)
                # the pixel unit does not price — the operator's margin defense.
                # `accept` is the operator's reference policy: a media type this
                # backend does not take is a capability gap, and a job carrying one
                # is handed back rather than sent to a backend that will refuse it.
                backend = model.backend_for(job.sla)
                try:
                    job_input, _ = plan_media_units(
                        job, job_input, caps=backend.param_caps, **backend.media_policy)
                except BackendError:
                    raise
                except Exception as exc:  # noqa: BLE001 — unpriceable input is one refusal
                    # A field of the wrong type, from an order no SDK built. Left
                    # to the crash handler below it would report no `fail`, and
                    # the claim would sit until the SLA reclaimed it at this
                    # provider's expense. The error's kind only: its text quotes
                    # the input, and this reason goes on the network.
                    raise MediaInputRefused(
                        f"the request could not be priced ({type(exc).__name__})"
                    ) from exc

            self._in_backend.add(job.job_id)
            normalized = await self._run_attempts(job, model, driver, job_input, resume=resume)
            # The backend phase is over: from here a stop drains rather than
            # suspends. The row stays, so a crash during the settle re-polls the
            # finished response at the next boot instead of submitting again.
            self._in_backend.discard(job.job_id)
            self._breakers[model.model].record_success()
            self._metrics.on_backend_latency(model.model, self._clock() - started)

            if not self._within_sla(job):
                log.warning("abandoned (SLA)", extra={"job_id": job.job_id, "model": model.model})
                self._failed(job, "sla_abandon")
                return

            result_bytes, completion_tokens = await self._build_result(
                job, normalized, result_key, custom_id)
            if completion_tokens is not None and job.units_out is not None:
                # units_out is the charged cap; a count above it must never settle above
                # what the client paid for. Both branches of
                # _build_result have already produced a valid non-negative count — text
                # from the backend's own counter, media from the frames it sealed — so
                # this only ever clamps a too-large one.
                completion_tokens = min(completion_tokens, job.units_out)
            completion_tok = int(completion_tokens or 0)
            answer = await self._settle(job, result_bytes, completion_tok)
            if answer is None:
                return
            result_cid = answer.result_cid
            self._metrics.on_settle(self._clock() - started)
            if not result_cid:
                # Not fatal — the job settled — but the answer is meant to name
                # what was delivered, and nothing here can reconstruct it.
                log.warning("settle answered without a result_cid", extra={"job_id": job.job_id})
            log.info("settled", extra={"job_id": job.job_id, "model": model.model,
                                       "result_cid": result_cid})
        except BackendError as exc:
            # The backend phase is over here too: a stop from now on must drain the report, not suspend it.
            self._in_backend.discard(job.job_id)
            log.warning("abandoned (backend): %s", exc, extra={"job_id": job.job_id})
            # A stable token per outcome, not the message: the reason is a
            # metric label.
            if isinstance(exc, DeadlineExceeded):
                reason = "deadline_wait"
            elif isinstance(exc, BackendExhausted):
                reason = "backend_exhausted"
            elif isinstance(exc, BackendGone):
                reason = "backend_gone"
            elif isinstance(exc, MediaInputRefused):
                # Its own label because it is its own thing: not a backend that
                # misbehaved but a reference the client declared smaller than it
                # sent, or of a type this backend does not take. An operator
                # seeing these on a dashboard is looking at their callers, not at
                # their hardware — and `_backend_at_fault` already keeps it off
                # the circuit breaker.
                reason = "media_input_refused"
            else:
                reason = "backend_error"
            self._failed(job, reason, str(exc))
            if _backend_at_fault(exc) and self._breakers[model.model].record_failure():
                await self._withdraw_tripped(model.model)
            await self._report_fail(job, str(exc))
        except asyncio.CancelledError:
            # `stop()` suspended this job: its handle stays on record so the
            # next boot resumes it. The slot is still freed below.
            suspended = True
            raise
        except Exception:
            log.exception("job %s crashed", job.job_id)
            self._metrics.on_fail("internal_error")
        finally:
            self._in_backend.discard(job.job_id)
            if not suspended:
                self._store.delete(job.job_id)
            self._inflight -= 1
            self._throttles[model.model].drop()
            self._metrics.set_capacity_free(max(0, self._slots() - self._inflight))

    async def _settle(self, job, result_bytes: bytes, completion_tok: int):
        """Deliver a finished job, retrying until the SLA closes.

        The work is done and paid for at the backend, and the node is the only
        route to the chain, so a failure of the node to relay is waited out, not
        answered by giving the job back: a ``429``, a ``5xx``, a transport fault,
        a settle that mined reverted and a ``StaleOp`` are each retried under a
        fresh ``issued_at`` — the time is inside the signature, so every attempt
        is signed again. An upload the node no longer holds is uploaded again.

        Returns the node's answer once a settle lands, and ``None`` when it
        cannot: the chain's verdict that the job is no longer this provider's
        (nothing is reported — the escrow is already resolved), a refusal no
        retry can change, or the deadline, after which the job is handed back so
        the client is refunded without waiting for a reclaim.

        A result at or under INLINE_MAX_BYTES rides with the op as base64; a
        bigger one is uploaded first and referenced by its cid. Either way the
        node pins the bytes, mints their name and puts it in `submitAndSettle`:
        the daemon never learns a CID before the node answers and never signs
        one.
        """
        deadline = job.claimed_at + sla_seconds(job.sla) if job.claimed_at is not None else None
        result_field: dict | None = None
        attempt = 0
        while True:
            try:
                ctx = await self._node.chain_context()
                if result_field is None:
                    if len(result_bytes) <= INLINE_MAX_BYTES:
                        result_field = {"result": base64.b64encode(result_bytes).decode()}
                    else:
                        result_field = {"result_cid": await self._node.upload_file("result", result_bytes)}
                # Stamped after the upload, not before it: `issued_at` is the op's
                # freshness, and a large result on a slow uplink would otherwise
                # arrive already stale.
                issued_at = int(self._clock())
                answer = await self._push(
                    "settle",
                    {"job_id": job.job_id, "completion_tok": completion_tok,
                     "issued_at": issued_at, **result_field},
                    self.ops.sign_settle(job.job_id, completion_tok, issued_at, ctx),
                )
                if self._landed(answer, "settle", job.job_id):
                    return answer
                # Mined and reverted: the job is still Claimed. The next attempt's
                # pre-relay simulate says why, as a refusal.
                reason, detail, final = "settle_reverted", "the settle mined reverted", False
            except OpRefused as exc:
                if exc.reason != "StaleOp":
                    # The chain's verdict, and it is not reported: this daemon no
                    # longer holds the job — it settled already, or the deadline
                    # passed and it was reclaimed — so the escrow is resolved and
                    # a fail would be refused in turn.
                    log.warning("settle refused (%s)", exc.reason, extra={"job_id": job.job_id})
                    self._failed(job, f"settle_{exc.reason}")
                    return None
                reason, detail, final = "settle_StaleOp", "the settle was refused as stale", False
            except OpRejected as exc:
                # The registries did not recognise the signer.
                reason, detail, final = "settle_rejected", str(exc), True
            except UploadInvalid as exc:
                # The upload answered 2xx with no name in it.
                reason, detail, final = "settle_upload_invalid", str(exc), True
            except httpx.HTTPError as exc:
                reason, detail = self._settle_failure(exc, len(result_bytes))
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                if status == 400 and _error_code(exc.response) == "unknown_result":
                    # The upload outlived its window before the settle named it.
                    result_field, final = None, False
                else:
                    final = status is not None and status != 429 and status < 500

            delay = min(SETTLE_RETRY_S * 2 ** attempt, SETTLE_RETRY_MAX_S)
            if final or deadline is None or self._clock() + delay >= deadline:
                # The work is done and cannot be delivered. Handed back so the
                # client's escrow refunds now instead of idling to a reclaim; the
                # report is best-effort and may need the surface that just failed.
                log.warning("job not settled: %s", detail, extra={"job_id": job.job_id})
                self._failed(job, reason, detail)
                await self._report_fail(job, reason)
                return None
            log.warning("settle not delivered (%s); retrying in %.0fs", detail, delay,
                        extra={"job_id": job.job_id})
            attempt += 1
            await self._sleep(delay)

    async def _withdraw_tripped(self, name: str) -> None:
        """Take a tripped model off the book now, from the job that tripped it.

        Not deferred to the sweep: every second the ask stands, the coordinator
        keeps leasing this daemon work it is about to fail. A push that does
        not land is not an error here — the book is unchanged, and the next
        sweep's `sync_asks` retries the same diff.
        """
        breaker = self._breakers[name]
        log.warning("backend tripped after %d consecutive faults; asks withdrawn for %.0fs",
                    breaker.streak, breaker.open_until - self._clock(), extra={"model": name})
        try:
            await self.sync_asks()
        except Exception:
            log.exception("could not withdraw the tripped model's asks now; the next sweep retries")

    async def _run_attempts(self, job, model, driver, job_input, *,
                            resume: str | None = None) -> Normalized:
        """Run the backend for one job, retrying inside the SLA budget.

        Each attempt first waits for the model's throttle to admit it, then runs
        under a timeout no longer than the time left before the deadline. A
        retryable failure is followed by a backoff — doubling from the entry's
        `retry_backoff_s`, capped at `retry_backoff_max_s`, or the backend's own
        `Retry-After` when that is longer — and another attempt, up to `retries`
        of them. Nothing here is ever scheduled past `deadline - safety_margin_s`:
        a wait or an attempt that would end there raises instead, so the job is
        failed back while the client can still be refunded now.

        A resumed job's first attempt polls the recorded handle; a retry after
        that submits afresh and records the new handle.
        """
        policy = self._policies[model.model]
        throttle = self._throttles[model.model]
        deadline = self._deadline(job)

        # Only a driver that can resume is told how: a sync driver keeps its
        # signature, and a handle it never returns is never asked for.
        def remember(handle: str) -> None:
            self._store.put(job.job_id, model.model, handle)
            log.info("backend_submitted", extra={"job_id": job.job_id, "model": model.model,
                                                 "handle": handle})

        resumable = bool(getattr(driver, "resumable", False))
        for attempt in range(policy.attempts):
            await self._admit(job, model.model, throttle, deadline, attempt)
            throttle.acquire()
            try:
                timeout_s = self._attempt_timeout(policy, deadline)
                try:
                    run_kw = ({"resume": resume if attempt == 0 else None, "on_handle": remember}
                              if resumable else {})
                    return await asyncio.wait_for(
                        driver.run(job, job_input, timeout_s=timeout_s, **run_kw),
                        timeout=max(1.0, deadline - self._clock()),
                    )
                except asyncio.TimeoutError as exc:
                    # The wall on the attempt as a whole. `timeout_s` is a read
                    # ceiling — one silence — and a streamed answer that keeps
                    # producing has no other end.
                    raise BackendError("the attempt ran past the SLA budget",
                                       retryable=True) from exc
            except BackendError as exc:
                if not exc.retryable:
                    raise
                if attempt == policy.attempts - 1:
                    raise BackendExhausted(
                        f"gave up after {policy.attempts} attempt(s): {exc}"
                    ) from exc
                delay = retry_delay(attempt, policy, exc.retry_after_s)
                if self._clock() + delay >= deadline:
                    raise DeadlineExceeded(
                        f"out of SLA budget after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                log.info("backend_retry", extra={
                    "job_id": job.job_id, "model": model.model, "attempt": attempt + 1,
                    "delay_s": round(delay, 1), "retry_after_s": exc.retry_after_s,
                    "error": str(exc),
                })
                self._metrics.on_retry(model.model)
                await self._sleep(delay)
            finally:
                throttle.release()
                self._wake(model.model)
        raise BackendExhausted("no attempt was made")  # pragma: no cover — attempts >= 1

    async def _admit(self, job, name: str, throttle, deadline: float, attempt: int) -> None:
        """Wait until the model's throttle admits one more attempt, or raise.

        Two things can hold an attempt: a spent `rate_limit` window, which clears
        on the clock, and every in-flight slot taken by a sibling job, which
        clears when one of them returns. The first is slept through, the second
        waited on (`_wake`), and neither past the deadline.
        """
        while True:
            wait = throttle.wait_s()
            if wait <= 0 and throttle.startable() > 0:
                return
            if self._clock() + wait >= deadline:
                raise DeadlineExceeded(
                    f"throttled past the deadline after {attempt} attempt(s)"
                )
            if wait > 0:
                log.info("backend_wait", extra={"job_id": job.job_id, "model": name,
                                                "wait_s": round(wait, 1)})
                await self._sleep(wait)
                continue
            log.info("backend_queued", extra={"job_id": job.job_id, "model": name,
                                              "inflight": throttle.inflight})
            try:
                await asyncio.wait_for(self._slot_freed[name].wait(),
                                       timeout=max(0.0, deadline - self._clock()))
            except asyncio.TimeoutError:
                raise DeadlineExceeded(
                    f"queued past the deadline after {attempt} attempt(s)"
                ) from None

    def _wake(self, name: str) -> None:
        """An in-flight slot freed: every job queued on this model re-checks."""
        event = self._slot_freed[name]
        event.set()
        event.clear()

    def _deadline(self, job) -> float:
        """The instant after which nothing for this job may still be running.

        `claimed_at` is the chain's own stamp on the claim; a job recovered at
        boot without one (a zero on the wire) is treated as claimed now, which is
        the most it can have left.
        """
        claimed_at = job.claimed_at if job.claimed_at is not None else self._clock()
        return claimed_at + sla_seconds(job.sla) - self._config.provider.safety_margin_s

    def _attempt_timeout(self, policy: RetryPolicy, deadline: float) -> float:
        """One attempt's read ceiling: the entry's `timeout_s`, and never more
        than the time left before the deadline — a result that lands after it
        could not be settled anyway, and the attempt is better turned over."""
        remaining = max(1.0, deadline - self._clock())
        return min(policy.timeout_s, remaining) if policy.timeout_s is not None else remaining

    @staticmethod
    def _stamped(payload: dict, job, custom_id: str | None) -> str:
        """The sealed body, carrying the stamp that says which line it answers.

        A claimant running many lines of one batch concurrently can pair a settle with
        another line's answer. Nothing outside catches it: the client's result cipher is
        derived from its wallet rather than per job, so a crossed result decrypts perfectly,
        and no signature binds a result to a job. The stamp makes the crossing visible.

        It is a correctness check on the provider's own scheduling, **not** a security
        boundary — these bytes are authored here, so a dishonest daemon stamps whatever it
        likes. `custom_id` is absent rather than null when the caller named none: the client
        reads it straight onto a result field, where a null would be a label of its own.

        A copy, never the backend's own object: `normalized.raw` is handed to this from the
        response and must not grow a key the caller did not send.
        """
        vorq: dict[str, Any] = {"job_id": job.job_id}
        if custom_id is not None:
            vorq["custom_id"] = custom_id
        return json.dumps({**payload, "vorq": vorq})

    async def _build_result(
        self, job, normalized, result_key, custom_id: str | None = None
    ) -> tuple[bytes, int | None]:
        """Produce the sealed result bytes that settle the job, and the billable count."""
        if normalized.kind == "embedding":
            # Settles at zero, and that is the correct bill rather than a missing one. An
            # embeddings backend reports `usage.prompt_tokens` and no completion count — the
            # output size is a property of the model, not of the request — so the charge is
            # `rate_in * units_in` and `_atomicCharge` multiplies the output leg by 0. The text
            # branch below fails closed on a missing count for exactly the opposite reason: there
            # it means the backend did not say what it produced.
            content = self._stamped(normalized.raw if normalized.raw is not None else {},
                                    job, custom_id)
            return self._seal_result(result_key, content), 0

        if normalized.kind == "text":
            # Bill strictly by the backend's reported output-token count. A missing or
            # non-integer count has no billable quantity; fail closed (abandon, never
            # settle) so the escrow's full-cap fallback (completionTokens ?? unitsOut)
            # can never fire on a text job.
            ct = normalized.completion_tokens
            if not isinstance(ct, int) or isinstance(ct, bool) or ct < 0:
                raise BackendError("backend returned no completion token count")
            content = self._stamped(
                normalized.raw if normalized.raw is not None else {"text": normalized.text},
                job, custom_id)
            return self._seal_result(result_key, content), normalized.completion_tokens

        # media: the frames travel inside the sealed result, exactly as text does.
        # One sealed object goes out with the settle call; the coordinator pins it
        # and records its CID. Nothing is uploaded, and no frame carries a name of
        # its own — a media result is a result.
        frames: list[dict[str, Any]] = []
        for url in normalized.media_urls or []:
            data, served_as = await self._fetch_media(url)
            frames.append(self._frame(data, normalized, served_as=served_as))
        for blob in normalized.media_blobs or []:
            frames.append(self._frame(blob, normalized))
        if not frames:
            raise BackendError("backend returned no media output")

        # The billable count is read off the frames that are going into the sealed
        # result, never off the request that asked for them: a backend that returns
        # one image of four settles one image. This is the same arithmetic the client
        # SDK runs over the same frames to display what it paid, so the settled charge
        # and the client's own figure are one number.
        if normalized.duration_secs is not None:
            # The clip's own length where its header states one. A job run with the
            # backend's "you choose" value has no priced length to fall back on.
            seconds = frames[0].pop("duration_secs", None) or int(normalized.duration_secs)
            if seconds < 1:
                raise BackendError("the delivered clip does not state its length")
            payload: dict[str, Any] = {"video": {**frames[0], "duration_secs": seconds}}
            units = _frame_pixels(frames[0]) * seconds
        else:
            for frame in frames:
                frame.pop("duration_secs", None)
            payload = {"images": frames}
            units = sum(_frame_pixels(frame) for frame in frames)
        # What settles, stated: the frames say what was delivered, and the order's
        # cap bounds what is charged. The client costs its result on this number.
        if job.units_out is not None:
            units = min(units, int(job.units_out))
        payload["units"] = units
        if normalized.seed is not None:
            payload["seed"] = normalized.seed
        return self._seal_result(result_key, self._stamped(payload, job, custom_id)), units

    def _seal_result(self, result_key: str, content: str) -> bytes:
        """Seal the result back to the key the client put in its own envelope.

        The order carries no result key — the envelope does, so only a daemon
        that actually opened the payload can address the answer. There is no
        unsealed branch: ``_open_envelope`` refuses an envelope without a key, so
        a result can never leave here readable by the coordinator.
        """
        sealed = seal_to(result_key, content.encode())
        return json.dumps(
            {"enc": "vorq-sealed-v1", "ciphertext": base64.b64encode(sealed).decode()}
        ).encode()

    def _frame(self, data: bytes, normalized: Normalized, *, served_as: str | None = None) -> dict[str, Any]:
        """One frame, base64 inline, carrying the dimensions billing reads.

        The dimensions are **what was delivered**, read off the bytes' own header:
        the frame table is a pricing convention and a model renders what it
        renders (a job priced 640x480 has come back 752x560), so a label that
        repeated the priced numbers would misdescribe the file in the client's
        hands and settle a size nobody rendered. The settled count is capped at
        the order's ``units_out`` either way. Output whose header cannot be read
        falls back on what was priced.

        The type is the backend's mapped field first, then what the bytes open
        as, then what they were served as.
        """
        found = media.delivered(data)
        kind, width, height, seconds = found if found else (None, normalized.width,
                                                            normalized.height, None)
        frame = {
            "b64": base64.b64encode(data).decode(),
            "content_type": (normalized.content_type or kind or served_as
                             or "application/octet-stream"),
            "width": width,
            "height": height,
        }
        if seconds is not None:
            frame["duration_secs"] = seconds
        return frame

    async def _fetch_media(self, url: str) -> tuple[bytes, str | None]:
        """The rendered bytes and the media type they were served as.

        A frame the backend named but cannot deliver is a backend failure like any
        other, so it is raised as one: that routes it to the report path, and the
        client's escrow refunds now instead of idling to SLA expiry with the job
        still claimed. The URL goes to the operator's log only — the reason travels
        to the network, and a media URL can carry a signed token and names the
        operator's own runtime.
        """
        if self._http is None:
            self._http = httpx.AsyncClient()
        try:
            resp = await self._http.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.warning("media fetch refused (HTTP %s): %s", exc.response.status_code, url)
            raise BackendError(
                f"backend could not serve a rendered frame (HTTP {exc.response.status_code})"
            ) from exc
        except httpx.HTTPError as exc:   # connect, read, timeout — the frame is just as gone
            log.warning("media fetch failed (%s): %s", type(exc).__name__, url)
            raise BackendError("backend could not serve a rendered frame") from exc
        served_as = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
        return resp.content, served_as

    # -- decision helpers ----------------------------------------------------

    def profitable(self, job, model, *, decimals: int, floor_pct: int | None = None) -> bool:
        """Is this bid worth claiming?

        The configured rates are the starting floor, exactly as they always
        were — and then, if the operator has asked for it, the floor is allowed
        to sit some way **under** the price this daemon publishes. That gap is
        private: it is never signed, never pushed and never in the book, because
        a floor every counterparty can read is a floor every bid converges onto.

        The chain permits the gap. ``JobRegistry.claim`` never consults the ask
        registry: it charges the rates the client signed, so a listed provider
        may take any open job at any price it is willing to accept.

        **This is the boundary, and the listing filter is not.** ``_poll_model``
        asks the node for affordable bids only, and hands the floor it asked for
        in as ``floor_pct`` so one sweep judges every row against one number. A
        node that ignored the filter, or answered against a stale floor, is
        refused right here.

        Exact integers throughout: the job's rates arrive atomic (converted from
        the wire's USD at the node boundary) and the configured USD rates are
        converted at the token's ``decimals`` before the discount applies.
        """
        rate = model.slas.get(job.sla)
        if rate is None:  # a window this provider does not serve
            return False
        if job.rate_out is None:
            return False
        pct = self._floor_pct(model) if floor_pct is None else floor_pct
        floor_out = discounted_units(
            _atomic_rate(rate.rate_out, decimals, model.model, job.sla, "rate_out"), pct)
        if job.rate_out < floor_out:
            return False
        if rate.rate_in is not None:  # model meters an input side
            floor_in = discounted_units(
                _atomic_rate(rate.rate_in, decimals, model.model, job.sla, "rate_in"), pct)
            if job.rate_in is None or job.rate_in < floor_in:
                return False
        return True

    def _floor_pct(self, model) -> int:
        """This model's whole floor discount for this sweep, in whole percent.

        A function of load alone, so it is one number per model per sweep rather
        than a decision retaken per bid — the same number the listing filter
        carries and the same one every returned row is then judged against.
        """
        if model.load is not None and model.load.source == "occupancy":
            # A backend with no metrics endpoint: the entry's own occupancy —
            # jobs held over holdable, attempts started over the day's budget —
            # is the load, the daemon's own count rather than a guess, reported
            # on the same gauge a probe would fill.
            load = self._throttles[model.model].occupancy()
            if load is not None:
                self._metrics.set_backend_load(model.model, load)
        else:
            load = self._load.load(model.model)
        return self._floor_discount(load, relaxed=self._low_load(load))

    def _floor_discount(self, load: float | None, *, relaxed: bool) -> int:
        """How far under the published ask the floor may sit, in whole percent.

        Truncated rather than rounded, so a fractional point always resolves in
        the provider's favour, and capped so the floor stays a floor. The cap is
        applied here as well as inside :func:`discounted_units` because this is
        the value the gauge reports: an unclamped sum would describe a discount
        the daemon does not actually take.
        """
        pricing = self._config.provider.pricing
        pct = int(raw_discount_pct(pricing, load))
        if relaxed:
            pct += pricing.bid_tolerance_pct
        return min(pct, MAX_FLOOR_DISCOUNT_PCT)

    def _low_load(self, load: float | None) -> bool:
        """Quiet enough that a cheap fill beats an idle GPU.

        An unknown load is never quiet: a dark probe closes this gate rather
        than opening it.
        """
        return load is not None and load * 100 < self._config.provider.pricing.low_load_pct

    def _slots(self) -> int:
        """The slots this daemon may fill: its configured capacity, or the
        network's grant when that is smaller."""
        return self._capacity if self._granted is None else min(self._capacity, self._granted)

    async def _refresh_grant(self) -> None:
        """Read the slots the network grants, once per sweep.

        The registry grants `min(requested, ceiling) × reputation / 1000` slots
        (never fewer than one) and refuses a claim past that. A daemon offering
        its configured `capacity` while the grant is smaller is leased rows it
        then skips at the simulate, and is named on client challenges it cannot
        serve — so the grant caps what the poll offers. It moves with reputation,
        which is why it is re-read every sweep; a read that fails keeps the last.
        """
        try:
            rec = await self._node.get_provider(self._coord.provider_id)
        except httpx.HTTPError as exc:
            log.warning("provider record unreadable (%s); keeping the last known grant",
                        type(exc).__name__)
            return
        self._apply_grant(rec)

    def _apply_grant(self, rec: dict) -> None:
        raw = rec.get("capacity")
        if raw is None:
            return   # a node that reports no grant: the configured capacity stands
        try:
            granted = max(1, int(raw))
        except (TypeError, ValueError):
            log.warning("provider record carries an unreadable capacity %r; ignored", raw)
            return
        if granted == self._granted:
            return
        if granted < self._capacity:
            log.warning("the network grants %d of the %d slots configured; the grant grows "
                        "with reputation", granted, self._capacity)
        else:
            log.info("the network grants %d slots", granted)
        self._granted = granted
        self._metrics.set_capacity_granted(granted)
        self._metrics.set_capacity_free(max(0, self._slots() - self._inflight))

    async def _refresh_load(self) -> None:
        """One scrape per configured backend, and the resulting floor, per sweep.

        The gauge reports exactly the number the sweep then acts on — the same
        :meth:`_floor_pct` that goes out as the listing filter.
        """
        await self._load.refresh()
        for model in self._config.models:
            self._metrics.set_floor_discount(model.model, self._floor_pct(model))

    def _within_sla(self, job) -> bool:
        if job.claimed_at is None:
            return True
        deadline = job.claimed_at + sla_seconds(job.sla)
        return self._clock() < deadline - self._config.provider.safety_margin_s
