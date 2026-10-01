"""Daemon wiring: startup -> publish asks -> claim -> execute -> settle -> drain."""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import httpx
import pytest

from vorqd._crypto import BoxCipher, seal_to
from .conftest import fake_cid, job_id_of, req_op, seal_container
from vorqd.cli import build_daemon
from vorqd.config import (
    BackendConfig,
    LoadProbeConfig,
    ModelConfig,
    ProviderConfig,
    SlaRate,
    VorqdConfig,
)
from vorqd.container import derive_dek
from vorqd.errors import ConfigError
from vorqd.pricing import LoadMonitor, NullLoadMonitor

#: What `GET /evm/chain` serves: the chain id and the four contracts every op
#: signs against. The addresses are the devnet's deterministic deployment.
CHAIN_BODY = {
    "chain_id": 31337,
    "contracts": {
        "job_registry": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
        "provider_registry": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
        "ask_registry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "usdc": "0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9",
    },
    "decimals": 6,
}

COORD = "coord.test"
BACKEND = "backend.test"
PROVIDER_ID = 7   # admin-issued; ambient from the session, never in a request body
OWNER = "0x" + "11" * 20
MODEL = "deepseek-ai/deepseek-v4-pro:fp8"
MODEL_ID = 7      # the uint32 the chain addresses the model by
SLA_SECS = 3600
CLIENT = BoxCipher.generate()   # the key the client puts in its own envelope
BOX_KEY = "aa" * 32             # the daemon's configured box key (see make_config)
DAEMON_BOX = BoxCipher(BOX_KEY).public_key


def task_envelope(payload: dict, *, owner: str = OWNER, result_key: str | None = None) -> bytes:
    return json.dumps({"v": "vorq-env-v1", "owner": owner, "result_key": result_key,
                       "input": payload}, sort_keys=True, separators=(",", ":")).encode()


class FakeStack:
    """One MockTransport standing in for the coordinator, chain, gateway and backend."""

    def __init__(self, *, task: bytes | None = None, pin: bool = True, recipient: str | None = None,
                 designated: int = PROVIDER_ID, escrow: BoxCipher | None = None):
        envelope = task if task is not None else task_envelope(
            {"messages": [{"role": "user", "content": "2+2?"}]}, result_key=CLIENT.public_key)
        # The container the order names: a 32-byte SEED sealed to whoever is
        # meant to open it (this daemon, for a designated bid), and the envelope
        # encrypted under the key derived from that seed and the order's owner.
        self.escrow = escrow
        container = seal_container(envelope, recipient=recipient or DAEMON_BOX, owner=OWNER)
        self.task_cid = fake_cid(container)
        # The pin the order names. `pin=False` stands in for bytes that no source
        # can serve — the daemon must give the claim back rather than run blind.
        self.blobs = {self.task_cid: container} if pin else {}
        # One chain-shaped job row, as the node serialises it: the rates are USD
        # decimal strings, `state` and `ended_because` are small ints, and the
        # model and deadline are numbers the catalog names.
        self.job = {
            "job_id": job_id_of(OWNER, container), "model_id": str(MODEL_ID), "state": 0,
            "sla_secs": str(SLA_SECS), "owner": OWNER, "provider_id": "0",
            "rate_in": "0.3", "rate_out": "0.8",
            "expires_at": None, "claimed_at": "0", "ended_because": 0,
            "designated": str(designated),
            "task_cid": self.task_cid, "units_in": "5", "units_out": "64",
            "result_cid": None, "completion_tok": "0",
        }
        self.asks_history: list[dict] = []
        self.ops: list[tuple[str, dict]] = []
        self.simulated: list[dict] = []
        self.releases: list[dict] = []
        self.settled = None
        self.capacity_requested = None
        self.session_calls = 0
        # The daemon's box public key, set by the test before start() so the
        # registry record matches (a mismatch is fatal). Overriding it with a
        # different key exercises the box-key-mismatch guard.
        self.box_public_key: str | None = None
        # 403 not_registered for the first N session handshakes, then registered.
        self.unregistered_calls = 0
        self.failed = None

    def _provider_rec(self):
        return {"provider_id": str(PROVIDER_ID), "operator": "0x" + "ab" * 20,
                "box_key": None if self.box_public_key is None else "0x" + self.box_public_key,
                "evidence": None, "listed": True, "reputation": "200",
                "allow_all_models": True, "allowed_models": [],
                "capacity": 2, "active_jobs": 0}

    def handler(self, req: httpx.Request) -> httpx.Response:
        host, path = req.url.host, req.url.path
        if host == BACKEND and path == "/metrics":
            # The runtime's own Prometheus endpoint, exactly as it exports it:
            # the daemon reads what the operator's monitoring already reads, and
            # asks the backend for nothing.
            return httpx.Response(200, text="vllm:kv_cache_usage_perc 0.25\n")
        if host == BACKEND and path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}], "usage": {"completion_tokens": 3}})
        if path == "/auth/nonce":
            return httpx.Response(200, json={"nonce": f"n{self.session_calls}", "expires_at": 9_999_999_999, "chain_id": 84532})
        if path == "/auth/session":
            self.session_calls += 1
            if self.session_calls <= self.unregistered_calls:
                return httpx.Response(403, json={"error": {"message": "Wallet is not registered as a provider.",
                                                           "type": "authentication_error", "code": "not_registered"}})
            return httpx.Response(200, json={"token": f"vorq_sess_{self.session_calls}",
                                             "expires_at": 9_999_999_999, "provider_id": PROVIDER_ID})
        if path.startswith("/evm/providers/") and req.method == "GET":
            return httpx.Response(200, json=self._provider_rec())
        if path == "/evm/asks" and req.method == "PUT":
            body = json.loads(req.content)
            assert set(body) == {"snapshot", "signature"}
            assert body["signature"].startswith("0x")
            self.asks_history.append(body["snapshot"])
            return httpx.Response(200, json={"published": True, "tx_hash": "0x" + "44" * 32})
        if path == "/evm/jobs" and req.method == "GET":
            # The book answers the state it was asked for, as the node does. A
            # fixture that ignored `state` would serve an Open job to the
            # boot-recovery read, and the daemon would run it twice: once
            # "recovered" and once claimed.
            wanted = {"Open": 0, "Claimed": 1}.get(req.url.params.get("state"))
            jobs = [self.job] if self.job["state"] == wanted else []
            if "free" in req.url.params:
                # The provider poll: session-gated, Open only, at most `free`.
                assert req.headers.get("authorization", "").startswith("Bearer vorq_sess_")
                assert wanted == 0 and "model" in req.url.params
                jobs = jobs[: int(req.url.params["free"])]
            return httpx.Response(200, json={"jobs": jobs, "as_of_block": "1"})
        if path.startswith("/ipfs/"):
            cid = path.rsplit("/", 1)[-1]
            if cid not in self.blobs:
                return httpx.Response(404, json={"error": {"message": "no blob", "type": "not_found"}})
            return httpx.Response(200, content=self.blobs[cid])
        if path == "/evm/models":
            # What a coordinator node serves: an id, a name and an enabled flag,
            # and no modality at all — the on-chain model record has none. The
            # config declares it instead (see `make_config`), which is the only
            # other source there is.
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"id": MODEL, "object": "model", "owned_by": "vorq",
                          "vorq": {"model_id": str(MODEL_ID), "enabled": True}}],
                "as_of_block": "1"})
        if path == "/evm/chain":
            return httpx.Response(200, json=CHAIN_BODY)
        if path == "/evm/simulate/claim":
            self.simulated.append(json.loads(req.content))
            return httpx.Response(200, json={"ok": True})
        if path == "/evm/ops":
            return self._op(req_op(req))
        if path == "/release":
            return self._release(json.loads(req.content))
        return httpx.Response(404, json={"error": {"message": "no route", "type": "not_found"}})

    def _op(self, body: dict) -> httpx.Response:
        """`POST /evm/ops` — the one door that costs money, and the only one the
        daemon signs for. Flat fields beside `op` and `signature`; a settle's
        `result` is the sealed bytes, base64."""
        assert body["signature"].startswith("0x")
        payload = {k: v for k, v in body.items() if k not in ("op", "signature")}
        self.ops.append((body["op"], payload))
        if body["op"] == "request_capacity":
            self.capacity_requested = payload["n"]
            return httpx.Response(201, json={"tx_hash": "0x" + "55" * 32, "status": "success",
                                             "block_number": 4})
        if body["op"] == "set_identity":
            self.box_public_key = payload["box_key"].removeprefix("0x")
            return httpx.Response(201, json={"tx_hash": "0x" + "66" * 32, "status": "success",
                                             "block_number": 5})
        if body["op"] == "claim":
            self.job["state"] = 1
            self.job["provider_id"] = str(PROVIDER_ID)
            self.job["claimed_at"] = str(int(payload["issued_at"]))
            return httpx.Response(201, json={"tx_hash": "0x" + "11" * 32, "status": "success",
                                             "block_number": 1})
        if body["op"] == "settle":
            self.settled = payload
            self.job["state"] = 2
            # The name is minted by the node from the bytes this op delivered, and
            # rides on the answer because nothing the claimant holds could
            # predict it.
            assert isinstance(payload["result"], str) and payload["result"]
            self.job["result_cid"] = fake_cid(base64.b64decode(payload["result"]))
            return httpx.Response(201, json={"tx_hash": "0x" + "22" * 32, "status": "success",
                                             "block_number": 2,
                                             "result_cid": self.job["result_cid"]})
        if body["op"] == "fail":
            self.failed = payload
            self.job["state"] = 3
            return httpx.Response(201, json={"tx_hash": "0x" + "33" * 32, "status": "success",
                                             "block_number": 3})
        return httpx.Response(400, json={"error": {"message": "unknown op", "type": "invalid_request"}})

    def _release(self, body: dict) -> httpx.Response:
        """The escrow's key oracle: unseal the seed, derive against the owner it
        read from chain, re-seal the result to the key this request named."""
        self.releases.append(body)
        if self.escrow is None:
            return httpx.Response(403, json={"error": {"message": "escrow is off",
                                                       "type": "invalid_request_error",
                                                       "code": "escrow_unavailable"}})
        seed = self.escrow.decrypt(base64.b64decode(body["seed_wrap"]))
        dek = derive_dek(seed, self.job["owner"])
        sealed = seal_to(body["response_pubkey"], dek)
        return httpx.Response(200, json={"dek_sealed": base64.b64encode(sealed).decode()})


def make_config(poll_interval_s: float = 5.0, *, load: LoadProbeConfig | None = None,
                state_db: str | None = None):
    return VorqdConfig(
        provider=ProviderConfig(wallet_key="0x" + "4a" * 32, box_key=BOX_KEY, api_url=f"http://{COORD}",
                                capacity=2, metrics_port=0, poll_interval_s=poll_interval_s,
                                state_db=state_db),
        models=[ModelConfig(
            # Where this backend reports how busy it is, if the operator said.
            load=load,
            model="deepseek-ai/deepseek-v4-pro:fp8",
            # Scaled integers: rates are atomic token units per RATE_SCALE units
            # of work, and the ask book carries them as uint128.
            slas={"1h": SlaRate(rate_in="0.2", rate_out="0.6")},
            backend=BackendConfig(preset="openai-chat", params={"base_url": f"http://{BACKEND}/v1", "model": "runtime"}),
            # The node's catalog names no modality, so the operator's own
            # declaration is what makes this model servable at all.
            modality="text",
        )],
    )


def job_ops(stack: FakeStack) -> list[str]:
    """The ops a *job* caused. Startup signs two of its own — `request_capacity`
    always, `set_identity` on a confidential boot — and they are asserted where
    they belong rather than prefixed onto every claim/settle sequence."""
    return [op for op, _ in stack.ops if op not in ("request_capacity", "set_identity")]


def live_quotes(stack: FakeStack) -> list[dict]:
    """The priced rows in the last pushed snapshot; a withdrawal is not one."""
    return [q for q in stack.asks_history[-1]["quotes"] if q["rate_out"] != "0"]


def _build(stack: FakeStack, **cfg_kwargs):
    daemon = build_daemon(make_config(**cfg_kwargs), transport=httpx.MockTransport(stack.handler), clock=lambda: 1001.0)
    stack.box_public_key = daemon._sched._cipher.public_key   # the record must match the local box key
    return daemon


async def test_a_model_that_reports_its_load_gets_a_live_probe():
    """The wiring, end to end: a `load:` block becomes a probe on the daemon's
    own HTTP client, reading the runtime's own `/metrics`.

    Without this line the feature is dormant and silently so — the scheduler
    builds itself the null monitor, every model's load is unknown, unknown
    prices exactly like busy, and the daemon runs, claims and settles at its
    configured rates while the operator believes a discount is in force.
    """
    stack = FakeStack()
    daemon = _build(stack, load=LoadProbeConfig(url=f"http://{BACKEND}/metrics",
                                                metric="vllm:kv_cache_usage_perc"))
    monitor = daemon._sched._load
    assert isinstance(monitor, LoadMonitor)

    await monitor.refresh()
    assert monitor.load(MODEL) == 0.25


def test_a_daemon_with_no_load_block_is_priced_exactly_as_configured():
    """No probe anywhere is the null monitor, and the null monitor answers
    "unknown" for every model — which is the busy answer, so nothing is
    discounted."""
    monitor = _build(FakeStack())._sched._load
    assert isinstance(monitor, NullLoadMonitor)
    assert monitor.load(MODEL) is None


def test_the_daemon_waits_out_a_coordinator_door_that_pins():
    """httpx's default read timeout is 5 s, and the settle door is slower than that.

    `POST /evm/ops` is not a database write. On the settle branch the coordinator
    files the sealed result with an object store over the public internet and
    answers with the name that store minted, so the daemon is waiting on somebody
    else's storage network, not on the API. Five seconds is inside the range a
    real store takes; the daemon then abandons a settle that is still in flight,
    reports a failure it did not have, and the job is lost at its own cost —
    which is exactly what it did the first time this ran against a real bucket.

    `connect` stays short because a coordinator that is not listening is not
    worth waiting on, and that is a different fact from a door that is working.
    """
    daemon = build_daemon(make_config(), transport=httpx.MockTransport(FakeStack().handler))
    timeout = daemon._client.timeout

    assert timeout.read is not None and timeout.read >= 60, (
        f"the daemon abandons a coordinator door after {timeout.read}s, which is inside the "
        "time a real object store takes to answer a settle"
    )
    assert timeout.connect is not None and timeout.connect <= 15


def test_a_window_with_its_own_backend_is_wired_as_a_second_driver():
    """`sla_backends` reaches the scheduler as a driver of its own, keyed by
    model and window; the model's default driver stays where it was."""
    config = make_config()
    model = config.models[0]
    config.models[0] = replace(model, sla_backends={
        "1h": BackendConfig(preset="openai-batch",
                            params={"base_url": f"http://{BACKEND}/v1", "model": "runtime"}),
    })
    daemon = build_daemon(config, transport=httpx.MockTransport(FakeStack().handler))

    sched = daemon._sched
    assert set(sched._sla_drivers) == {(model.model, "1h")}
    assert sched._sla_drivers[(model.model, "1h")] is not sched._drivers[model.model]


async def test_daemon_requests_capacity_publishes_and_settles():
    stack = FakeStack()
    daemon = _build(stack)
    await daemon.start()
    # Capacity is a signed ProviderRegistry op now, not a request body.
    assert stack.capacity_requested == 2
    assert [op for op, _ in stack.ops] == ["request_capacity"]
    snapshot = stack.asks_history[-1]
    assert set(snapshot) == {"provider_id", "signed_at", "quotes"}
    # The book is ids and seconds, and it names its own publisher because
    # `setAsks` compares that against `idOf(signer)` and skips a mismatch.
    assert snapshot["provider_id"] == PROVIDER_ID
    assert live_quotes(stack) == [{"model_id": MODEL_ID, "sla": SLA_SECS,
                                   "rate_in": "0.2", "rate_out": "0.6"}]

    await daemon.run_once()
    await daemon.join()
    assert stack.settled is not None
    # One op, carrying the bytes. Identity is the session and the signature, never
    # a body field; and the name of what was delivered is the node's to mint, so
    # nothing the daemon signed could have contained one.
    assert job_ops(stack) == ["claim", "settle"]
    assert set(stack.settled) == {"job_id", "completion_tok", "result", "issued_at"}
    assert stack.simulated == [{"job_id": stack.job["job_id"],
                                "address": stack.simulated[0]["address"]}]
    assert stack.settled["completion_tok"] == 3
    sealed = json.loads(base64.b64decode(stack.settled["result"]))
    assert sealed["enc"] == "vorq-sealed-v1"
    # ...and it opens with the key the client put inside its own sealed envelope.
    assert json.loads(CLIENT.decrypt(base64.b64decode(sealed["ciphertext"])))["choices"]
    assert stack.job["state"] == 2   # Settled

    await daemon.shutdown()
    # Withdrawn on drain — and withdrawal is a quote at rate_out 0, not silence:
    # on chain a slot the snapshot omits keeps its price forever.
    assert stack.asks_history[-1]["quotes"] == [
        {"model_id": MODEL_ID, "sla": SLA_SECS, "rate_in": "0", "rate_out": "0"}]
    assert live_quotes(stack) == []


async def test_daemon_serve_stops_cleanly_on_event():
    import asyncio

    stack = FakeStack()
    daemon = _build(stack)
    await daemon.start()
    stop = asyncio.Event()
    stop.set()  # SIGTERM before any sweep -> no new claims, just drain + close cleanly
    await daemon.serve(stop)
    assert stack.job["state"] == 0             # never claimed
    assert live_quotes(stack) == []            # asks withdrawn on drain


async def test_startup_fails_on_box_key_mismatch():
    # The coordinator holds a different box key than the daemon's local cipher:
    # sealed payloads would be undecryptable, so startup is fatal.
    stack = FakeStack()
    daemon = _build(stack)
    stack.box_public_key = "00" * 32   # override with a key that is not the daemon's
    with pytest.raises(ConfigError, match="box public key"):
        await daemon.start()
    await daemon.shutdown()


def test_a_state_db_the_daemon_cannot_open_is_a_named_config_error(tmp_path):
    # sqlite answers "unable to open database file" and names nothing an operator
    # can act on. The default path is relative, so a container that runs the image
    # from a directory its user cannot write hits exactly this — and has to be
    # told which setting to point somewhere else.
    config = make_config(state_db=str(tmp_path / "missing" / "state.sqlite"))
    with pytest.raises(ConfigError, match="state_db"):
        build_daemon(config)


async def test_unresolvable_task_bytes_are_never_claimed(monkeypatch):
    # The order names a CID no source can serve. The fetch runs before the claim,
    # so the daemon simply never claims it: the order stays on the book for a
    # provider whose gateway answers, and this one signed nothing. (The
    # propagation backoff is zeroed so the test does not wait out a pin that was
    # never made — what is under test is the refusal, not the patience.)
    monkeypatch.setattr("vorqd.blob.GATEWAY_BACKOFF_S", 0)
    stack = FakeStack(pin=False)
    daemon = _build(stack)
    await daemon.start()
    await daemon.run_once()
    await daemon.join()
    assert stack.settled is None
    assert stack.failed is None
    assert job_ops(stack) == []             # not one signed op for the job
    assert stack.job["state"] == 0          # Open
    await daemon.shutdown()


async def test_envelope_addressed_to_another_owner_is_refused():
    # Somebody else's sealed terms, replayed under a job this wallet paid for (D8).
    victim = "0x" + "22" * 20
    stack = FakeStack(task=task_envelope({"messages": []}, owner=victim))
    daemon = _build(stack)
    await daemon.start()
    await daemon.run_once()
    await daemon.join()
    assert stack.settled is None
    # The claim went out, the payload was refused, and the job was handed back at
    # once. The reason is not on the wire — a Fail op carries the job id and a
    # timestamp — so it is the op sequence that says what happened.
    assert job_ops(stack) == ["claim", "fail"]
    assert set(stack.failed) == {"job_id", "issued_at"}
    assert stack.failed["job_id"] == stack.job["job_id"]
    assert stack.job["state"] == 3   # Cancelled
    await daemon.shutdown()


async def test_startup_waits_for_admin_provisioning_then_proceeds():
    # The wallet is registered by the admin only after the daemon starts: the first
    # two handshakes 403 (not_registered); the daemon waits, then proceeds.
    stack = FakeStack()
    stack.unregistered_calls = 2
    daemon = _build(stack, poll_interval_s=0.01)   # tiny sleep between provisioning polls
    await daemon.start()
    assert stack.session_calls == 3   # two unregistered waits, then success
    assert stack.capacity_requested == 2
    assert live_quotes(stack)[0]["model_id"] == MODEL_ID
    await daemon.shutdown()


async def test_gateway_reads_are_the_only_read_path_with_a_propagation_budget(monkeypatch):
    # The daemon ships reading from the storage network's public gateway: no
    # configuration at all selects it, and the coordinator serves no bytes at all.
    # The retry budget is sized for a fresh pin's propagation — the sub-second
    # default would refuse real jobs seconds after their bytes were pinned.
    from vorqd.blob import DEFAULT_GATEWAY, GatewayBlobSource

    monkeypatch.delenv("VORQ_PIN_GATEWAY", raising=False)
    daemon = _build(FakeStack())
    resolver = daemon._sched._blobs
    (source,) = resolver._sources
    assert isinstance(source, GatewayBlobSource)
    assert source._gateway == DEFAULT_GATEWAY
    assert resolver._attempts >= 5
    assert resolver._attempts * resolver._backoff_s >= 5.0

    monkeypatch.setenv("VORQ_PIN_GATEWAY", "https://gw.test")
    daemon = _build(FakeStack())
    (source,) = daemon._sched._blobs._sources
    assert isinstance(source, GatewayBlobSource)
    assert source._gateway == "https://gw.test"


async def test_an_open_bid_is_released_by_the_escrow_and_settles():
    # An open order's `designated` is 0, and its wrap is sealed to the escrow's
    # key rather than to this daemon's. The shipped wiring signs a release
    # request against the node's own escrow, unseals the answer with a keypair
    # made for that one request, and runs the job like any other.
    escrow = BoxCipher.generate()
    stack = FakeStack(recipient=escrow.public_key, designated=0, escrow=escrow)
    daemon = _build(stack)
    await daemon.start()
    await daemon.run_once()
    await daemon.join()

    (release,) = stack.releases
    assert set(release) == {"job_id", "seed_wrap", "ct_hash", "response_pubkey",
                            "issued_at", "signature"}
    assert release["job_id"] == stack.job["job_id"]
    assert job_ops(stack) == ["claim", "settle"]
    assert stack.settled is not None and stack.failed is None
    await daemon.shutdown()


async def test_an_open_bid_the_escrow_will_not_serve_is_refused_under_its_own_code():
    # The escrow says it cannot serve this endpoint. The daemon hands the claim
    # straight back under the escrow's own code — it never runs blind, and it
    # never sources a key from another surface.
    escrow_key = BoxCipher.generate().public_key
    stack = FakeStack(recipient=escrow_key, designated=0, escrow=None)   # /release refuses
    daemon = _build(stack)
    await daemon.start()
    await daemon.run_once()
    await daemon.join()
    assert stack.settled is None
    assert job_ops(stack) == ["claim", "fail"]
    assert stack.failed["job_id"] == stack.job["job_id"]
    assert stack.job["state"] == 3   # Cancelled: refunded now, not at SLA expiry
    await daemon.shutdown()
