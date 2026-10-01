"""Integration harness: a MockTransport standing in for a coordinator node,
composed with an in-test backend, so the real Scheduler / NodeClient /
CoordinatorClient / BackendDriver run end-to-end without a live node.

Configs are loaded from the public example YAMLs so the shapes stay honest.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
from pathlib import Path

import httpx

from vorqd._crypto import BoxCipher

from ..conftest import fake_cid, job_id_of, req_op, seal_container
from vorqd.config import load_config

EXAMPLES = Path(__file__).parents[2] / "docs" / "examples"

#: What `GET /evm/chain` serves: the chain id and the four contracts the ops sign
#: against.
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

OWNER = "0x" + "11" * 20
CLIENT = BoxCipher.generate()   # the key the client names inside its own envelope


def task_envelope(payload: dict, *, owner: str = OWNER, result_key: str | None = None) -> bytes:
    """The bytes a client pins for its order: the sealed-envelope shape, pinned in
    the clear here so the harness stays about the daemon loop, not about crypto.
    The result key is always named — the daemon refuses an envelope without one."""
    return json.dumps({"v": "vorq-env-v1", "owner": owner,
                       "result_key": result_key or CLIENT.public_key,
                       "input": payload}, sort_keys=True, separators=(",", ":")).encode()


def daemon_box_public_key() -> str:
    """The box key the example configs give the daemon — what a designated bid's
    ``seed_wrap`` must be sealed to for the daemon to open it in-process."""
    os.environ.setdefault("VORQ_BOX_KEY", "aa" * 32)
    return BoxCipher(os.environ["VORQ_BOX_KEY"]).public_key


def pinned_job(job: dict, payload: dict) -> tuple[dict, bytes]:
    """Name a fixture job by its content: build the container the order names,
    pin it, and derive the job id from its commitment.

    These fixtures are designated bids — the DEK is sealed to the daemon's own
    box key, so the container opens with no escrow in the path.
    """
    container = seal_container(task_envelope(payload), recipient=daemon_box_public_key(),
                               owner=OWNER)
    job = {**job, "owner": OWNER, "task_cid": fake_cid(container),
           "job_id": job_id_of(OWNER, container),
           "designated": job.get("designated", Emulator.PROVIDER_ID)}
    return job, container


async def no_sleep(_seconds):
    return None


def load_example(name: str):
    """Load an example config, forcing the ops port to an ephemeral one and the
    in-flight store into memory — an in-process daemon writes no sqlite file."""
    # The examples reference the required box key via env: indirection; supply a
    # valid throwaway Curve25519 key so the daemon's box cipher builds.
    os.environ.setdefault("VORQ_BOX_KEY", "aa" * 32)
    cfg = load_config(EXAMPLES / name)
    provider = dataclasses.replace(cfg.provider, metrics_port=0, state_db=None)
    return dataclasses.replace(cfg, provider=provider)


#: ``JobState`` from the contracts' ``Types.sol``, so a fixture can spell a state
#: the way the node serves it — a small integer.
STATES = {"Open": 0, "Claimed": 1, "Settled": 2, "Cancelled": 3}


class Emulator:
    """A coordinator node's surface, backed by an in-memory job store."""

    PROVIDER_ID = 7   # admin-issued; ambient from the session

    def __init__(self, jobs: list[dict], models: list[dict] | None = None):
        pinned = [pinned_job(j, j.get("input_body", {})) for j in jobs]
        self.jobs = {j["job_id"]: j for j, _ in pinned}
        self.blobs = {j["task_cid"]: raw for j, raw in pinned}
        self.models = models or []
        self.settled: dict[str, dict] = {}
        self.failed: dict[str, dict] = {}
        self.uploads: list[bytes] = []
        self.asks_history: list[dict] = []
        self.ops: list[tuple[str, dict]] = []
        self.claim_conflicts = 0  # force N claim races before succeeding
        self.capacity_requested: int | None = None
        # Set by the test harness (run helpers) to the daemon's box public key so the
        # provider record matches — a mismatch would fail startup.
        self.box_public_key: str | None = None
        self._fid = 0

    def _provider_rec(self):
        return {"provider_id": str(self.PROVIDER_ID), "operator": "0x" + "ab" * 20,
                "box_key": None if self.box_public_key is None else "0x" + self.box_public_key,
                "evidence": None, "listed": True, "reputation": "200",
                "allow_all_models": True, "allowed_models": [],
                "capacity": 1, "active_jobs": 0}

    def live_quotes(self) -> list[dict]:
        """The priced rows in the last snapshot pushed; a withdrawal is not one."""
        return [q for q in self.asks_history[-1]["quotes"] if q["rate_out"] != "0"]

    def route(self, req: httpx.Request) -> httpx.Response | None:
        path = req.url.path
        method = req.method
        if path == "/auth/nonce":
            return httpx.Response(200, json={"nonce": "n", "expires_at": 9_999_999_999, "chain_id": 84532})
        if path == "/auth/session":
            return httpx.Response(200, json={"token": "vorq_sess_1", "expires_at": 9_999_999_999,
                                             "provider_id": self.PROVIDER_ID})
        if path.startswith("/evm/providers/") and method == "GET":
            return httpx.Response(200, json=self._provider_rec())
        if path == "/evm/asks" and method == "PUT":
            body = json.loads(req.content)
            assert set(body) == {"snapshot", "signature"}
            assert body["signature"].startswith("0x")
            self.asks_history.append(body["snapshot"])
            return httpx.Response(200, json={"published": True, "tx_hash": "0x" + "44" * 32})
        if path == "/evm/jobs" and method == "GET":
            state = req.url.params.get("state")
            model = req.url.params.get("model")
            provider = req.url.params.get("provider")
            jobs = [j for j in self.jobs.values()
                    if (state is None or j["state"] == STATES[state])
                    and (model is None or j["model_id"] == model)
                    and (provider is None or str(j.get("provider_id")) == provider)]
            if "free" in req.url.params:
                # The provider poll: session-gated, Open rows on the model, at most `free`.
                assert req.headers.get("authorization", "").startswith("Bearer vorq_sess_")
                assert state == "Open" and model is not None
                jobs = jobs[: int(req.url.params["free"])]
            return httpx.Response(200, json={"jobs": jobs, "as_of_block": "1"})
        if path.startswith("/ipfs/"):
            cid = path.rsplit("/", 1)[-1]
            if cid not in self.blobs:
                return httpx.Response(404, json={"error": {"message": "no blob", "type": "not_found"}})
            return httpx.Response(200, content=self.blobs[cid])
        if path == "/evm/models":
            return httpx.Response(200, json={"object": "list", "data": self.models,
                                             "as_of_block": "1"})
        if path == "/evm/chain":
            return httpx.Response(200, json=CHAIN_BODY)
        if path == "/evm/simulate/claim":
            # The advisory gate. It refuses while a race is being forced, which
            # is what a daemon that is about to lose one actually sees.
            if self.claim_conflicts > 0:
                return httpx.Response(200, json={"ok": False, "reason": "NotOpen"})
            return httpx.Response(200, json={"ok": True})
        if path == "/evm/ops":
            return self.op(req_op(req))
        return None  # not a coordinator route -> fall through to the backend

    def op(self, body: dict) -> httpx.Response:
        """`POST /evm/ops`: one signed op, relayed. Flat fields beside `op` and
        `signature`; a settle's `result` is the sealed bytes, base64."""
        assert body["signature"].startswith("0x")
        op = body["op"]
        payload = {k: v for k, v in body.items() if k not in ("op", "signature")}
        self.ops.append((op, payload))
        if op == "request_capacity":
            self.capacity_requested = payload["n"]
            return httpx.Response(201, json={"tx_hash": "0x" + "55" * 32, "status": "success",
                                             "block_number": 4})
        if op == "set_identity":
            self.box_public_key = payload["box_key"].removeprefix("0x")
            return httpx.Response(201, json={"tx_hash": "0x" + "66" * 32, "status": "success",
                                             "block_number": 5})
        job_id = payload["job_id"]
        if op == "claim":
            if self.claim_conflicts > 0:
                self.claim_conflicts -= 1
                return httpx.Response(409, json={"ok": False, "reason": "NotOpen"})
            self.jobs[job_id].update(state=STATES["Claimed"], provider_id=str(self.PROVIDER_ID),
                                     claimed_at=str(int(payload["issued_at"])))
            return httpx.Response(201, json={"tx_hash": "0x" + "11" * 32, "status": "success",
                                             "block_number": 1})
        if op == "settle":
            payload["completion_tok"] = int(payload["completion_tok"])
            self.settled[job_id] = payload
            # The node pins the delivered bytes and mints the name; it comes back
            # on the answer because the claimant cannot compute it.
            assert isinstance(payload["result"], str) and payload["result"]
            cid = fake_cid(base64.b64decode(payload["result"]))
            self.jobs[job_id].update(state=STATES["Settled"], result_cid=cid)
            return httpx.Response(201, json={"tx_hash": "0x" + "22" * 32, "status": "success",
                                             "block_number": 2, "result_cid": cid})
        if op == "fail":
            # No reason travels: a Fail op is the job id and a timestamp.
            self.failed[job_id] = payload
            self.jobs[job_id].update(state=STATES["Cancelled"])
            return httpx.Response(201, json={"tx_hash": "0x" + "33" * 32, "status": "success",
                                             "block_number": 3})
        return httpx.Response(400, json={"error": {"message": "unknown op", "type": "invalid_request"}})


def transport_for(emulator: Emulator, backend):
    def handler(req: httpx.Request) -> httpx.Response:
        return emulator.route(req) or backend(req)

    return httpx.MockTransport(handler)
