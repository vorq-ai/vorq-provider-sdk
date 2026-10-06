"""``vorqd.node`` — the daemon's one door onto the chain.

The node is stood in for by a routed ``httpx.MockTransport``, so every assertion
here is about the exact bytes this client puts on the wire and the exact shapes
it makes of what comes back.
"""

from __future__ import annotations

import base64
import inspect
from dataclasses import fields

import httpx
import pytest

from tests.conftest import json_response, mock_transport, req_form, req_json
from vorqd._crypto import WalletSigner
from vorqd.config import BackendConfig, ModelConfig, ProviderConfig, SlaRate, VorqdConfig
from vorqd.errors import (
    CatalogUnresolved,
    ChainConflict,
    OpRefused,
    OpRejected,
    UnknownModel,
    UploadInvalid,
)
from vorqd.node import (
    CatalogModel,
    ModelResolver,
    NodeClient,
    job_from_wire,
    sla_window_seconds,
)
from vorqd.cli import HTTP_TIMEOUT
from vorqd.money import format_usd
from vorqd.types import (
    ChainContext,
    ENVELOPE_RESERVE_BYTES,
    INLINE_MAX_BYTES,
    MAX_BODY_BYTES,
    base64_length,
)

API = "https://node.test"

CHAIN_BODY = {
    "chain_id": 84532,
    "contracts": {
        "job_registry": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
        "provider_registry": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
        "ask_registry": "0x5FC8d32690cc91D4c39d9d3abcBD16989F875707",
        "usdc": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
    },
    # The one shift between the wire's USD strings and the signed atomic rates.
    "decimals": 6,
    # Answered by the node, and deliberately never carried into ChainContext:
    # the daemon builds, funds and broadcasts nothing, and signs no payment
    # artifact.
    "token_domain": {"name": "USDC", "version": "2"},
    "head_block": "4242",
    "block_time_ms": 1000,
    "fee_bps": 250,
}

MODELS_BODY = {
    "object": "list",
    "data": [
        {"id": "org/e2ee-model:fp8", "object": "model", "owned_by": "vorq",
         "vorq": {"model_id": "7", "enabled": True}},
        {"id": "org/e2ee-media:v1", "object": "model", "owned_by": "vorq",
         "vorq": {"model_id": "9", "enabled": True}},
    ],
    "as_of_block": "4242",
}

#: One chain-shaped job row, exactly as the node serialises it: money a USD
#: decimal string, every other integer a JSON integer, and ``designated`` is
#: ``0`` for an open job.
OPEN_ROW = {
    "job_id": "0xb286522b28be749d46c85b069f4d981ddca94793a9c2429659e0de2bc27347c6",
    "owner": "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
    "c": "0x0fb4cbd8cf6e39d81226a3ba314952733089628fadd2ccbc17e12138eb9682ba",
    "model_id": 7,
    "sla_secs": 3600,
    "designated": 0,
    "rate_in": "0.0001",
    "rate_out": "0.0002",
    "units_in": 10,
    "units_out": 1500,
    "expires_at": 1800000000,
    "state": 0,
    "ended_because": 0,
    "provider_id": 0,
    "claimed_at": 0,
    "completion_tok": 0,
    "task_cid": "cid-task",
    "result_cid": None,
    "posted_block": 4200,
}

CLAIMED_ROW = {
    **OPEN_ROW,
    "state": 1,
    "designated": "1",
    "provider_id": "1",
    "claimed_at": "1799999000",
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def config(models: dict[str, dict[str, str]] | None = None) -> VorqdConfig:
    spec = models if models is not None else {"org/e2ee-model:fp8": {"1h": "0.6"}}
    return VorqdConfig(
        provider=ProviderConfig(wallet_key=None, api_url=API, capacity=4, box_key=None),
        models=[
            ModelConfig(
                model=name,
                slas={w: SlaRate(rate_in=None, rate_out=r) for w, r in windows.items()},
                backend=BackendConfig(preset="openai-chat", params={}),
            )
            for name, windows in spec.items()
        ],
    )


def resolver(models: dict[str, dict[str, str]] | None = None) -> ModelResolver:
    return ModelResolver.resolve(
        config(models),
        [
            CatalogModel(model_id=7, name="org/e2ee-model:fp8", enabled=True),
            CatalogModel(model_id=9, name="org/e2ee-media:v1", enabled=True),
        ],
    )


class Node:
    """A recording stand-in for a coordinator node."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []
        self.sessions = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/nonce":
            return json_response(200, {"nonce": "n-1", "chain_id": 84532})
        if path == "/auth/session":
            self.sessions += 1
            return json_response(
                200, {"token": f"vorq_sess_{self.sessions}", "expires_at": 4e9, "provider_id": 1}
            )
        if path == "/evm/chain" and "GET /evm/chain" not in self.routes:
            # The job doors read `decimals` off it; unrecorded unless a test routes it.
            return json_response(200, CHAIN_BODY)
        self.requests.append(request)
        route = self.routes.get(f"{request.method} {path}")
        if route is None:
            raise AssertionError(f"unrouted request: {request.method} {path}")
        return route(request) if callable(route) else route

    def client(self, **kwargs) -> NodeClient:
        http = httpx.AsyncClient(transport=mock_transport(self.handler))
        return NodeClient(http, API, WalletSigner.generate(), **kwargs)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def node(**routes) -> Node:
    """``node(**{"POST /evm/ops": response})`` — keys are ``METHOD path``."""
    return Node(routes)


# --------------------------------------------------------------------------- #
# The chain context (Q6)
# --------------------------------------------------------------------------- #


async def test_chain_context_carries_all_four_contracts() -> None:
    n = node(**{"GET /evm/chain": json_response(200, CHAIN_BODY)})
    ctx = await n.client().chain_context()

    assert ctx.chain_id == 84532
    assert ctx.job_registry == CHAIN_BODY["contracts"]["job_registry"]
    assert ctx.provider_registry == CHAIN_BODY["contracts"]["provider_registry"]
    assert ctx.ask_registry == CHAIN_BODY["contracts"]["ask_registry"]
    assert ctx.usdc == CHAIN_BODY["contracts"]["usdc"]
    assert ctx.decimals == 6
    assert not hasattr(ctx, "permit2")


async def test_chain_context_refuses_a_body_without_decimals() -> None:
    """Every rate converts by it; a guessed 6 would sign a different price."""
    body = {k: v for k, v in CHAIN_BODY.items() if k != "decimals"}
    n = node(**{"GET /evm/chain": json_response(200, body)})
    with pytest.raises(ValueError, match="decimals"):
        await n.client().chain_context()


async def test_chain_context_is_cached() -> None:
    n = node(**{"GET /evm/chain": json_response(200, CHAIN_BODY)})
    client = n.client()
    first = await client.chain_context()
    second = await client.chain_context()
    assert first is second
    assert len(n.requests) == 1


async def test_chain_context_refuses_a_body_missing_a_contract() -> None:
    """A missing address would be signed against ``None`` and refused silently."""
    body = {"chain_id": 84532, "contracts": dict(CHAIN_BODY["contracts"])}
    del body["contracts"]["provider_registry"]
    n = node(**{"GET /evm/chain": json_response(200, body)})
    with pytest.raises(ValueError, match="provider_registry"):
        await n.client().chain_context()


def test_chain_context_holds_no_rpc_concept() -> None:
    """No gas, no nonce, no block. The daemon funds and broadcasts nothing."""
    names = {f.name for f in fields(ChainContext)}
    assert not any(
        token in name for name in names for token in ("gas", "nonce", "block", "raw", "fee")
    )


# --------------------------------------------------------------------------- #
# POST /evm/ops — the wire body
# --------------------------------------------------------------------------- #


OP_OK = json_response(201, {"tx_hash": "0xtx", "status": "success", "block_number": "4243"})


async def test_push_op_posts_the_flat_op_fields_as_json() -> None:
    n = node(**{"POST /evm/ops": OP_OK})
    result = await n.client().push_op(
        "claim", {"job_id": "0xjob", "issued_at": 1800000000}, "0xsig"
    )

    assert n.last.method == "POST"
    assert n.last.url.path == "/evm/ops"
    assert n.last.headers["authorization"] == "Bearer vorq_sess_1"
    assert n.last.headers["content-type"] == "application/json"
    # No envelope: the op's fields sit beside `op` and `signature`.
    assert req_json(n.last) == {
        "op": "claim",
        "job_id": "0xjob",
        "issued_at": 1800000000,
        "signature": "0xsig",
    }
    assert (result.tx_hash, result.status, result.block_number) == ("0xtx", "success", 4243)
    assert result.result_cid is None


@pytest.mark.parametrize("op, payload", [
    ("fail", {"job_id": "0xjob", "issued_at": 1800000000}),
    ("set_identity", {"box_key": "0x" + "ab" * 32, "evidence": "0xcafe", "issued_at": 1}),
    ("request_capacity", {"n": 4, "issued_at": 1}),
])
async def test_every_op_that_carries_nothing_is_flat_json(op, payload) -> None:
    n = node(**{"POST /evm/ops": OP_OK})
    await n.client().push_op(op, payload, "0xsig")
    assert n.last.headers["content-type"] == "application/json"
    assert req_json(n.last) == {"op": op, **payload, "signature": "0xsig"}


async def test_a_settle_carries_the_inline_result_as_flat_json() -> None:
    """Same shape as every other op: no envelope, and the sealed bytes ride as
    base64 beside the signed fields rather than as a file part.

    The node pins those bytes, mints the name and puts it in the transaction,
    and the minted name comes back on the answer — the only way the claimant
    ever learns what its delivery was called.
    """
    sealed = b"sealed result bytes\r\n\x00\xff"
    n = node(**{
        "POST /evm/ops": json_response(
            201,
            {"tx_hash": "0xtx", "status": "success", "block_number": "5",
             "result_cid": "cid-minted-by-the-node"},
        )
    })
    payload = {
        "job_id": "0xjob",
        "completion_tok": 1500,
        "issued_at": 1800000000,
        "result": base64.b64encode(sealed).decode(),
    }
    result = await n.client().push_op("settle", payload, "0xsig")

    assert n.last.headers["content-type"] == "application/json"
    assert req_json(n.last) == {"op": "settle", **payload, "signature": "0xsig"}
    assert result.result_cid == "cid-minted-by-the-node"


async def test_a_settle_may_instead_carry_a_result_cid() -> None:
    """A result already uploaded is referenced, never re-sent."""
    n = node(**{"POST /evm/ops": OP_OK})
    payload = {"job_id": "0xjob", "completion_tok": 1500, "issued_at": 1800000000,
              "result_cid": "cid-uploaded-earlier"}
    await n.client().push_op("settle", payload, "0xsig")
    assert req_json(n.last) == {"op": "settle", **payload, "signature": "0xsig"}


async def test_a_413_from_the_node_propagates_as_a_status_error() -> None:
    """Past the ceiling is the node's verdict on the size, not the chain's on the op."""
    n = node(**{
        "POST /evm/ops": json_response(413, {"error": {"type": "invalid_request", "code": "file_too_large"}})
    })
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await n.client().push_op("settle", {"job_id": "0xjob", "result": "eA=="}, "0xsig")
    assert exc.value.response.status_code == 413


async def test_a_settle_naming_neither_result_nor_cid_is_refused_before_the_request() -> None:
    """A settle carrying nothing would still charge the client."""
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the request must not be built")

    n = Node({"POST /evm/ops": never})
    with pytest.raises(ValueError, match="result.*result_cid"):
        await n.client().push_op("settle", {"job_id": "0xjob"}, "0xsig")
    assert n.requests == []


async def test_upload_file_posts_purpose_then_the_file_and_returns_the_cid() -> None:
    """``purpose`` is a form field, read before the node touches a byte of the
    file; the file itself always goes as an opaque blob, whatever it holds."""
    n = node(**{
        "POST /v1/files": json_response(
            201, {"object": "file", "vorq": {"cid": "cid-minted-by-the-node"}}
        )
    })
    content = b"sealed result bytes\r\n\x00\xff"
    cid = await n.client().upload_file("result", content, filename="result")

    assert n.last.method == "POST"
    assert n.last.url.path == "/v1/files"
    fields, file = req_form(n.last)   # asserts no field follows the file part
    assert fields == {"purpose": "result"}
    assert file == ("file", content)
    assert b"Content-Type: application/octet-stream" in n.last.content
    assert cid == "cid-minted-by-the-node"


async def test_upload_file_refuses_an_answer_with_no_cid_in_it() -> None:
    """A 2xx with no ``vorq.cid`` — the key missing, or ``null`` — is a typed
    error, not a ``KeyError``/``TypeError``. The caller is the settle path, whose
    handlers catch what they know about; an untyped raise escapes all of them and
    leaves a claimed job with no fail report."""
    for body in ({"object": "file"}, {"object": "file", "vorq": {}},
                 {"object": "file", "vorq": {"cid": None}},
                 {"object": "file", "vorq": {"cid": ""}}):
        n = node(**{"POST /v1/files": json_response(201, body)})
        with pytest.raises(UploadInvalid, match="no vorq.cid"):
            await n.client().upload_file("result", b"x")


async def test_upload_file_raises_for_status_on_a_413() -> None:
    n = node(**{
        "POST /v1/files": json_response(413, {"error": {"type": "invalid_request", "code": "file_too_large"}})
    })
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await n.client().upload_file("result", b"x")
    assert exc.value.response.status_code == 413


# --------------------------------------------------------------------------- #
# POST /evm/ops — the answers
# --------------------------------------------------------------------------- #


async def test_a_409_is_a_typed_refusal_carrying_the_contract_error() -> None:
    n = node(**{"POST /evm/ops": json_response(409, {"ok": False, "reason": "NotOpen"})})
    with pytest.raises(OpRefused) as exc:
        await n.client().push_op("claim", {"job_id": "0xjob"}, "0xsig")
    assert exc.value.reason == "NotOpen"
    assert exc.value.raw is None
    # Typed, so no call site has to reach into a body.
    assert str(exc.value) == "NotOpen"


async def test_an_undecodable_revert_keeps_its_bytes() -> None:
    """A payment-token failure inside ``claim`` is in neither registry's ABI."""
    n = node(**{
        "POST /evm/ops": json_response(409, {"ok": False, "reason": "unknown", "raw": "0xdeadbeef"})
    })
    with pytest.raises(OpRefused) as exc:
        await n.client().push_op("claim", {"job_id": "0xjob"}, "0xsig")
    assert (exc.value.reason, exc.value.raw) == ("unknown", "0xdeadbeef")


async def test_a_403_is_a_rejection() -> None:
    n = node(**{
        "POST /evm/ops": json_response(
            403,
            {"error": {"type": "invalid_op_signature",
                       "message": "the op signature does not recover to a registered provider"}},
        )
    })
    with pytest.raises(OpRejected, match="does not recover"):
        await n.client().push_op("claim", {"job_id": "0xjob"}, "0xsig")


async def test_a_retryable_503_is_not_a_refusal() -> None:
    """The node failing to relay is not the chain refusing the op."""
    n = node(**{
        "POST /evm/ops": json_response(503, {"error": {"type": "relay_unavailable"}})
    })
    with pytest.raises(httpx.HTTPStatusError):
        await n.client().push_op("claim", {"job_id": "0xjob"}, "0xsig")


async def test_a_401_re_handshakes_once() -> None:
    seen: list[str] = []

    def door(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        if len(seen) == 1:
            return json_response(401, {"error": {"type": "authentication_error"}})
        return OP_OK

    n = Node({"POST /evm/ops": door})
    result = await n.client().push_op("claim", {"job_id": "0xjob"}, "0xsig")
    assert result.tx_hash == "0xtx"
    assert seen == ["Bearer vorq_sess_1", "Bearer vorq_sess_2"]
    assert n.sessions == 2


async def test_a_second_401_propagates() -> None:
    """Exactly one retry, and then it gives up.

    A session that is refused twice is not a stale token — it is a wallet the
    node will not authenticate — and re-handshaking in a loop would turn that
    into an unbounded spin against the auth surface.
    """
    n = node(**{"POST /evm/ops": json_response(401, {"error": {"type": "authentication_error"}})})
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await n.client().push_op("claim", {"job_id": "0xjob"}, "0xsig")
    assert exc.value.response.status_code == 401
    assert n.sessions == 2


# --------------------------------------------------------------------------- #
# The advisory gate
# --------------------------------------------------------------------------- #


async def test_simulate_claim_ok() -> None:
    n = node(**{"POST /evm/simulate/claim": json_response(200, {"ok": True})})
    answer = await n.client().simulate_claim("0xjob", "0xaddr")
    assert (answer.ok, answer.reason) == (True, None)
    assert req_json(n.last) == {"job_id": "0xjob", "address": "0xaddr"}


async def test_simulate_claim_surfaces_a_reason_typed() -> None:
    n = node(**{
        "POST /evm/simulate/claim": json_response(200, {"ok": False, "reason": "AtCapacity"})
    })
    answer = await n.client().simulate_claim("0xjob", "0xaddr")
    assert answer.ok is False
    assert answer.reason == "AtCapacity"


# --------------------------------------------------------------------------- #
# Q15 — model names and SLA windows to uint32 ids
# --------------------------------------------------------------------------- #


async def test_get_models_reads_the_numeric_catalog() -> None:
    n = node(**{"GET /evm/models": json_response(200, MODELS_BODY)})
    catalog = await n.client().get_models()
    assert catalog == [
        CatalogModel(model_id=7, name="org/e2ee-model:fp8", enabled=True),
        CatalogModel(model_id=9, name="org/e2ee-media:v1", enabled=True),
    ]


def test_resolver_maps_names_and_windows() -> None:
    r = resolver({"org/e2ee-model:fp8": {"1h": "0.6", "24h": "0.4"}})
    assert r.model_id("org/e2ee-model:fp8") == 7
    assert r.model_name(7) == "org/e2ee-model:fp8"
    assert r.sla_secs("24h") == 86_400
    assert r.sla_window("org/e2ee-model:fp8", 3600) == "1h"
    assert r.model_ids == {"org/e2ee-model:fp8": 7}


def test_a_model_the_catalog_does_not_carry_is_a_startup_failure() -> None:
    """Never a silent skip: the operator would run at a capacity they did not choose."""
    with pytest.raises(UnknownModel, match="does not carry it"):
        ModelResolver.resolve(
            config({"org/not-listed:fp8": {"1h": "0.6"}}),
            [CatalogModel(model_id=7, name="org/e2ee-model:fp8", enabled=True)],
        )


def test_an_unreadable_sla_window_is_a_startup_failure() -> None:
    with pytest.raises(UnknownModel, match="sla window"):
        resolver({"org/e2ee-model:fp8": {"soonish": "0.6"}})


def test_the_reverse_window_lookup_is_per_model() -> None:
    """Two models may spell the same deadline differently; each gets its own back."""
    r = resolver({"org/e2ee-model:fp8": {"24h": "0.6"}, "org/e2ee-media:v1": {"1d": "0.1"}})
    assert r.sla_window("org/e2ee-model:fp8", 86_400) == "24h"
    assert r.sla_window("org/e2ee-media:v1", 86_400) == "1d"


def test_a_deadline_this_model_never_priced_is_refused() -> None:
    r = resolver({"org/e2ee-model:fp8": {"1h": "0.6"}})
    with pytest.raises(UnknownModel, match="86400"):
        r.sla_window("org/e2ee-model:fp8", 86_400)


@pytest.mark.parametrize(
    "window,secs", [("30s", 30), ("5m", 300), ("1h", 3600), ("24h", 86_400), ("7d", 604_800)]
)
def test_window_parsing(window: str, secs: int) -> None:
    assert sla_window_seconds(window) == secs


@pytest.mark.parametrize("window", ["", "1", "h", "1w", "1 h", "-1h", "1.5h"])
def test_an_unparseable_window_raises_rather_than_defaulting(window: str) -> None:
    with pytest.raises(ValueError):
        sla_window_seconds(window)


# --------------------------------------------------------------------------- #
# The job book
# --------------------------------------------------------------------------- #


async def test_list_open_jobs_filters_by_model_id() -> None:
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [OPEN_ROW], "as_of_block": 1})})
    jobs = await n.client(models=resolver()).list_open_jobs(7)

    assert dict(n.last.url.params) == {"state": "Open", "model": "7"}
    assert len(jobs) == 1


async def test_a_job_at_a_window_this_model_does_not_quote_is_skipped() -> None:
    # The node leases by model and price, not by window: a 24 h order on a model
    # this daemon quotes at 1 h only still arrives. It is not this daemon's to
    # price, and it must not cost the rows around it — or the process.
    other_window = {**OPEN_ROW, "job_id": "0x" + "ab" * 32, "sla_secs": 86_400}
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [other_window, OPEN_ROW], "as_of_block": 1})})
    jobs = await n.client(models=resolver({"org/e2ee-model:fp8": {"1h": "0.6"}})).list_open_jobs(7)

    assert [job.job_id for job in jobs] == [OPEN_ROW["job_id"]]


async def test_list_open_jobs_narrows_the_book_to_what_this_daemon_can_act_on() -> None:
    """The filters go on the wire, or the narrowing did not happen at all."""
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [OPEN_ROW], "as_of_block": 1})})
    await n.client(models=resolver()).list_open_jobs(7, min_rate_in=160_000, min_rate_out=480_000)

    assert dict(n.last.url.params) == {
        "state": "Open", "model": "7", "min_rate_out": "0.48", "min_rate_in": "0.16",
    }


async def test_list_open_jobs_puts_a_rate_floor_on_the_wire_as_a_usd_string() -> None:
    """The floor is atomic inside the daemon and USD on the wire, exact at the
    token's decimals however wide it is."""
    floor = 2**127 + 1
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": 1})})
    await n.client(models=resolver()).list_open_jobs(7, min_rate_out=floor)

    assert n.last.url.params["min_rate_out"] == format_usd(floor, 6)


async def test_list_open_jobs_omits_every_filter_it_was_not_given() -> None:
    """``None`` is absent, never ``0``: a ``min_rate_in`` of 0 would read as a
    floor the daemon never set, and an unmetered input side would be excluded."""
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": 1})})
    await n.client(models=resolver()).list_open_jobs(7, min_rate_out=480_000)

    assert dict(n.last.url.params) == {"state": "Open", "model": "7", "min_rate_out": "0.48"}


ASSIGNED = {"jobs": [OPEN_ROW], "as_of_block": 1}


async def test_the_sweep_s_poll_is_the_open_book_with_free_under_the_session() -> None:
    """With ``free`` the same listing is *this provider's* poll: the coordinator
    leases it what it can start and answers only that, so the call rides the
    session — the node resolves the caller from the token and the daemon sends
    no provider id of its own."""
    n = node(**{"GET /evm/jobs": json_response(200, ASSIGNED)})
    jobs = await n.client(models=resolver()).list_open_jobs(
        7, free=2, min_rate_in=160_000, min_rate_out=480_000
    )

    assert n.last.headers["authorization"] == "Bearer vorq_sess_1"
    assert dict(n.last.url.params) == {
        "state": "Open", "model": "7", "free": "2",
        "min_rate_out": "0.48", "min_rate_in": "0.16",
    }
    assert len(jobs) == 1


async def test_the_public_read_carries_no_session() -> None:
    n = node(**{"GET /evm/jobs": json_response(200, ASSIGNED)})
    await n.client(models=resolver()).list_open_jobs(7)
    assert "authorization" not in n.last.headers


async def test_the_poll_omits_a_floor_it_was_not_given() -> None:
    """``None`` is absent, never ``0``: a ``min_rate_in`` of 0 would read as a
    floor the daemon never set."""
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": 1})})
    await n.client(models=resolver()).list_open_jobs(7, free=1, min_rate_out=480_000)

    assert dict(n.last.url.params) == {
        "state": "Open", "model": "7", "free": "1", "min_rate_out": "0.48",
    }


async def test_the_poll_puts_a_rate_floor_on_the_wire_as_a_usd_string() -> None:
    floor = 2**127 + 1
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": 1})})
    await n.client(models=resolver()).list_open_jobs(7, free=1, min_rate_out=floor)

    assert n.last.url.params["min_rate_out"] == format_usd(floor, 6)


async def test_the_poll_re_handshakes_once_on_a_401() -> None:
    seen: list[str] = []

    def door(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        if len(seen) == 1:
            return json_response(401, {"error": {"type": "authentication_error"}})
        return json_response(200, ASSIGNED)

    n = Node({"GET /evm/jobs": door})
    jobs = await n.client(models=resolver()).list_open_jobs(7, free=1)
    assert len(jobs) == 1
    assert seen == ["Bearer vorq_sess_1", "Bearer vorq_sess_2"]


async def test_the_poll_records_the_block_it_was_answered_at() -> None:
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": "4242"})})
    client = n.client(models=resolver())
    await client.list_open_jobs(7, free=1)
    assert client.as_of_block == 4242


async def test_a_listing_records_the_block_it_was_answered_at() -> None:
    """`as_of_block` is on the envelope, not on any row — and it is the only
    clock the book has.

    A job carries `posted_block`, so its age is `as_of_block - posted_block`,
    and the client keeps whatever the last listing was answered at as that
    reference.
    """
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": "4242"})})
    client = n.client(models=resolver())
    assert client.as_of_block == 0        # nothing read yet: no reference at all

    await client.list_open_jobs(7)
    assert client.as_of_block == 4242


async def test_the_block_the_book_was_read_at_moves_with_the_book() -> None:
    """A stale reference would age every bid by the time since the daemon
    booted, so it is replaced on every read rather than latched once."""
    blocks = iter(["4242", "4300"])
    n = node(**{"GET /evm/jobs":
                lambda _req: json_response(200, {"jobs": [], "as_of_block": next(blocks)})})
    client = n.client(models=resolver())
    await client.list_open_jobs(7)
    await client.list_claimed_jobs(1)     # the claimed listing carries one too
    assert client.as_of_block == 4300


async def test_a_listing_without_an_envelope_block_keeps_the_last_one() -> None:
    """Absent is not zero. A body that carries no `as_of_block` is a node too
    old to serve one, and forgetting the reference would silently turn the age
    filter off instead of leaving it where it was."""
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [], "as_of_block": "4242"})})
    client = n.client(models=resolver())
    await client.list_open_jobs(7)
    n.routes["GET /evm/jobs"] = json_response(200, {"jobs": []})
    await client.list_open_jobs(7)
    assert client.as_of_block == 4242


async def test_the_block_time_is_read_from_the_chain_door() -> None:
    """Served by `GET /evm/chain`, and deliberately not a `ChainContext` field:
    the daemon signs nothing that mentions a block. It is held beside the
    context, on the one read that already happens, because turning an age in
    seconds into a block count needs it."""
    n = node(**{"GET /evm/chain": json_response(200, CHAIN_BODY)})
    client = n.client()
    assert await client.block_time_ms() == 1000
    await client.block_time_ms()
    assert len(n.requests) == 1           # the chain read is cached, as the context is


async def test_a_node_that_serves_no_block_time_yields_none() -> None:
    """Not a guess and not a default: an unknown block time means the daemon
    cannot convert an age, and it says so rather than inventing a rate."""
    body = {k: v for k, v in CHAIN_BODY.items() if k != "block_time_ms"}
    n = node(**{"GET /evm/chain": json_response(200, body)})
    assert await n.client().block_time_ms() is None


async def test_list_claimed_jobs_filters_by_provider_id() -> None:
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [CLAIMED_ROW]})})
    jobs = await n.client(models=resolver()).list_claimed_jobs(1)

    assert dict(n.last.url.params) == {"state": "Claimed", "provider": "1",
                                       "limit": "1000", "offset": "0"}
    assert jobs[0].state == "Claimed"
    assert jobs[0].provider == 1
    assert jobs[0].claimed_at == 1799999000


async def test_the_claimed_listing_reads_every_page() -> None:
    """Boot recovery forgets the handle of every job this listing does not
    carry, so a first page read as the whole answer would drop live work."""
    pages = iter([[CLAIMED_ROW] * 1000, [CLAIMED_ROW] * 3])
    n = node(**{"GET /evm/jobs": lambda _req: json_response(200, {"jobs": next(pages)})})
    jobs = await n.client(models=resolver()).list_claimed_jobs(1)

    assert len(jobs) == 1003
    assert [r.url.params["offset"] for r in n.requests] == ["0", "1000"]


async def test_a_truncated_short_page_is_followed() -> None:
    """The node cuts a page in the headers, not in the body: a short page that
    says it was truncated is not the end of the listing."""
    responses = iter([
        httpx.Response(200, json={"jobs": [CLAIMED_ROW, CLAIMED_ROW]},
                       headers={"x-vorq-page-truncated": "true", "x-vorq-next-offset": "2"}),
        json_response(200, {"jobs": [CLAIMED_ROW]}),
    ])
    n = node(**{"GET /evm/jobs": lambda _req: next(responses)})
    jobs = await n.client(models=resolver()).list_claimed_jobs(1)

    assert len(jobs) == 3
    assert [r.url.params["offset"] for r in n.requests] == ["0", "2"]


async def test_a_short_page_ends_the_claimed_listing() -> None:
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [CLAIMED_ROW]})})
    jobs = await n.client(models=resolver()).list_claimed_jobs(1)

    assert len(jobs) == 1
    assert len(n.requests) == 1


def test_an_open_job_is_designated_zero_and_never_null() -> None:
    """Q22: ``0`` is the contracts' open-order sentinel (``Order.designated``)."""
    job = job_from_wire(OPEN_ROW, resolver(), 6)
    assert job.designated == 0
    assert job.designated is not None


def test_a_designated_job_carries_its_provider_id() -> None:
    assert job_from_wire(CLAIMED_ROW, resolver(), 6).designated == 1


def test_job_translation_reads_money_as_atomic_units() -> None:
    job = job_from_wire(OPEN_ROW, resolver(), 6)
    assert job.job_id == OPEN_ROW["job_id"]
    assert job.owner == OPEN_ROW["owner"]
    assert job.model == "org/e2ee-model:fp8"
    assert job.sla == "1h"
    assert job.state == "Open"
    assert job.units_in == 10
    assert job.units_out == 1500
    assert job.expires_at == 1800000000
    assert job.rate_in == 100     # "0.0001" USD at 6 decimals
    assert job.rate_out == 200
    assert job.task_cid == "cid-task"
    assert job.result_cid is None
    assert job.ended_because is None
    # Unclaimed is None, not 0: a zero would put the SLA deadline in 1970.
    assert job.claimed_at is None


def test_a_job_rate_finer_than_the_token_is_refused() -> None:
    with pytest.raises(ValueError, match="fraction digits"):
        job_from_wire({**OPEN_ROW, "rate_out": "0.0000001"}, resolver(), 6)


def test_modality_is_not_on_the_wire() -> None:
    """It is a catalog fact about the model, never a term of the order.

    ``EvmJob.modality`` carries a placeholder the scheduler overwrites before
    anything meters a job — and refuses to claim at all when nothing can
    establish it — so what matters here is that no row can assert one.
    """
    assert "modality" not in OPEN_ROW
    assert job_from_wire(OPEN_ROW, resolver(), 6).modality == "text"


def test_no_key_ever_rides_on_a_job_row() -> None:
    """The DEK is derived from the seed inside the container's own ``seed_wrap``,
    sealed to this provider (designated) or to the escrow (open) — never delivered
    by the surface that answers a claim, and never by the book."""
    job = job_from_wire(CLAIMED_ROW, resolver(), 6)
    assert not hasattr(job, "dek")
    assert not any("key" in f.name or "dek" in f.name for f in fields(job))


def test_an_ended_job_names_its_cause_from_the_contracts_enum() -> None:
    for code, name in ((2, "cancelled"), (3, "provider_fail"), (4, "reclaim"), (5, "expired")):
        row = {**OPEN_ROW, "state": 3, "ended_because": code}
        assert job_from_wire(row, resolver(), 6).ended_because == name


async def test_the_job_book_cannot_be_read_before_the_catalog() -> None:
    """A job whose model the daemon cannot name is one it must not claim."""
    n = node(**{"GET /evm/jobs": json_response(200, {"jobs": [OPEN_ROW]})})
    with pytest.raises(CatalogUnresolved):
        await n.client().list_open_jobs(7)


async def test_bind_models_after_reading_the_catalog() -> None:
    n = node(**{
        "GET /evm/models": json_response(200, MODELS_BODY),
        "GET /evm/jobs": json_response(200, {"jobs": [OPEN_ROW]}),
    })
    client = n.client()
    client.bind_models(ModelResolver.resolve(config(), await client.get_models()))
    assert (await client.list_open_jobs(7))[0].model == "org/e2ee-model:fp8"


# --------------------------------------------------------------------------- #
# The other reads
# --------------------------------------------------------------------------- #


async def test_get_provider() -> None:
    row = {"provider_id": 1, "box_key": "0xabc", "listed": True, "capacity": 8}
    n = node(**{"GET /evm/providers/1": json_response(200, row)})
    assert await n.client().get_provider(1) == row


async def test_get_allowlist() -> None:
    body = {"entries": [{"key": "0x01", "status": 1, "entry": {}}], "as_of_block": 1}
    n = node(**{"GET /evm/allowlist": json_response(200, body)})
    assert await n.client().get_allowlist() == body


async def test_public_reads_carry_no_session() -> None:
    """None of these doors is gated, so none of them mints a token."""
    n = node(**{
        "GET /evm/chain": json_response(200, CHAIN_BODY),
        "GET /evm/models": json_response(200, MODELS_BODY),
        "GET /evm/allowlist": json_response(200, {"entries": []}),
    })
    client = n.client()
    await client.chain_context()
    await client.get_models()
    await client.get_allowlist()
    assert n.sessions == 0
    assert all("authorization" not in r.headers for r in n.requests)


# --------------------------------------------------------------------------- #
# The ask push
# --------------------------------------------------------------------------- #


SNAPSHOT = {
    "provider_id": 1,
    "signed_at": 1800000000,
    "quotes": [{"model_id": 7, "sla": 3600, "rate_in": "0.0001", "rate_out": "0.0002"}],
}


async def test_push_asks_posts_the_snapshot_and_its_signature() -> None:
    answer = {"provider_id": 1, "signed_at": 1800000000, "published": True, "tx_hash": "0xtx"}
    n = node(**{"PUT /evm/asks": json_response(200, answer)})
    assert await n.client().push_asks(SNAPSHOT, "0xsig") == answer

    assert n.last.method == "PUT"
    assert req_json(n.last) == {"snapshot": SNAPSHOT, "signature": "0xsig"}
    assert n.last.headers["authorization"] == "Bearer vorq_sess_1"


async def test_a_stale_snapshot_is_a_typed_conflict() -> None:
    n = node(**{
        "PUT /evm/asks": json_response(
            409,
            {"error": {"type": "invalid_request", "message": "not newer than this provider's floor",
                       "code": "stale_snapshot"}},
        )
    })
    with pytest.raises(ChainConflict) as exc:
        await n.client().push_asks(SNAPSHOT, "0xsig")
    assert exc.value.code == "stale_snapshot"


async def test_an_ask_push_from_the_wrong_signer_is_rejected() -> None:
    n = node(**{
        "PUT /evm/asks": json_response(
            403, {"error": {"type": "invalid_op_signature", "message": "not a registered provider"}}
        )
    })
    with pytest.raises(OpRejected):
        await n.client().push_asks(SNAPSHOT, "0xsig")


# --------------------------------------------------------------------------- #
# Q16 / Q21 — what is deliberately here, and what is deliberately not
# --------------------------------------------------------------------------- #


async def test_the_provider_id_comes_from_the_session_handshake() -> None:
    """Q16: ``POST /auth/session`` already answers it. No new route."""
    n = node(**{"POST /evm/ops": OP_OK})
    client = n.client()
    assert client.provider_id is None
    await client.push_op("claim", {"job_id": "0xjob"}, "0xsig")
    assert client.provider_id == 1


def test_the_daemon_holds_no_rpc_concept() -> None:
    """Q21, as a forward prohibition: none of this is ever introduced here."""
    surface = {name for name in dir(NodeClient) if not name.startswith("__")}
    for banned in ("send_raw", "tx_nonce", "nonce", "gas_price", "gas", "raw_tx", "broadcast"):
        assert banned not in surface


def test_release_is_not_a_chain_door() -> None:
    """It is a key oracle with a refusal vocabulary of its own; it lives in the escrow seam."""
    assert not hasattr(NodeClient, "release")
    paths = {
        line.strip()
        for name, member in inspect.getmembers(NodeClient)
        if callable(member) and not name.startswith("__")
        for line in (inspect.getsource(member) if inspect.isfunction(member) else "").splitlines()
        if "/release" in line
    }
    assert paths == set()


def test_the_shared_timeout_budgets_its_body_legs_for_the_largest_upload() -> None:
    """One static budget on the session, not one computed per result.

    `read` and `write` carry a body and are sized once for the largest thing the
    daemon sends — the files door stores up to 200 MiB and answers only once the
    object store has taken it. `connect` and `pool` stay short, so a coordinator
    that is simply unreachable is still refused in seconds rather than blocking a
    settle well inside the job's SLA window.
    """
    assert HTTP_TIMEOUT.read == HTTP_TIMEOUT.write
    assert HTTP_TIMEOUT.read is not None and HTTP_TIMEOUT.read >= 900.0
    assert HTTP_TIMEOUT.connect == 10.0
    assert HTTP_TIMEOUT.pool == 10.0


def test_the_inline_threshold_fills_the_body_exactly() -> None:
    """The invariant the derivation's docstrings advertise, actually checked.

    ``base64_length`` was defined here and referenced by nothing, so the exact-fit
    property was prose. A result at the threshold has to encode to precisely the
    room the envelope reserve leaves it, or the daemon inlines a settle the op
    door answers ``413`` to.
    """
    assert base64_length(INLINE_MAX_BYTES) + ENVELOPE_RESERVE_BYTES == MAX_BODY_BYTES
    assert base64_length(INLINE_MAX_BYTES + 1) + ENVELOPE_RESERVE_BYTES > MAX_BODY_BYTES
    # Counted the way the encoder counts, not approximated.
    for n in (0, 1, 2, 3, 61, 1024):
        assert base64_length(n) == len(base64.b64encode(b"\0" * n))
