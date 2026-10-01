"""The coordinator node seam: the daemon's one door onto the chain.

A coordinator node reads the chain and relays what this daemon signs, paying the
gas itself. So this client carries **no RPC concepts at all** — no transaction
nonce, no gas price, no raw bytes, no broadcast. It hands over an op name, a
payload and a signature; the node builds the transaction, simulates it, funds it
and answers with what happened. Everything an operator would need a wallet
balance for is the node's problem, and keeping it that way is the whole reason
this surface is narrow.

**``POST /release`` is deliberately not here.** It is a key oracle, not a chain
door: it answers a refusal vocabulary of its own, it never touches a
transaction, and putting it on this object is how the two mechanisms got
conflated once already. It belongs to the escrow seam.

The session is a **transport gate and nothing else**. It is attached to the two
doors the node gates — ``POST /evm/ops`` and ``PUT /evm/asks`` — and to nothing
else, because nothing else is gated: the job book, the provider directory, the
catalog, the allowlist, the chain context and the advisory claim simulate are
all public reads. Authority over an op is the op's own signature, which the node
recovers and the contract re-checks; a session never stands in for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

from ._crypto import Signer
from .config import VorqdConfig
from .coordinator import CoordinatorClient
from .errors import (
    CatalogUnresolved,
    ChainConflict,
    OpRefused,
    OpRejected,
    UnknownModel,
    UploadInvalid,
)
from .limits import window_seconds
from .money import format_usd, parse_usd
from .types import ChainContext, EvmJob

#: ``JobState`` from the contracts' ``Types.sol``, in its declared order.
JOB_STATES = ("Open", "Claimed", "Settled", "Cancelled")

#: ``EndedBecause`` from the same file. ``0`` is "not ended"; ``1`` and ``5`` are
#: view-only (settlement has its own event, and expiry is computed rather than
#: stored).
ENDED_BECAUSE = {
    0: None,
    1: "settled",
    2: "cancelled",
    3: "provider_fail",
    4: "reclaim",
    5: "expired",
}

#: An SLA window as an operator writes it in ``vorqd.yaml``: a count and a unit.
#: Strict on purpose — the ask book and the order both carry ``slaSecs`` as a
#: ``uint32``, so a window this cannot read is a price the daemon would publish
#: against the wrong deadline. The same spelling serves every window in the
#: config, so the parser lives with the other pure policy in ``limits``.
def sla_window_seconds(window: str) -> int:
    """``"24h"`` → ``86400``. Raises :class:`ValueError` on anything else."""
    try:
        return window_seconds(window)
    except ValueError as exc:
        raise ValueError(f"sla {exc}") from None


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OpResult:
    """What ``POST /evm/ops`` answers for a relayed op.

    ``result_cid`` is set only for a settle, and it is the **only** way a
    claimant ever learns the name of what it delivered: the node mints it from
    its own pin of the bytes, and nothing the daemon holds could predict it.
    """

    tx_hash: str
    status: str
    block_number: int | None
    result_cid: str | None = None

    @classmethod
    def from_wire(cls, d: dict) -> "OpResult":
        block = d.get("block_number")
        return cls(
            tx_hash=d["tx_hash"],
            status=str(d.get("status", "")),
            block_number=None if block is None else int(block),
            result_cid=d.get("result_cid"),
        )


@dataclass(frozen=True)
class ClaimSimulation:
    """The advisory gate's answer, typed.

    Advisory by construction, and it says so by what it omits: the escrow pull,
    the op signature and the freshness window are not evaluated. ``ok`` false
    with ``reason`` ``"AtCapacity"`` / ``"NotOpen"`` / ``"NotDesignated"`` is a
    reason to not sign; ``ok`` true is not a promise the claim will land.
    """

    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class CatalogModel:
    """One curated catalog entry: the name an operator writes, and its chain id.

    ``modality`` is read when the catalog names it and is ``None`` otherwise —
    which is what a coordinator node answers today, since the on-chain model
    record is an id, a name and an enabled flag and nothing else. It is carried
    rather than assumed because modality decides whether a job is metered by
    tokens or by pixels, and a guess of ``"text"`` would settle a media job at
    the full cap. When the catalog is silent the operator's own ``modality:``
    declaration is the only source, and a model neither names is one the daemon
    refuses to claim for.
    """

    model_id: int
    name: str
    enabled: bool
    modality: str | None = None


class ModelResolver:
    """Config model names and SLA windows ↔ the ``uint32`` ids the chain uses.

    The daemon's config addresses a model by name (``org/model:fp8``) and an SLA
    by window (``"24h"``); the ops, the order and the ask book address both by
    number. This resolves the one into the other, **once, at startup**, off the
    coordinator's catalog.

    A configured model the catalog does not carry raises :class:`UnknownModel`
    and the daemon does not start. That is deliberate and it is the whole
    reason this is resolved eagerly: skipping such a model would leave the
    operator running silently at a fraction of the capacity they configured,
    with no error anywhere and asks published for work that can never arrive.

    The reverse SLA lookup is **per model**, not global, because it has to
    return the operator's own spelling: two models may write ``"24h"`` and
    ``"1d"`` for the same 86 400 seconds, and the scheduler prices a job by
    looking its window up in that model's own ``slas`` mapping.
    """

    def __init__(
        self,
        ids: dict[str, int],
        names: dict[int, str],
        windows: dict[str, dict[int, str]],
        enabled: dict[str, bool],
    ) -> None:
        self._ids = ids
        self._names = names
        self._windows = windows
        self._enabled = enabled

    @classmethod
    def resolve(cls, config: VorqdConfig, catalog: Iterable[CatalogModel]) -> "ModelResolver":
        """Bind every configured model and window, or raise.

        ``catalog`` is what :meth:`NodeClient.get_models` answered.
        """
        by_name = {entry.name: entry for entry in catalog}
        ids: dict[str, int] = {}
        names: dict[int, str] = {}
        windows: dict[str, dict[int, str]] = {}
        enabled: dict[str, bool] = {}

        for model in config.models:
            entry = by_name.get(model.model)
            if entry is None:
                raise UnknownModel(
                    f"model {model.model!r} is configured but the coordinator's catalog does not "
                    f"carry it, so it has no model id and cannot be claimed, quoted or served"
                )
            ids[model.model] = entry.model_id
            names[entry.model_id] = entry.name
            enabled[model.model] = entry.enabled
            reverse: dict[int, str] = {}
            for window in model.slas:
                try:
                    secs = sla_window_seconds(window)
                except ValueError as exc:
                    raise UnknownModel(f"model {model.model!r}: {exc}") from None
                reverse.setdefault(secs, window)
            windows[model.model] = reverse

        return cls(ids, names, windows, enabled)

    def model_id(self, name: str) -> int:
        try:
            return self._ids[name]
        except KeyError:
            raise UnknownModel(f"model {name!r} has no resolved model id") from None

    def model_name(self, model_id: int) -> str:
        try:
            return self._names[int(model_id)]
        except KeyError:
            raise UnknownModel(f"model id {model_id} is not one this daemon serves") from None

    def is_enabled(self, name: str) -> bool:
        return self._enabled.get(name, False)

    @property
    def model_ids(self) -> dict[str, int]:
        """Every configured model's id, by name — the scheduler's poll list."""
        return dict(self._ids)

    def sla_secs(self, window: str) -> int:
        """A config window as the ``uint32`` the chain carries."""
        return sla_window_seconds(window)

    def sla_window(self, model: str, secs: int) -> str:
        """A job's ``sla_secs`` back to **this model's** own window spelling."""
        try:
            return self._windows[model][int(secs)]
        except KeyError:
            raise UnknownModel(
                f"model {model!r} quotes no sla window of {secs} s, so a job at that deadline "
                f"is not one this daemon priced"
            ) from None


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


class NodeClient:
    """Everything the daemon asks a coordinator node for.

    ``models`` is the catalog resolution (:class:`ModelResolver`). It is not a
    constructor requirement because it cannot be: it is built from
    :meth:`get_models`, which needs this client. Bind it with
    :meth:`bind_models` once the catalog is read; the two job listings need it
    to name a job's model and window, and raise :class:`CatalogUnresolved`
    without it.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        api_url: str,
        signer: Signer,
        *,
        session: CoordinatorClient | None = None,
        models: ModelResolver | None = None,
    ) -> None:
        self._http = http
        self._api_url = api_url.rstrip("/")
        # One session per wallet, shared if the caller already has one: a second
        # handshake for the same key would mint a second token for no reason.
        self._session = session if session is not None else CoordinatorClient(http, self._api_url, signer)
        self._models = models
        self._chain: ChainContext | None = None
        # Served by `GET /evm/chain` and deliberately not a `ChainContext`
        # field: the daemon signs nothing that mentions a block, and the context
        # is what op signatures are made against. It is kept here because the
        # age filter has to spell seconds as blocks. `None` until that door is
        # read, and `None` afterwards if the node serves no block time.
        self._block_time_ms: int | None = None
        # The block the last job listing was answered at, off its envelope. A
        # job row carries `posted_block`, so this is the other half of a bid's
        # age — and 0 means "no listing read yet", which is no reference at all.
        self._as_of_block = 0

    # -- wiring ------------------------------------------------------------ #

    @property
    def session(self) -> CoordinatorClient:
        return self._session

    @property
    def provider_id(self) -> int | None:
        """This wallet's provider id, learned from ``POST /auth/session``."""
        return self._session.provider_id

    @property
    def address(self) -> str:
        return self._session.address

    @property
    def models(self) -> ModelResolver | None:
        return self._models

    def bind_models(self, resolver: ModelResolver) -> None:
        self._models = resolver

    @property
    def as_of_block(self) -> int:
        """The block the most recent job listing was answered at, or ``0``.

        Every ``/evm/jobs`` response carries it on the **envelope** — it
        describes the answer, not any row — and it is the reference a bid's age
        is measured against. Replaced on each listing rather than latched once:
        a reference from boot would age every bid by this daemon's uptime.
        """
        return self._as_of_block

    # -- transport --------------------------------------------------------- #

    async def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self._session.token()}"}

    async def _authed(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """A session-gated request, re-handshaking **once** on a 401.

        A 401 means the session was revoked out from under us — wallet rotation,
        node restart — so the cached token is stale though not yet expired. Drop
        it and retry once; a second 401 propagates.
        """
        url = f"{self._api_url}{path}"
        resp = await self._http.request(method, url, headers=await self._headers(), **kwargs)
        if resp.status_code == 401:
            self._session.invalidate()
            resp = await self._http.request(method, url, headers=await self._headers(), **kwargs)
        return resp

    async def _read(self, path: str, **kwargs: Any) -> dict:
        """A public read. No session: none of these doors is gated."""
        resp = await self._http.get(f"{self._api_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()

    # -- chain context ----------------------------------------------------- #

    async def chain_context(self) -> ChainContext:
        """The chain id and the five contract addresses, read once and cached.

        Cached because it cannot change under a running daemon: a node pointed
        at a different deployment is a different node, and the addresses are
        read from its own configuration rather than from chain state. Every op
        signature depends on two of these, so re-reading them per op would put
        a network round trip in front of every signature to learn a constant.
        """
        if self._chain is None:
            body = await self._read("/evm/chain")
            self._chain = ChainContext.from_wire(body)
            self._block_time_ms = _opt_int(body.get("block_time_ms"))
        return self._chain

    async def block_time_ms(self) -> int | None:
        """The chain's block time, or ``None`` from a node that serves none.

        Read on the same cached ``GET /evm/chain`` the op signatures already
        need, so asking for it costs no request of its own. ``None`` is not a
        default to fall back on: a caller that cannot convert seconds to blocks
        must send no age filter rather than guess one.
        """
        await self.chain_context()
        return self._block_time_ms

    # -- the op door ------------------------------------------------------- #

    async def push_op(self, op: str, payload: dict, signature: str) -> OpResult:
        """Relay one signed op. ``payload`` is the op's flat fields.

        The body is ``{op, …fields, signature}``, sent as JSON for every op. A
        settle payload carries either ``result`` (base64 of the sealed bytes, at
        or under :data:`~vorqd.types.INLINE_MAX_BYTES`) or ``result_cid`` (from
        :meth:`upload_file`) — never neither.

        The node pins those bytes (from the body or from the upload it already
        holds), mints the name, puts it in the transaction and answers with it —
        the daemon holds no pinning credential and could not compute the name if
        it wanted to, which is why no CID is in the signature.

        ``403`` raises :class:`OpRejected`; ``409`` raises :class:`OpRefused`
        carrying the contract's own decoded error name. Everything else — the
        node's retryable ``429``/``503``, its ``504`` — propagates as an
        ``httpx`` status error, because those are the node's failure to relay
        rather than the chain's verdict on the op, and collapsing the two would
        tell a daemon to stop retrying something a retry would fix.
        """
        fields = {"op": op, **payload, "signature": signature}
        if op == "settle" and "result" not in fields and "result_cid" not in fields:
            raise ValueError("a settle must carry either the sealed result (base64) or a result_cid")
        resp = await self._authed("POST", "/evm/ops", json=fields)

        if resp.status_code == 403:
            raise OpRejected(_message(resp))
        if resp.status_code == 409:
            refusal = resp.json()
            raise OpRefused(str(refusal.get("reason", "unknown")), raw=refusal.get("raw"))
        resp.raise_for_status()
        return OpResult.from_wire(resp.json())

    async def upload_file(self, purpose: str, content: bytes, filename: str = "result") -> str:
        """Upload one blob through ``POST /v1/files`` and return its cid.

        httpx renders ``data`` fields before ``files`` parts, so the node reads
        ``purpose`` before it reads a byte of the file.

        An answer with no ``vorq.cid`` raises :class:`~vorqd.errors.UploadInvalid`
        rather than ``KeyError``/``TypeError``: the caller is the settle path,
        whose handlers catch the failures it knows about, and an untyped one
        escapes all of them and leaves the job claimed with no fail report.
        """
        # No deadline of its own: `HTTP_TIMEOUT`'s body legs are already sized
        # for the largest upload this door takes. What the door will *store* is
        # the door's business — the daemon keeps no copy of `MAX_BLOB_BYTES`.
        resp = await self._authed(
            "POST", "/v1/files",
            data={"purpose": purpose},
            files={"file": (filename, content, "application/octet-stream")},
        )
        resp.raise_for_status()
        body = resp.json()
        cid = (body.get("vorq") or {}).get("cid") if isinstance(body, dict) else None
        if not isinstance(cid, str) or not cid:
            raise UploadInvalid(
                f"POST /v1/files answered no vorq.cid for purpose {purpose!r}, so there is "
                "no name to reference these bytes by"
            )
        return cid

    async def simulate_claim(self, job_id: str, address: str) -> ClaimSimulation:
        """Ask, before signing anything, whether a claim would land.

        Chain reads only, never the index, which is what makes the answer
        meaningful during the finality lag — precisely the window in which a
        daemon is deciding whether to sign.
        """
        resp = await self._http.post(
            f"{self._api_url}/evm/simulate/claim", json={"job_id": job_id, "address": address}
        )
        resp.raise_for_status()
        answer = resp.json()
        ok = bool(answer.get("ok"))
        return ClaimSimulation(ok=ok, reason=None if ok else str(answer.get("reason", "unknown")))

    # -- reads ------------------------------------------------------------- #

    async def list_open_jobs(
        self,
        model_id: int,
        *,
        free: int | None = None,
        min_rate_in: int | None = None,
        min_rate_out: int | None = None,
    ) -> list[EvmJob]:
        """The open book for one model, narrowed to what this daemon can act on.

        With ``free`` this is the sweep's poll: ``GET /evm/jobs?state=Open&model=…&free=N``
        under the session. The coordinator leases this provider the oldest open
        rows that clear the filters — at most ``free``, the slots the daemon can
        start now, counting what it already holds — and answers only those.
        Nobody else is handed them until the lease lapses. The same call is the
        daemon's heartbeat: ``free`` goes on record as its presence, and a
        daemon that stops polling stops being named on a challenge. The node
        resolves the caller from the token, so no provider id is ever sent.

        Without ``free`` it is the public book, as anyone reads it.

        ``model_id``, never a model name. The filters are an **optimisation, not
        a guarantee**: a lease is short (about 20 s) and advisory, a row claimed
        by somebody else anyway is refused at the chain like any lost race, and
        :meth:`~vorqd.scheduler.Scheduler.profitable` remains the boundary on
        every row that comes back.

        The rate floors are atomic here and cross as USD decimal strings, like
        all money on the node's API. An omitted filter is **absent**, never
        zero: a ``min_rate_in`` of 0 is a floor the daemon never set, and it
        would read as one.
        """
        decimals = (await self.chain_context()).decimals
        params: dict[str, str] = {"state": "Open", "model": str(int(model_id))}
        if free is not None:
            params["free"] = str(int(free))
        if min_rate_out is not None:
            params["min_rate_out"] = format_usd(min_rate_out, decimals)
        if min_rate_in is not None:
            params["min_rate_in"] = format_usd(min_rate_in, decimals)
        if free is None:
            return self._jobs(await self._read("/evm/jobs", params=params), decimals)
        resp = await self._authed("GET", "/evm/jobs", params=params)
        resp.raise_for_status()
        return self._jobs(resp.json(), decimals)

    async def list_claimed_jobs(self, provider_id: int) -> list[EvmJob]:
        """Jobs this provider claimed and has not settled — the boot-recovery read.

        Paged to exhaustion: boot recovery deletes every recorded handle whose
        job is not in this list, so a partial read would forget live work. The
        node serves 100 rows by default and 1000 at most, and it signals a cut
        page in the **headers** rather than in the body, so both signals are
        read — a full page, and the truncation header.
        """
        decimals = (await self.chain_context()).decimals
        limit = 1000
        offset = 0
        jobs: list[EvmJob] = []
        while True:
            resp = await self._http.get(
                f"{self._api_url}/evm/jobs",
                params={"state": "Claimed", "provider": str(int(provider_id)),
                        "limit": str(limit), "offset": str(offset)},
            )
            resp.raise_for_status()
            rows = self._jobs(resp.json(), decimals)
            jobs.extend(rows)
            truncated = resp.headers.get("x-vorq-page-truncated", "").lower() == "true"
            # An empty page ends the read whatever the header says: a node that
            # keeps answering "truncated" with nothing in it would loop forever.
            if not rows or (len(rows) < limit and not truncated):
                return jobs
            offset += len(rows)

    async def get_provider(self, provider_id: int) -> dict:
        """One provider row: its box key, evidence, listing, capacity and load."""
        return await self._read(f"/evm/providers/{int(provider_id)}")

    async def get_allowlist(self) -> dict:
        """The measurement allowlist, served unsigned — the chain read is the root of trust."""
        return await self._read("/evm/allowlist")

    async def get_models(self) -> list[CatalogModel]:
        """The curated catalog: every model's name and its ``uint32`` id."""
        body = await self._read("/evm/models")
        out: list[CatalogModel] = []
        for entry in body.get("data", []):
            vorq = entry.get("vorq") or {}
            if "model_id" not in vorq:
                continue
            modality = vorq.get("modality")
            out.append(
                CatalogModel(
                    model_id=int(vorq["model_id"]),
                    name=str(entry["id"]),
                    enabled=bool(vorq.get("enabled", True)),
                    modality=str(modality) if modality else None,
                )
            )
        return out

    async def push_asks(self, snapshot: dict, signature: str) -> dict:
        """Publish a signed ask snapshot. The node lands it on chain and pays.

        The snapshot's signature is the only authority — the node has no key
        that could author a price — and it is made over the **AskRegistry's**
        domain, which is a third address again.
        """
        resp = await self._authed(
            "PUT", "/evm/asks", json={"snapshot": snapshot, "signature": signature}
        )
        if resp.status_code == 403:
            raise OpRejected(_message(resp))
        if resp.status_code == 409:
            err = resp.json().get("error", {})
            raise ChainConflict(
                err.get("type", "state_conflict"), err.get("message", ""), code=err.get("code")
            )
        resp.raise_for_status()
        return resp.json()

    # -- job translation --------------------------------------------------- #

    def _jobs(self, body: dict, decimals: int) -> list[EvmJob]:
        # The envelope's own field, kept before the rows are read. Absent is not
        # zero: a body carrying no `as_of_block` leaves the last reference
        # standing, because forgetting it would quietly turn an age filter off.
        as_of = _opt_int(body.get("as_of_block"))
        if as_of is not None:
            self._as_of_block = as_of
        resolver = self._models
        if resolver is None:
            raise CatalogUnresolved(
                "the model catalog has not been resolved, so a job's model_id and sla_secs "
                "cannot be named; call get_models() and bind_models() at startup"
            )
        return [job_from_wire(row, resolver, decimals) for row in body.get("jobs", [])]


def job_from_wire(row: dict, resolver: ModelResolver, decimals: int) -> EvmJob:
    """One chain-shaped job row as the daemon's :class:`~vorqd.types.EvmJob`.

    Rates arrive as USD decimal strings and become atomic at ``decimals``, so
    the floor comparison is exact integer arithmetic. Every other integer
    arrives as a JSON integer. ``state`` and ``ended_because`` are named from the
    contracts' own enums.

    ``designated`` is passed through exactly as the chain holds it, which means
    **``0`` for an open job and never a null** — the sentinel the contracts use
    (``Order.designated``: "0 = open order").
    """
    model = resolver.model_name(row["model_id"])
    claimed_at = int(row.get("claimed_at") or 0)
    return EvmJob(
        job_id=row["job_id"],
        model=model,
        state=JOB_STATES[int(row["state"])],
        sla=resolver.sla_window(model, int(row["sla_secs"])),
        # The book answers `posted_block`, not a wall-clock post time, and
        # nothing in the daemon reads this field. It is not invented from a
        # block number, which would be a timestamp that is not one.
        created_at=0,
        owner=row.get("owner"),
        provider=_opt_int(row.get("provider_id")),
        rate_in=_opt_usd(row.get("rate_in"), decimals),
        rate_out=_opt_usd(row.get("rate_out"), decimals),
        expires_at=_opt_int(row.get("expires_at")),
        # 0 means "not claimed", and the SLA deadline check reads this: a zero
        # would put every unclaimed job's deadline in 1970.
        claimed_at=claimed_at or None,
        ended_because=ENDED_BECAUSE.get(int(row.get("ended_because") or 0)),
        task_cid=row.get("task_cid"),
        result_cid=row.get("result_cid"),
        units_in=_opt_int(row.get("units_in")),
        units_out=_opt_int(row.get("units_out")),
        completion_tokens=_opt_int(row.get("completion_tok")),
        designated=int(row.get("designated") or 0),
    )


def _opt_int(value: str | int | None) -> int | None:
    return None if value is None else int(value)


def _opt_usd(value: str | None, decimals: int) -> int | None:
    return None if value is None else parse_usd(value, decimals)


def _message(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error", {}).get("message", "")) or resp.text
    except ValueError:  # pragma: no cover — a 403 without a JSON body
        return resp.text


__all__: Sequence[str] = (
    "JOB_STATES",
    "ENDED_BECAUSE",
    "CatalogModel",
    "ClaimSimulation",
    "ModelResolver",
    "NodeClient",
    "OpResult",
    "job_from_wire",
    "sla_window_seconds",
)
