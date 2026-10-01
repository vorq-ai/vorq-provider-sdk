"""Content-addressed task fetch: the source is untrusted, the commitment decides.

An order names its container with a CID minted by the service that pinned it.
The daemon treats that name as an opaque locator, not an authorization — anyone
can pin anything under any name, so the bytes a fetch returns prove nothing about
*whose* task they are. What makes a fetch trustworthy is the job's own name::

    job_id == keccak256( owner ‖ c )      c = keccak256(version ‖ seed_wrap ‖ keccak256(ciphertext))

That commitment binds the bytes to the wallet that signed and paid for the order,
and it is computed in :mod:`vorqd.container` — one definition of the format, used
by the fetch here and by the unseal that follows it. :class:`BlobResolver` walks
its sources in availability preference order and returns the first body that
satisfies the commitment, so a source can only ever fail to answer — never
substitute.

Bytes come from the storage network's public gateway, by name, with no
credentials and no coordinator in the path.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Protocol
from urllib.parse import quote

import httpx

from .container import ContainerError, commitment, content_job_id

log = logging.getLogger("vorqd")

class BlobError(Exception):
    """No source produced bytes that satisfy the job's content commitment."""


class BlobSource(Protocol):
    """Anything that can answer "the bytes under this CID", honestly or not."""

    async def get(self, cid: str) -> bytes: ...


class MemoryBlobSource:
    """In-process source — same contract, no network."""

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self._blobs: dict[str, bytes] = dict(blobs or {})

    async def get(self, cid: str) -> bytes:
        try:
            return self._blobs[cid]
        except KeyError:
            raise BlobError(f"no blob under {cid}") from None

    def put(self, raw: bytes, *, cid: str) -> str:
        """Pin ``raw`` under ``cid``; returns ``cid``. The name is the caller's to
        supply — the storage service mints it in production — so a lying source is
        simply a wrong ``cid``."""
        self._blobs[cid] = raw
        return cid


#: The read gateway the daemon ships with — the storage network's public
#: gateway, so a stock daemon resolves any CID with zero configuration.
#: Overridden by ``$VORQ_PIN_GATEWAY``; set it to your own gateway or your own
#: node. There is no other read path: the coordinator serves no blob endpoint.
DEFAULT_GATEWAY = "https://ipfs.filebase.io"

#: Retries sized for pin propagation, not for a local surface. A task is fetched
#: seconds after its bytes were pinned and a fresh name can take a few seconds to
#: become resolvable; the claim is already held and the window is hours, so
#: patience only delays the truly-missing case.
GATEWAY_ATTEMPTS = 8
GATEWAY_BACKOFF_S = 1.5


class GatewayBlobSource:
    """A storage-network read gateway, path mode (``GET {gateway}/ipfs/{cid}``).

    Trust is unchanged by which gateway answers: it is as untrusted as any
    source, and the commitment check decides.
    """

    def __init__(self, client: httpx.AsyncClient, gateway: str) -> None:
        self._client = client
        self._gateway = gateway.rstrip("/")

    async def get(self, cid: str) -> bytes:
        resp = await self._client.get(
            f"{self._gateway}/ipfs/{quote(cid, safe='')}", follow_redirects=True
        )
        if resp.status_code == 404:
            raise BlobError(f"no blob under {cid}")
        resp.raise_for_status()
        return resp.content


def gateway_url() -> str:
    """The gateway this process reads from: ``$VORQ_PIN_GATEWAY`` or the default.

    Empty means *no gateway is configured* — the operator supplies their own URL
    or runs their own node. It is not a fallback onto some other surface, because
    there is no other surface.
    """
    configured = os.environ.get("VORQ_PIN_GATEWAY")
    return DEFAULT_GATEWAY if configured is None else configured.strip()


def default_resolver(client: httpx.AsyncClient) -> "BlobResolver":
    """The daemon's blob resolution, defined once.

    Every caller that needs task bytes builds its resolver here — the CLI wiring
    and the scheduler's own fallback alike — so there is one way to choose a
    gateway and not two that can drift apart.
    """
    gateway = gateway_url()
    if not gateway:
        log.warning("no storage gateway configured ($VORQ_PIN_GATEWAY is empty); "
                    "no source can serve task bytes")
        return BlobResolver([])
    return BlobResolver([GatewayBlobSource(client, gateway)],
                        attempts=GATEWAY_ATTEMPTS, backoff_s=GATEWAY_BACKOFF_S)


class BlobResolver:
    """Ordered-source task fetch: the first body that satisfies the commitment wins.

    A claimed job cannot be re-run by anyone else, so giving up on the first
    unlucky 404 or timeout would cancel work that a second look would have found.
    Each fetch therefore sweeps every source, then retries the whole sweep a
    bounded number of times with a short linear backoff before conceding.
    """

    def __init__(self, sources: list[BlobSource], *, attempts: int = 3, backoff_s: float = 0.2) -> None:
        self._sources = sources
        self._attempts = max(1, attempts)
        self._backoff_s = backoff_s

    async def fetch_task(self, job) -> bytes:
        cid = getattr(job, "task_cid", None)
        if not cid:
            raise BlobError(f"job {job.job_id} names no task_cid")
        # THE RETRY BUDGET IS FOR AVAILABILITY, NEVER FOR TRUST. A source that
        # could not answer may answer on the next sweep — a fresh pin takes a few
        # seconds to become resolvable, which is what the backoff is sized for. A
        # source that answered with bytes failing the commitment will answer with
        # the same bytes forever: the name is content-addressed, so re-asking is
        # not patience, it is a stall. Such a source is dropped for the rest of
        # this fetch, and when every source has been dropped there is nothing left
        # to wait for and the sweep ends immediately rather than sleeping out the
        # remaining attempts.
        live = list(self._sources)
        for attempt in range(self._attempts):
            if not live:
                break
            if attempt:
                await asyncio.sleep(self._backoff_s * attempt)
            for source in list(live):
                try:
                    raw = await source.get(cid)
                except (BlobError, httpx.HTTPError) as exc:
                    log.debug("blob source could not serve %s: %s", cid, exc)
                    continue   # unreachable source: availability, not trust
                if self._committed(job, raw):
                    return raw
                # Right name, wrong bytes: the source is lying, the pin was
                # replaced, or the wrap was lifted from another order. Neither is
                # our problem — move on and let another source answer.
                live.remove(source)
                log.warning("bytes under %s fail the job's commitment; source skipped", cid)
        raise BlobError(f"no source served bytes satisfying the commitment for {cid}")

    @staticmethod
    def _committed(job, raw: bytes) -> bool:
        """Do these bytes carry this job's name? A malformed container, owner or
        job id is a failed check, never an exception — the caller is mid-sweep
        over sources and must not have one bad answer abort it."""
        try:
            return content_job_id(job.owner or "", commitment(raw)) == str(job.job_id).lower()
        except ContainerError as exc:
            log.warning("bytes under %s are not a container (%s)", job.task_cid, exc.fault)
            return False
        except (ValueError, AttributeError, TypeError) as exc:
            log.warning("job %s has an unusable owner for the commitment check: %s", job.job_id, exc)
            return False
