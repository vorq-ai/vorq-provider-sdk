"""The blob seam: content addressing means the source is untrusted by design."""

from __future__ import annotations

import pytest

from .conftest import fake_cid, job_id_of, seal_container
from vorqd._crypto import BoxCipher
from vorqd.blob import (
    DEFAULT_GATEWAY,
    BlobError,
    BlobResolver,
    GatewayBlobSource,
    MemoryBlobSource,
    default_resolver,
    gateway_url,
)

OWNER = "0x" + "11" * 20
RECIPIENT = BoxCipher.generate().public_key
REAL = seal_container(b'{"v":"vorq-env-v1","owner":"0x11","input":{}}', recipient=RECIPIENT, owner=OWNER)


class _Job:
    def __init__(self, job_id: str, owner: str, task_cid: str | None):
        self.job_id, self.owner, self.task_cid = job_id, owner, task_cid


_UNSET = object()


def _job(container: bytes = REAL, *, owner: str = OWNER, task_cid=_UNSET) -> _Job:
    cid = fake_cid(container) if task_cid is _UNSET else task_cid
    return _Job(job_id_of(owner, container), owner, cid)


async def test_fetch_task_returns_bytes_that_satisfy_the_commitment():
    job = _job()
    source = MemoryBlobSource({job.task_cid: REAL})
    assert await BlobResolver([source]).fetch_task(job) == REAL


async def test_fetch_task_rejects_bytes_that_fail_the_commitment():
    # A CID match alone proves nothing: the source names these bytes correctly for
    # itself, but they are not the bytes this job's id commits to.
    job = _job()
    other = seal_container(b"somebody else's task", recipient=RECIPIENT, owner=OWNER)
    source = MemoryBlobSource({job.task_cid: other})
    with pytest.raises(BlobError, match="commitment"):
        await BlobResolver([source]).fetch_task(job)


async def test_bytes_that_are_not_a_container_at_all_are_refused():
    job = _job()
    # The last one is a container whose version byte names a layout no build reads
    # — ONE byte, not five. The old `b"XXXX\x01" + REAL[5:]` kept passing after the
    # tag shrank because it stripped four bytes of the wrap instead, so it failed
    # the commitment for a reason that had nothing to do with the version.
    for junk in (b"", b"not a container", REAL[:80], b"\x02" + REAL[1:]):
        with pytest.raises(BlobError, match="commitment"):
            await BlobResolver([MemoryBlobSource({job.task_cid: junk})], attempts=1).fetch_task(job)


async def test_fetch_task_tries_sources_in_order_and_skips_the_liar():
    job = _job()
    empty = MemoryBlobSource()
    liar = MemoryBlobSource({job.task_cid: seal_container(b"different", recipient=RECIPIENT, owner=OWNER)})
    honest = MemoryBlobSource({job.task_cid: REAL})
    assert await BlobResolver([empty, liar, honest]).fetch_task(job) == REAL
    with pytest.raises(BlobError, match="no source"):
        await BlobResolver([empty, liar]).fetch_task(job)


async def test_fetch_task_without_a_task_cid_is_an_error():
    with pytest.raises(BlobError, match="task_cid"):
        await BlobResolver([MemoryBlobSource()]).fetch_task(_job(task_cid=None))


async def test_gateway_source_reads_the_storage_network_without_a_bearer():
    # The read path: the coordinator is not consulted at all, and the gateway
    # needs no credentials — the CID is the whole request.
    import httpx

    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        if req.url.path.endswith("/missing"):
            return httpx.Response(404)
        return httpx.Response(200, content=REAL)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = GatewayBlobSource(client, "https://gw.test/")   # trailing slash normalized
    assert await source.get("QmOpaqueName") == REAL
    assert seen == {"url": "https://gw.test/ipfs/QmOpaqueName", "auth": None}
    with pytest.raises(BlobError):
        await source.get("missing")
    await client.aclose()


def test_the_gateway_is_chosen_in_exactly_one_place(monkeypatch):
    # `default_resolver` is the single resolution every caller shares — the CLI
    # wiring and the scheduler's own fallback alike.
    import httpx

    client = httpx.AsyncClient()
    monkeypatch.delenv("VORQ_PIN_GATEWAY", raising=False)
    assert gateway_url() == DEFAULT_GATEWAY
    (source,) = default_resolver(client)._sources
    assert isinstance(source, GatewayBlobSource) and source._gateway == DEFAULT_GATEWAY

    monkeypatch.setenv("VORQ_PIN_GATEWAY", " https://gw.test/ ")
    assert gateway_url() == "https://gw.test/"
    (source,) = default_resolver(client)._sources
    assert source._gateway == "https://gw.test"

    # Empty is "no gateway configured", not a fallback onto another surface:
    # there is no other surface, so nothing can serve.
    monkeypatch.setenv("VORQ_PIN_GATEWAY", "")
    assert default_resolver(client)._sources == []


def test_the_shipped_gateway_is_the_one_that_serves_our_pins():
    # Pinned so that changing what a stock daemon reads from is a deliberate edit,
    # and it must stay identical to the client SDK's default: a client writing to
    # one gateway and a provider reading from another is an interop break no test
    # in either repo would catch.
    #
    # A CID is content-addressed but retrievability is not: objects pinned through
    # the project's pinning service are served promptly by that service's gateway,
    # while an arbitrary public gateway serves them only once the content has
    # propagated — which is neither guaranteed nor prompt. The URL is a
    # configuration value, not a name; the names around it stay vendor-neutral.
    assert DEFAULT_GATEWAY == "https://ipfs.filebase.io"


async def test_an_unusable_owner_is_a_failed_check_not_a_raised_ValueError():
    # A job whose owner is not hex cannot satisfy any commitment — but the
    # derivation must not throw past the caller's `except BlobError` and abort the
    # whole sweep with the job orphaned in Claimed.
    for owner in ("client", "0xnothex", "0x123", None):
        job = _Job(job_id_of(OWNER, REAL), owner, fake_cid(REAL))
        with pytest.raises(BlobError, match="no source"):
            await BlobResolver([MemoryBlobSource({job.task_cid: REAL})],
                               attempts=1).fetch_task(job)


async def test_fetch_task_retries_the_sweep_before_conceding():
    # A claimed job cannot be re-run by anyone else, so one unlucky 404 must not
    # be terminal: the sweep is retried a bounded number of times.
    job = _job()
    calls = {"n": 0}

    class FlakySource:
        async def get(self, cid):
            calls["n"] += 1
            if calls["n"] < 3:
                raise BlobError("not pinned yet")
            return REAL

    assert await BlobResolver([FlakySource()], attempts=3, backoff_s=0).fetch_task(job) == REAL
    assert calls["n"] == 3


async def test_retries_are_bounded_and_then_it_gives_up():
    job = _job()
    calls = {"n": 0}

    class DeadSource:
        async def get(self, cid):
            calls["n"] += 1
            raise BlobError("gone")

    with pytest.raises(BlobError, match="no source"):
        await BlobResolver([DeadSource()], attempts=3, backoff_s=0).fetch_task(job)
    assert calls["n"] == 3   # three sweeps, not an unbounded loop


async def test_a_missing_task_cid_does_not_burn_the_retry_budget():
    calls = {"n": 0}

    class CountingSource:
        async def get(self, cid):  # pragma: no cover - must never be reached
            calls["n"] += 1
            raise BlobError("x")

    with pytest.raises(BlobError, match="task_cid"):
        await BlobResolver([CountingSource()], attempts=3, backoff_s=0).fetch_task(_job(task_cid=None))
    assert calls["n"] == 0


async def test_a_lying_source_does_not_burn_the_retry_budget():
    # The budget is for availability, never for trust. A source that answered
    # once with bytes that miss the commitment will answer with the same bytes
    # forever — the name is content-addressed — so re-asking it is not patience,
    # it is a stall. Asked once, then dropped for the rest of the fetch.
    job = _job()
    calls = {"n": 0}
    wrong = seal_container(b"not this job", recipient=RECIPIENT, owner=OWNER)

    class LiarSource:
        async def get(self, cid):
            calls["n"] += 1
            return wrong

    with pytest.raises(BlobError, match="no source"):
        await BlobResolver([LiarSource()], attempts=8, backoff_s=0).fetch_task(job)
    assert calls["n"] == 1


async def test_dropping_the_liar_does_not_stop_an_honest_source_being_retried():
    # Only the liar is dropped. A source that merely could not answer keeps its
    # place in every remaining sweep, so a pin still propagating is still found.
    job = _job()
    calls = {"liar": 0, "flaky": 0}
    wrong = seal_container(b"not this job", recipient=RECIPIENT, owner=OWNER)

    class LiarSource:
        async def get(self, cid):
            calls["liar"] += 1
            return wrong

    class FlakySource:
        async def get(self, cid):
            calls["flaky"] += 1
            if calls["flaky"] < 3:
                raise BlobError("not pinned yet")
            return REAL

    resolver = BlobResolver([LiarSource(), FlakySource()], attempts=4, backoff_s=0)
    assert await resolver.fetch_task(job) == REAL
    assert calls["liar"] == 1
    assert calls["flaky"] == 3
