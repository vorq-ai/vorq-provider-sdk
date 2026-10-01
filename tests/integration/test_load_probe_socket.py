"""`LoadProbe` against a real socket serving a real exposition.

Every other test of this path hands the probe a string through an
`httpx.MockTransport`, which proves the parse and nothing about the fetch. Here
the probe makes an actual HTTP request to an actual server over loopback, using
the same `httpx.AsyncClient` the daemon builds, and the body it receives is a
vLLM-shaped exposition rather than the two lines a mock usually carries.

That gap is not hypothetical. What a mock cannot fail on: a URL the loader
accepted and `httpx` will not, a timeout that never fires because nothing was
ever slow, a non-200 that arrives as a status rather than an exception, a body
whose content type is not what the parser assumes. Each of those is a scrape the
daemon must survive by falling back to the operator's configured price — and
each would look identical to a healthy probe in a mocked test.

The e2e `pricing` tier runs this same shape against the daemon in a container
(see `e2e/test_pricing.py`); this one is here so a broken fetch path fails in the
unit suite, in under a second, without docker.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from vorqd.pricing import PROBE_TIMEOUT_S, STALE_AFTER_S, LoadProbe

#: A vLLM exposition, trimmed but structurally whole: `:`-spelled names, HELP and
#: TYPE lines, two engines on the metric under test, and a counter beside it.
#: `{load}` is substituted per-response.
EXPOSITION = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0",model_name="m"}} 4.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} {load}
vllm:kv_cache_usage_perc{{engine="1",model_name="m"}} 0.010000
# HELP vllm:generation_tokens Number of generation tokens processed.
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{{engine="0",model_name="m"}} 27453.0
"""

METRIC = "vllm:kv_cache_usage_perc"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):                           # noqa: N802 - stdlib naming
        mode = self.server.mode                 # type: ignore[attr-defined]
        if mode == "slow":
            # Longer than the probe's own timeout, so the timeout is what ends
            # the request rather than the server being merely unhurried.
            import time

            time.sleep(PROBE_TIMEOUT_S * 2)
        if mode == "error":
            self.send_response(503)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        body = (
            b"<html>not an exposition</html>" if mode == "garbage"
            else EXPOSITION.format(load=self.server.load).encode()  # type: ignore[attr-defined]
        )
        self.send_response(200)
        self.send_header("content-type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    """A real HTTP server on an ephemeral loopback port, torn down after."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.mode = "ok"
    srv.load = 0.25
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def url_of(srv, path: str = "/metrics") -> str:
    host, port = srv.server_address[:2]
    return f"http://{host}:{port}{path}"


async def probe_once(url: str, *, metric: str = METRIC, scale: float = 1.0,
                     clock=None) -> tuple[bool, float | None]:
    """One real scrape; answers `(ok, load)` after it."""
    async with httpx.AsyncClient() as http:
        probe = LoadProbe(url, metric, scale, http,
                          **({} if clock is None else {"clock": clock}))
        ok = await probe.refresh()
        return ok, probe.load()


def test_a_real_scrape_reads_the_busiest_engine(server):
    """The happy path, over a socket: max across label sets, exact value."""
    server.load = 0.25
    ok, load = asyncio.run(probe_once(url_of(server)))
    assert ok is True
    # 0.25, not the 0.01 of the quieter engine and not their mean.
    assert load == 0.25


def test_scale_divides_a_percentage_into_a_ratio(server):
    """A DCGM-style percentage gauge with `scale: 100`."""
    server.load = 40.0
    ok, load = asyncio.run(probe_once(url_of(server), scale=100.0))
    assert ok is True
    assert load == 0.4


def test_a_reading_over_the_scale_clamps_rather_than_exceeding_one(server):
    server.load = 250.0
    ok, load = asyncio.run(probe_once(url_of(server), scale=100.0))
    assert ok is True
    assert load == 1.0


@pytest.mark.parametrize("mode", ["error", "garbage", "slow"])
def test_a_failing_endpoint_is_a_false_and_never_an_exception(server, mode):
    """Every failure the socket can produce ends as an unknown load.

    `refresh` runs inline on the poll sweep, so an exception here would take the
    sweep with it — the daemon would stop claiming because a *metrics* endpoint
    was unhealthy, which inverts the whole posture of the feature.
    """
    server.mode = mode
    ok, load = asyncio.run(probe_once(url_of(server)))
    assert ok is False
    assert load is None


def test_a_missing_metric_on_a_healthy_endpoint_is_unknown(server):
    """A 200 with a perfectly good body that does not carry the series."""
    ok, load = asyncio.run(probe_once(url_of(server), metric="vllm:no_such_series"))
    assert ok is False
    assert load is None


def test_a_url_the_loader_accepts_but_httpx_cannot_reach_is_survivable():
    """A scheme-valid URL with nothing behind it — a typo in an operator's config.

    `config._build_load` validates the scheme and not the host, so this is what
    a misconfigured probe actually looks like at runtime.
    """
    ok, load = asyncio.run(probe_once("http://127.0.0.1:1/metrics"))
    assert ok is False
    assert load is None


def test_a_reading_expires_and_leaves_no_discount_behind(server):
    """A probe that answered once and then stopped must not look idle forever.

    The clock is injected rather than slept through — `STALE_AFTER_S` is 30 s and
    this is the fast suite — but the reading under it came off a real socket.
    """
    # Above the quieter engine's pinned 0.01, so the value asserted below is
    # unambiguously the one this test set.
    server.load = 0.05
    now = [1_000.0]

    async def run():
        async with httpx.AsyncClient() as http:
            probe = LoadProbe(url_of(server), METRIC, 1.0, http, clock=lambda: now[0])
            assert await probe.refresh() is True
            assert probe.load() == 0.05
            now[0] += STALE_AFTER_S / 2
            assert probe.load() == 0.05, "a reading must survive a missed scrape"
            now[0] += STALE_AFTER_S
            return probe.load()

    assert asyncio.run(run()) is None
