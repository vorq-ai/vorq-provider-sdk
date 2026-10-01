"""The discount arithmetic behind the private acceptance floor."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from vorqd.config import PricingConfig
from vorqd.pricing import (
    MAX_FLOOR_DISCOUNT_PCT,
    STALE_AFTER_S,
    LoadMonitor,
    LoadProbe,
    NullLoadMonitor,
    discounted_units,
    raw_discount_pct,
)


def test_a_discount_is_integer_arithmetic_on_atomic_units():
    assert discounted_units(600_000, 10) == 540_000
    assert discounted_units(600_000, 0) == 600_000
    # 18-decimal rates (what the example configs carry) stay exact — no float
    # anywhere in this path.
    assert discounted_units(750_000_000_000_000_000, 20) == 600_000_000_000_000_000


def test_a_discount_rounds_toward_the_undiscounted_price():
    """The remainder stays with the provider: `999` at 5% is `950`, not `949`.
    A floor must never drift under what the operator meant by a fraction."""
    assert discounted_units(999, 5) == 950   # 999 - (4995 // 100 = 49)


def test_a_nonzero_floor_never_discounts_to_zero():
    """A floor of zero would accept literally any bid, including one that pays
    nothing. The cap plus the rounding rule is what forecloses that."""
    for pct in range(0, MAX_FLOOR_DISCOUNT_PCT + 1):
        assert discounted_units(1, pct) == 1


def test_a_discount_past_the_cap_is_clamped_not_inverted():
    """Two allowances can sum past the cap. Unclamped, `base - (base*180)//100`
    is *negative* — a floor that accepts a bid paying less than nothing. The cap
    is enforced here so the function is safe on its own."""
    assert discounted_units(600_000, 180) == discounted_units(600_000, MAX_FLOOR_DISCOUNT_PCT)
    assert discounted_units(1, 180) == 1


def test_the_raw_discount_falls_linearly_to_nothing_as_load_rises():
    cfg = PricingConfig(max_discount_pct=20)
    assert raw_discount_pct(cfg, 0.0) == 20.0
    assert raw_discount_pct(cfg, 0.5) == 10.0
    assert raw_discount_pct(cfg, 1.0) == 0.0


def test_an_unknown_load_discounts_nothing():
    """A dark or stale probe is not an idle GPU. The conservative reading is the
    price the operator configured."""
    assert raw_discount_pct(PricingConfig(max_discount_pct=20), None) == 0.0


# --- the load probe ----------------------------------------------------------

VLLM = """# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="r"} 0.42
vllm:num_requests_running{model_name="r"} 3.0
"""

DCGM = """# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 71.0
DCGM_FI_DEV_GPU_UTIL{gpu="1"} 88.0
"""


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def probe_client(body, *, status=200, fail=False):
    def handle(request):
        if fail:
            raise httpx.ConnectError("nothing is listening", request=request)
        return httpx.Response(status, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def test_a_gauge_becomes_a_load_reading():
    probe = LoadProbe("http://b/metrics", "vllm:kv_cache_usage_perc", 1.0,
                      probe_client(VLLM), clock=Clock())
    assert await probe.refresh() is True
    assert probe.load() == pytest.approx(0.42)


async def test_the_colon_and_underscore_spellings_are_one_metric():
    """Runtimes have renamed `vllm:x` to `vllm_x` across versions and operators
    copy whichever docs they found. Both must name the same series."""
    probe = LoadProbe("http://b/metrics", "vllm_kv_cache_usage_perc", 1.0,
                      probe_client(VLLM), clock=Clock())
    await probe.refresh()
    assert probe.load() == pytest.approx(0.42)


async def test_the_busiest_sample_wins():
    """One metric, many label sets — one GPU per sample on a DCGM exporter. The
    busiest is what another job actually has to fit onto, so a mean would grant
    a discount the hot card cannot honour."""
    probe = LoadProbe("http://b/metrics", "DCGM_FI_DEV_GPU_UTIL", 100.0,
                      probe_client(DCGM), clock=Clock())
    await probe.refresh()
    assert probe.load() == pytest.approx(0.88)


async def test_a_reading_is_clamped_to_the_unit_interval():
    probe = LoadProbe("http://b/metrics", "x", 1.0,
                      probe_client("x 1.7\n"), clock=Clock())
    await probe.refresh()
    assert probe.load() == 1.0


NAN_ONLY = """# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0"} NaN
"""

NAN_AND_A_BUSY_CARD = """# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0"} NaN
DCGM_FI_DEV_GPU_UTIL{gpu="1"} 90.0
"""


async def test_a_non_finite_sample_reads_as_unknown_not_as_an_idle_gpu():
    """`NaN` is ordinary in this exposition — a runtime gauge before its first
    sample, a card that is unreadable or MIG-partitioned. It must not survive
    into the clamp, where `max(0.0, nan)` is `0.0`: a full discount handed out
    for a GPU nothing can read."""
    probe = LoadProbe("http://b/metrics", "DCGM_FI_DEV_GPU_UTIL", 100.0,
                      probe_client(NAN_ONLY), clock=Clock())
    assert await probe.refresh() is False
    assert probe.load() is None


async def test_one_unreadable_card_does_not_hide_a_busy_fleet():
    """A leading `NaN` seeds the max reduction, and `nan > best` is always False,
    so every real sample after it would be shadowed."""
    probe = LoadProbe("http://b/metrics", "DCGM_FI_DEV_GPU_UTIL", 100.0,
                      probe_client(NAN_AND_A_BUSY_CARD), clock=Clock())
    await probe.refresh()
    assert probe.load() == pytest.approx(0.90)


NEGATIVE_ONLY = """# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0"} -1
"""


async def test_a_negative_sample_reads_as_unknown_not_as_an_idle_gpu():
    """A faulted card is the same failure as an unreadable one, through the same
    clamp. An exporter that cannot read a field emits a negative sentinel, and
    `max(0.0, -1 / 100)` is `0.0` — a one-GPU host whose card faulted would
    otherwise read as completely idle and earn the largest discount the config
    allows, with no warning and no probe-failure tick to show for it."""
    probe = LoadProbe("http://b/metrics", "DCGM_FI_DEV_GPU_UTIL", 100.0,
                      probe_client(NEGATIVE_ONLY), clock=Clock())
    assert await probe.refresh() is False
    assert probe.load() is None


async def test_an_unusable_probe_url_is_a_failed_scrape_not_a_crash():
    """`httpx.InvalidURL` is neither an `HTTPError` nor a `ValueError`, and the
    loader only checks the scheme — so a bad port reaches this call. `refresh()`
    runs inline on the poll sweep and must never raise into it."""
    probe = LoadProbe("http://b:notaport/metrics", "x", 1.0,
                      probe_client("x 0.1\n"), clock=Clock())
    assert await probe.refresh() is False
    assert probe.load() is None


async def test_a_metric_the_endpoint_does_not_carry_reads_as_unknown():
    probe = LoadProbe("http://b/metrics", "not_there", 1.0,
                      probe_client(VLLM), clock=Clock())
    assert await probe.refresh() is False
    assert probe.load() is None


async def test_a_failed_scrape_keeps_the_last_reading_until_it_goes_stale():
    """One missed scrape must not move the floor. A minute of them must, because
    by then nothing here knows what the backend is doing."""
    clock = Clock()
    probe = LoadProbe("http://b/metrics", "vllm:kv_cache_usage_perc", 1.0,
                      probe_client(VLLM), clock=clock)
    await probe.refresh()
    probe._http = probe_client("", fail=True)

    clock.advance(5)
    assert await probe.refresh() is False
    assert probe.load() == pytest.approx(0.42)   # still fresh

    clock.advance(STALE_AFTER_S)
    assert probe.load() is None                  # stale: unknown, so no discount


async def test_an_http_error_is_a_failed_scrape_not_a_crash():
    """A metrics endpoint is not a dependency of serving work."""
    probe = LoadProbe("http://b/metrics", "x", 1.0,
                      probe_client("nope", status=503), clock=Clock())
    assert await probe.refresh() is False
    assert probe.load() is None


class CountingMetrics:
    def __init__(self):
        self.loads = []
        self.failures = []

    def set_backend_load(self, model, value):
        self.loads.append((model, value))

    def on_probe_failure(self, model):
        self.failures.append(model)


async def test_the_monitor_reports_each_backend_and_counts_its_failures():
    metrics = CountingMetrics()
    good = LoadProbe("http://a/metrics", "vllm:kv_cache_usage_perc", 1.0,
                     probe_client(VLLM), clock=Clock())
    bad = LoadProbe("http://b/metrics", "x", 1.0, probe_client("", fail=True), clock=Clock())
    monitor = LoadMonitor({"a:fp8": good, "b:fp8": bad}, metrics)

    await monitor.refresh()
    assert monitor.load("a:fp8") == pytest.approx(0.42)
    assert monitor.load("b:fp8") is None
    assert monitor.load("unconfigured:fp8") is None   # no probe is not an idle GPU
    assert metrics.loads == [("a:fp8", pytest.approx(0.42))]
    assert metrics.failures == ["b:fp8"]


class Rendezvous:
    """A transport that answers nobody until every probe has arrived.

    The seam that tells concurrent refreshes from serial ones without measuring
    wall-clock time: refreshed one after another, the first probe waits for
    peers that only start once it has returned, and never gets an answer.
    """

    def __init__(self, expected: int, body: str) -> None:
        self._expected = expected
        self._body = body
        self._all_here = asyncio.Event()
        self.arrived = 0

    async def __call__(self, request):
        self.arrived += 1
        if self.arrived == self._expected:
            self._all_here.set()
        # Bounded, so a serial implementation fails the test rather than hanging it.
        await asyncio.wait_for(self._all_here.wait(), 5.0)
        return httpx.Response(200, text=self._body)


async def test_every_probe_is_scraped_concurrently_not_one_after_another():
    """The refresh runs inline on the poll sweep, ahead of publishing and
    claiming, so N hanging probes at `PROBE_TIMEOUT_S` apiece would stretch the
    whole sweep by `~2N` seconds — delaying the claim, not just the gauge."""
    handler = Rendezvous(3, VLLM)
    metrics = CountingMetrics()
    probes = {
        f"m{i}:fp8": LoadProbe(f"http://b{i}/metrics", "vllm:kv_cache_usage_perc", 1.0,
                               httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                               clock=Clock())
        for i in range(3)
    }
    monitor = LoadMonitor(probes, metrics)

    await monitor.refresh()

    assert handler.arrived == 3
    assert [m for m, _ in metrics.loads] == ["m0:fp8", "m1:fp8", "m2:fp8"]


async def test_a_failing_probe_among_healthy_ones_still_meters_each_model_once():
    """Gathering must not shuffle a result onto another model's metric: every
    probe still reports either a load or a failure, and only for itself."""
    metrics = CountingMetrics()
    probes = {
        "good:fp8": LoadProbe("http://a/metrics", "vllm:kv_cache_usage_perc", 1.0,
                              probe_client(VLLM), clock=Clock()),
        "dark:fp8": LoadProbe("http://b/metrics", "x", 1.0,
                              probe_client("", fail=True), clock=Clock()),
        "busy:fp8": LoadProbe("http://c/metrics", "DCGM_FI_DEV_GPU_UTIL", 100.0,
                              probe_client(DCGM), clock=Clock()),
    }
    await LoadMonitor(probes, metrics).refresh()

    assert metrics.loads == [("good:fp8", pytest.approx(0.42)),
                             ("busy:fp8", pytest.approx(0.88))]
    assert metrics.failures == ["dark:fp8"]


async def test_the_null_monitor_answers_nothing_for_everything():
    monitor = NullLoadMonitor()
    await monitor.refresh()
    assert monitor.load("anything") is None
