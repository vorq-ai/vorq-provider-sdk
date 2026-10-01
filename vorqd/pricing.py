"""What this daemon will quietly accept, as against what it advertises.

A provider publishes one price and holds another. The published ask is the
operator's list price and it does not move with load — a floor the network can
see is a floor the network can price against, and bids would simply converge onto
it. What moves is the **private** floor underneath: when the backend is idle, a
fill below the list price beats an idle GPU, so the daemon claims work its own
ask says is underpriced.

Nothing here is published, signed or transmitted. It exists entirely inside one
decision — :meth:`Scheduler.profitable` — and the chain permits that decision
because ``JobRegistry.claim`` never consults the ask book: it charges the rates
the client signed, and any listed provider may take any open job.

Everything is integer arithmetic on atomic units, and the one rounding rule is
**round toward the undiscounted price**, so a floor never drifts under what the
operator meant by a fraction of a unit.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families

# The cap is defined beside the knobs it bounds, and imported here because this
# is where a percentage first becomes a price.
from .config import MAX_FLOOR_DISCOUNT_PCT, PricingConfig

log = logging.getLogger("vorqd")


def discounted_units(base: int, pct: int) -> int:
    """``base`` less ``pct`` percent, rounded toward ``base``.

    Integer floor division on the *discount* rather than on the result, so the
    remainder always stays with the provider.

    ``pct`` is clamped to :data:`MAX_FLOOR_DISCOUNT_PCT` here, not by the caller.
    Callers sum two independently-capped allowances, so the total can exceed the
    cap; past 100 an unclamped floor goes *negative*, which would accept a bid
    paying less than nothing. Enforcing it at the one point where a percentage
    becomes a price keeps the guarantee true of this function standing alone:
    every nonzero rate stays nonzero, whatever it is handed.
    """
    if pct <= 0:
        return base
    pct = min(pct, MAX_FLOOR_DISCOUNT_PCT)
    return base - (base * pct) // 100


def raw_discount_pct(cfg: PricingConfig, load: float | None) -> float:
    """The discount this load earns, in percentage points.

    Linear: the whole allowance at an idle backend, nothing at a full one.
    ``None`` — no probe configured, or no fresh sample — earns nothing. An
    unknown load is not an idle GPU, and the conservative reading is the price
    the operator configured.
    """
    if load is None:
        return 0.0
    return cfg.max_discount_pct * (1.0 - load)


#: How long a load reading stays usable after the scrape that produced it.
#:
#: Long enough that a missed scrape or two does not move the floor, short enough
#: that a runtime which stopped answering stops earning a discount.
STALE_AFTER_S = 30.0

#: Per-scrape timeout. The probe runs inline on the poll sweep, so a backend
#: whose metrics endpoint hangs must not hold up claiming.
PROBE_TIMEOUT_S = 2.0


def _read(text: str, metric: str) -> float | None:
    """The busiest sample of ``metric`` in a Prometheus text exposition.

    Names are compared with ``:`` and ``_`` folded together: the same series has
    been spelled ``vllm:kv_cache_usage_perc`` and ``vllm_kv_cache_usage_perc``
    across runtime versions, and an operator copying either spelling means the
    same thing.

    **Max, not mean.** A metric with several label sets is usually one sample per
    GPU, and the question this answers is whether another job fits — which the
    busiest card decides.

    **Samples that are not a load are skipped rather than compared**, and that is
    a safety property rather than tidiness. Both kinds are ordinary in this
    exposition: ``NaN`` for a gauge before its first sample or a card that is
    unreadable or MIG-partitioned, and a negative sentinel for a field an
    exporter could not read. Neither can be a load, and both share one fate at
    the caller — ``min(1.0, max(0.0, value / scale))`` turns each into ``0.0``,
    a perfectly idle GPU earning the largest discount there is. Skipping them
    here makes the reading *unknown* instead, which every caller treats as busy.
    ``NaN`` has a second reason on top: it loses every ``>`` comparison, so a
    leading one would shadow every real sample behind it.
    """
    want = metric.replace(":", "_")
    best: float | None = None
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.replace(":", "_") != want:
                continue
            # Also drops ±Inf and every negative value, deliberately: neither an
            # infinite reading nor a below-zero one is a real load, and
            # "unknown" is the conservative answer to both.
            if not math.isfinite(sample.value) or sample.value < 0.0:
                continue
            if best is None or sample.value > best:
                best = sample.value
    return best


class LoadProbe:
    """One backend's load, scraped from the Prometheus endpoint it already serves.

    Deliberately read-only and deliberately dumb: it does not know what a rate
    is, it holds one number, and it forgets that number when it gets old. The
    daemon never asks the backend to do anything on its behalf — it reads the
    same ``/metrics`` the operator's own monitoring reads.
    """

    def __init__(self, url: str, metric: str, scale: float, http: httpx.AsyncClient,
                 *, clock=time.time) -> None:
        self._url = url
        self._metric = metric
        self._scale = scale
        self._http = http
        self._clock = clock
        self._sample: float | None = None
        self._at = 0.0

    async def refresh(self) -> bool:
        """Scrape once. ``False`` is an ordinary outcome, never an exception.

        A metrics endpoint is not a dependency of serving work: if it does not
        answer, the daemon keeps claiming and settling at its configured price.
        """
        try:
            resp = await self._http.get(self._url, timeout=PROBE_TIMEOUT_S)
            resp.raise_for_status()
            value = _read(resp.text, self._metric)
        # `InvalidURL` is neither an `HTTPError` nor a `ValueError`, and the
        # loader only validates the scheme — a bad port reaches this call. This
        # runs inline on the poll sweep, so nothing here may raise into it.
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            log.warning("load probe %s failed (%s); the floor falls back to the configured rates",
                        self._url, type(exc).__name__)
            return False
        if value is None:
            # `_read` answers `None` for two different endpoint states, and an
            # operator debugging one is not helped by being told the other: the
            # metric may be absent, or present with every sample unusable (a
            # faulted card reporting `NaN` or a negative sentinel). Say both,
            # rather than name the one that is easier to phrase.
            log.warning("load probe %s carries no usable sample for metric %r — absent, or "
                        "present with every sample unreadable; the floor falls back to the "
                        "configured rates", self._url, self._metric)
            return False
        self._sample = min(1.0, max(0.0, value / self._scale))
        self._at = self._clock()
        return True

    def load(self) -> float | None:
        """The last reading if it is still fresh, else ``None``.

        ``None`` is "unknown", and every caller treats unknown as *busy* — no
        discount, no tolerance. A stale probe must never look like an idle GPU.
        """
        if self._sample is None or self._clock() - self._at > STALE_AFTER_S:
            return None
        return self._sample


class LoadMonitor:
    """Every configured probe, refreshed once per poll sweep."""

    def __init__(self, probes: dict[str, LoadProbe], metrics) -> None:
        self._probes = probes
        self._metrics = metrics

    async def refresh(self) -> None:
        """Scrape every backend at once, then meter the results.

        Concurrently, because this runs inline on the poll sweep and **ahead of**
        publishing and claiming: awaited one after another, N backends whose
        metrics endpoints hang would add ``N * PROBE_TIMEOUT_S`` to every sweep,
        delaying the work the sweep exists to do. Gathering is safe because
        :meth:`LoadProbe.refresh` reports a bad scrape as ``False`` rather than
        raising — there is no exception here for a sibling to lose.

        Results are zipped back to the models that produced them, so each model
        still gets exactly one of ``set_backend_load`` / ``on_probe_failure``.
        """
        models = list(self._probes)
        results = await asyncio.gather(*(self._probes[m].refresh() for m in models))
        for model, ok in zip(models, results):
            if ok:
                self._metrics.set_backend_load(model, self._probes[model].load())
            else:
                self._metrics.on_probe_failure(model)

    def load(self, model: str) -> float | None:
        probe = self._probes.get(model)
        return probe.load() if probe is not None else None


class NullLoadMonitor:
    """No probes configured: every model's load is unknown, forever.

    Which makes the whole feature inert rather than guessing — a daemon with no
    probe accepts exactly its configured rates.
    """

    async def refresh(self) -> None: ...

    def load(self, model: str) -> float | None:
        return None
