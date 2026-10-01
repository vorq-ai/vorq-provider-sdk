"""Per-model admission and retry policy for a backend with a request quota.

A hosted or shared backend usually bounds what one key may do: so many requests
per rolling window, so many in flight at once, and a ceiling on how long a
single request may run. The daemon has to keep to those bounds *before* it
claims — a claim it cannot start burns the client's order — and it has to give
a failed attempt another chance without ever running past the job's deadline.

Two pure pieces, both driven by an injected clock so tests can run them dry:

- :class:`Throttle` — one per model entry. Counts the attempts started inside
  each configured window and the requests in flight, and answers how many more
  jobs the daemon may take: what it could start **now**, or, with a window at
  the entry's SLA, what that window can still hold. That number is what the
  poll sends the coordinator as ``free``, so a model whose budget is spent is
  leased nothing.
- :class:`RetryPolicy` and :func:`retry_delay` — how attempts are spaced. The
  scheduler owns the loop and the deadline check; this module only says how
  long to wait.
- :class:`Breaker` — one per model entry. Counts consecutive jobs the backend
  failed outright and, past ``trip_after`` of them, says the model is off the
  book for a cooldown. The scheduler withdraws and re-lists; this only keeps
  the streak and the clock.

Nothing here does I/O or knows about jobs.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

_WINDOW_RE = re.compile(r"^(\d+)([smhd])$")
_WINDOW_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86_400}

#: What an unconfigured bound answers: more than any capacity the daemon could
#: request, so ``min()`` against the real slots always picks the real slots.
UNBOUNDED = 1 << 30


def window_seconds(window: str) -> int:
    """``"24h"`` → ``86400``. Raises :class:`ValueError` on anything else.

    The one spelling for a rolling window anywhere in the config — the SLA
    windows under ``slas`` and the quota windows under ``rate_limit`` read the
    same way.
    """
    match = _WINDOW_RE.match(str(window).strip())
    if match is None:
        raise ValueError(
            f"window {window!r} is not a count and a unit (s, m, h, d), e.g. '1m' or '24h'"
        )
    return int(match.group(1)) * _WINDOW_UNITS[match.group(2)]


def parse_retry_after(value: str | None) -> float | None:
    """The seconds a ``Retry-After`` header asks for, or ``None``.

    Only the delay-seconds form is read. The HTTP-date form needs the server's
    clock to mean anything and is rare on inference APIs, so it reads as absent
    and the configured backoff applies.
    """
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


@dataclass(frozen=True)
class RetryPolicy:
    """How one job's attempts are spaced.

    ``attempts`` is the first try plus ``retries``. The delay before retry *n*
    (0-based) is ``backoff_s * 2**n`` capped at ``backoff_max_s``; a larger
    ``Retry-After`` replaces it. ``timeout_s`` bounds one attempt — the
    scheduler additionally caps it at the time left before the deadline.
    """

    attempts: int = 1
    backoff_s: float = 30.0
    backoff_max_s: float = 900.0
    timeout_s: float | None = None

    @classmethod
    def from_backend(cls, be) -> "RetryPolicy":
        return cls(
            attempts=int(be.retries) + 1,
            backoff_s=float(be.retry_backoff_s),
            backoff_max_s=float(be.retry_backoff_max_s),
            timeout_s=float(be.timeout_s) if be.timeout_s is not None else None,
        )


def retry_delay(attempt: int, policy: RetryPolicy, retry_after_s: float | None = None) -> float:
    """Seconds to wait before retry number ``attempt`` (0 for the first retry)."""
    delay = min(policy.backoff_s * (2 ** attempt), policy.backoff_max_s)
    if retry_after_s is not None:
        delay = max(delay, retry_after_s)
    return float(delay)


class Throttle:
    """Admission for one model entry: rolling-window request counts plus a
    concurrency cap.

    Two counters, because a claimed job and a request on the wire are different
    things. ``held`` is jobs this entry owns — claimed and not yet settled or
    failed, a job sleeping between attempts included. ``inflight`` is attempts
    currently running. The claim gate reads ``held`` (a job waiting to retry
    still occupies its slot, or the poll would keep claiming), the attempt gate
    reads ``inflight``.

    Windows count *attempts started*, retries included, against the limit the
    operator set for that window. Timestamps older than the longest window are
    pruned on every call, so memory is bounded by the largest limit.

    The window **equal to the shortest SLA the entry serves** (``sla_s``) is
    also what the entry may **hold**: a job claimed for a ``24h`` window need
    not start now, only inside its day, so the day's budget — less the attempts
    already started in it and one reserved for every job still waiting — is how
    many jobs the entry may have claimed at once. A longer window bounds the
    claim the same way (a held job must start inside its SLA, which lies inside
    any longer window); shorter windows pace the starts. Without such a window
    (or with no ``sla_s``) every claim must be startable now, so
    ``concurrency`` and every window bound the claim gate alike.
    """

    def __init__(self, *, concurrency: int | None = None,
                 windows: dict[int, int] | None = None, sla_s: int | None = None,
                 clock=time.monotonic) -> None:
        self._concurrency = concurrency
        # window seconds -> max attempts started inside it; longest first so a
        # single pass can prune against the longest and count the rest.
        self._windows = sorted((windows or {}).items(), reverse=True)
        # The window at the SLA is what the entry may hold; with one, every
        # window no shorter than the SLA bounds the claim.
        self._hold_limit = next((n for s, n in self._windows if s == sla_s), None)
        self._bounding = [(s, n) for s, n in self._windows if s >= sla_s] \
            if self._hold_limit is not None else []
        self._clock = clock
        self._started: deque[float] = deque()
        self.held = 0
        self.inflight = 0

    @classmethod
    def from_model(cls, model, clock=time.monotonic) -> "Throttle":
        """The throttle for one model entry: its backend limits, and its
        shortest SLA window as the window that bounds holding."""
        be = model.backend
        windows = {window_seconds(w): int(n) for w, n in (be.rate_limit or {}).items()}
        slas = [window_seconds(w) for w in (model.slas or {})]
        return cls(concurrency=be.concurrency, windows=windows,
                   sla_s=min(slas) if slas else None, clock=clock)

    @property
    def bounded(self) -> bool:
        return self._concurrency is not None or bool(self._windows)

    @property
    def capacity(self) -> int | None:
        """The most jobs this entry may hold at once: the budget of the window
        at its SLA, else ``concurrency``; ``None`` when nothing bounds it."""
        return self._concurrency if self._hold_limit is None else self._hold_limit

    # -- jobs ------------------------------------------------------------------

    def hold(self) -> None:
        self.held += 1

    def drop(self) -> None:
        self.held = max(0, self.held - 1)

    # -- attempts ----------------------------------------------------------------

    def acquire(self) -> None:
        """Start one attempt now: stamp every window and take an in-flight slot."""
        self._prune()
        self._started.append(self._clock())
        self.inflight += 1

    def release(self) -> None:
        self.inflight = max(0, self.inflight - 1)

    # -- questions ---------------------------------------------------------------

    def claimable(self) -> int:
        """How many more jobs this entry could take — the poll's ``free``.

        With no window at the SLA, how many it could *start* now. Otherwise how
        many it could hold: the tightest slack among that window and every
        longer one, once every waiting job has an attempt reserved.
        """
        if self._hold_limit is None:
            return self._slack(self.held)
        self._prune()
        now = self._clock()
        waiting = max(0, self.held - self.inflight)
        slack = min(limit - sum(1 for t in self._started if t > now - seconds) - waiting
                    for seconds, limit in self._bounding)
        return max(0, slack)

    def startable(self) -> int:
        """How many more attempts could go on the wire now."""
        return self._slack(self.inflight)

    def occupancy(self) -> float | None:
        """How much of this entry's own allowance is spoken for, 0..1.

        The load reading for a backend that exports none — a hosted API behind
        a quota has no metrics endpoint, but the daemon knows exactly what it
        has committed: the jobs held over the jobs it may hold, and the attempts
        started in the longest window over that window's budget. The fuller of
        the two is the answer, as the busiest sample is for a probe. ``None``
        when nothing bounds the entry: an unbounded backend has no fullness to
        report.
        """
        readings = []
        if self.capacity is not None:
            readings.append(self.held / self.capacity)
        if self._windows:
            self._prune()
            seconds, limit = self._windows[0]
            inside = sum(1 for t in self._started if t > self._clock() - seconds)
            readings.append(inside / limit)
        if not readings:
            return None
        return min(1.0, max(readings))

    def wait_s(self) -> float:
        """Seconds until the windows admit another attempt; ``0`` when they do now.

        Only the windows are waited on. A concurrency stall clears when a
        running attempt returns, which the caller sees by asking again after
        its own release — there is no instant to sleep until.
        """
        self._prune()
        now = self._clock()
        wait = 0.0
        for seconds, limit in self._windows:
            inside = [t for t in self._started if t > now - seconds]
            if len(inside) >= limit:
                # The oldest stamp still inside is the next one to fall out.
                oldest = inside[-limit]
                wait = max(wait, oldest + seconds - now)
        return wait

    def _slack(self, occupied: int) -> int:
        self._prune()
        now = self._clock()
        slack = UNBOUNDED
        if self._concurrency is not None:
            slack = min(slack, self._concurrency - occupied)
        for seconds, limit in self._windows:
            inside = sum(1 for t in self._started if t > now - seconds)
            slack = min(slack, limit - inside)
        return max(0, slack)

    def _prune(self) -> None:
        if not self._windows:
            self._started.clear()
            return
        horizon = self._clock() - self._windows[0][0]
        while self._started and self._started[0] <= horizon:
            self._started.popleft()


class Breaker:
    """A per-model circuit: ``trip_after`` consecutive backend faults take the
    model off the book for ``cooldown_s``.

    ``streak`` is faults since the last success. It is **not** cleared when the
    cooldown ends: the model is re-listed on trust, so one more fault re-trips
    it at once, while one success clears everything. ``trip_after`` of ``0``
    disables the breaker.
    """

    def __init__(self, *, trip_after: int = 3, cooldown_s: float = 60.0,
                 clock=time.monotonic) -> None:
        self._trip_after = trip_after
        self._cooldown_s = cooldown_s
        self._clock = clock
        self.streak = 0
        self.open_until: float | None = None

    @classmethod
    def from_backend(cls, be, clock=time.monotonic) -> "Breaker":
        return cls(trip_after=int(be.trip_after), cooldown_s=float(be.trip_cooldown_s),
                   clock=clock)

    def record_failure(self) -> bool:
        """Count one fault. ``True`` exactly when this one trips the breaker."""
        self.streak += 1
        if self._trip_after <= 0 or self.streak < self._trip_after or self.tripped():
            return False
        self.open_until = self._clock() + self._cooldown_s
        return True

    def record_success(self) -> None:
        self.streak = 0
        self.open_until = None

    def tripped(self) -> bool:
        return self.open_until is not None and self._clock() < self.open_until
