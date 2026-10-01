"""Per-model admission and retry spacing, run dry against a fake clock."""

from __future__ import annotations

import pytest

from vorqd.config import BackendConfig, ModelConfig, SlaRate
from vorqd.limits import (
    UNBOUNDED,
    Breaker,
    RetryPolicy,
    Throttle,
    parse_retry_after,
    retry_delay,
    window_seconds,
)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# --- retry spacing -----------------------------------------------------------


def test_retry_delay_doubles_from_the_backoff_and_caps():
    policy = RetryPolicy(attempts=8, backoff_s=30, backoff_max_s=900)
    assert [retry_delay(n, policy) for n in range(7)] == [30, 60, 120, 240, 480, 900, 900]


def test_a_retry_after_replaces_the_backoff_only_when_longer():
    policy = RetryPolicy(attempts=3, backoff_s=30, backoff_max_s=900)
    assert retry_delay(0, policy, retry_after_s=120) == 120
    assert retry_delay(3, policy, retry_after_s=5) == 240   # backoff already longer


@pytest.mark.parametrize("header, seconds", [
    ("7", 7.0), (" 2.5 ", 2.5), ("0", 0.0),
    (None, None), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("-3", None), ("", None),
])
def test_retry_after_reads_delay_seconds_only(header, seconds):
    assert parse_retry_after(header) == seconds


def test_the_policy_comes_from_the_backend_limits():
    be = BackendConfig(preset="openai-chat", params={}, retries=6, retry_backoff_s=10,
                       retry_backoff_max_s=100, timeout_s=310)
    assert RetryPolicy.from_backend(be) == RetryPolicy(
        attempts=7, backoff_s=10, backoff_max_s=100, timeout_s=310)


@pytest.mark.parametrize("window, secs", [("30s", 30), ("1m", 60), ("1h", 3600), ("24h", 86_400)])
def test_window_spelling_is_the_slas_spelling(window, secs):
    assert window_seconds(window) == secs


@pytest.mark.parametrize("window", ["", "1", "h", "1w", "1 h", "1.5h"])
def test_an_unreadable_window_raises(window):
    with pytest.raises(ValueError):
        window_seconds(window)


# --- the throttle --------------------------------------------------------------


def test_an_unconfigured_throttle_is_unbounded():
    t = Throttle(clock=Clock())
    assert not t.bounded
    assert t.claimable() == UNBOUNDED
    assert t.startable() == UNBOUNDED
    assert t.wait_s() == 0
    t.acquire()
    t.hold()
    assert t.claimable() == UNBOUNDED


def test_claimable_counts_held_jobs_and_startable_counts_attempts():
    t = Throttle(concurrency=2, clock=Clock())
    t.hold()
    assert t.claimable() == 1 and t.startable() == 2
    t.acquire()
    assert t.claimable() == 1 and t.startable() == 1
    t.release()
    assert t.startable() == 2 and t.claimable() == 1   # the job is still held
    t.drop()
    assert t.claimable() == 2


def test_the_tightest_window_bounds_admission():
    clock = Clock()
    t = Throttle(windows={60: 2, 3600: 3}, clock=clock)
    assert t.claimable() == 2
    t.acquire(); t.release()
    t.acquire(); t.release()
    assert t.claimable() == 0            # the minute is full
    clock.advance(61)
    assert t.claimable() == 1            # the minute cleared; the hour has one left
    t.acquire(); t.release()
    assert t.claimable() == 0
    clock.advance(3600)
    assert t.claimable() == 2


def test_wait_is_until_the_oldest_stamp_leaves_the_binding_window():
    clock = Clock()
    t = Throttle(windows={60: 2}, clock=clock)
    t.acquire(); t.release()
    clock.advance(10)
    t.acquire(); t.release()
    assert t.wait_s() == 50              # the first stamp falls out at +60
    clock.advance(50)
    assert t.wait_s() == 0
    assert t.claimable() == 1


def test_a_concurrency_stall_has_no_wait_of_its_own():
    t = Throttle(concurrency=1, clock=Clock())
    t.acquire()
    assert t.startable() == 0
    assert t.wait_s() == 0               # it clears on release, not on the clock


def test_retries_count_against_the_windows():
    clock = Clock()
    t = Throttle(concurrency=4, windows={60: 3}, clock=clock)
    t.hold()
    t.acquire(); t.release()             # first attempt
    t.acquire(); t.release()             # a retry
    assert t.claimable() == 1            # 3 - 2 started, and 4 - 1 held


def test_a_window_at_the_sla_lets_the_entry_hold_jobs_beyond_its_concurrency():
    t = Throttle(concurrency=2, windows={3600: 5}, sla_s=3600, clock=Clock())
    assert t.capacity == 5
    assert t.claimable() == 5 and t.startable() == 2
    t.hold(); t.acquire()
    t.hold(); t.acquire()
    assert t.claimable() == 3 and t.startable() == 0   # full on the wire, room to hold
    t.hold()                                            # a job waiting for a slot
    assert t.claimable() == 2 and t.startable() == 0   # 5 - 2 started - 1 waiting
    t.release()
    assert t.startable() == 1                            # the waiting job may start
    assert t.wait_s() == 0                               # and has no clock to wait on


def test_without_a_window_at_the_sla_a_claim_must_be_startable_now():
    t = Throttle(concurrency=2, windows={60: 10}, sla_s=3600, clock=Clock())
    assert t.capacity == 2                               # the minute is shorter than the hour
    assert t.claimable() == 2
    t.hold(); t.acquire()
    t.hold(); t.acquire()
    assert t.claimable() == 0
    assert Throttle(windows={60: 10}, clock=Clock()).capacity is None   # no SLA, no bound


def test_windows_shorter_than_the_sla_pace_starts_and_the_sla_window_bounds_the_claim():
    clock = Clock()
    t = Throttle(concurrency=1, windows={60: 1, 3600: 4}, sla_s=3600, clock=clock)
    assert t.claimable() == 4                            # the hour's budget, not the minute's
    t.hold(); t.acquire()                                # one on the wire; the minute is spent
    assert t.startable() == 0 and t.wait_s() == 60
    assert t.claimable() == 3                            # 4 - 1 started - 0 waiting
    t.hold(); t.hold()                                   # two more queued behind the minute
    assert t.claimable() == 1                            # 4 - 1 started - 2 waiting
    t.release(); t.drop()                                # the first settled
    assert t.claimable() == 1                            # still 4 - 1 started - 2 waiting
    clock.advance(61)
    t.acquire()                                          # one of the queued jobs started
    assert t.claimable() == 1                            # 4 - 2 started - 1 waiting
    t.release(); t.drop(); t.drop()
    assert t.claimable() == 2                            # 4 - 2 started, nothing waiting


def test_occupancy_is_the_fuller_of_held_jobs_and_the_days_budget():
    clock = Clock()
    t = Throttle(concurrency=2, windows={60: 10, 86400: 4}, sla_s=86400, clock=clock)
    assert t.occupancy() == 0.0
    t.hold(); t.hold()
    assert t.occupancy() == 0.5                              # 2 of 4 held, nothing started
    t.acquire(); t.release()
    t.acquire(); t.release()
    t.acquire(); t.release()
    assert t.occupancy() == 0.75                             # 3 of the day's 4 beats 2 of 4 held
    t.drop(); t.drop()
    assert t.occupancy() == 0.75                             # the budget does not free on a settle
    clock.advance(86401)
    assert t.occupancy() == 0.0
    assert Throttle(clock=Clock()).occupancy() is None      # nothing bounds it
    assert Throttle(windows={60: 5}, clock=Clock()).occupancy() == 0.0


def test_the_throttle_comes_from_the_entrys_limits_and_its_shortest_sla():
    be = BackendConfig(preset="openai-chat", params={}, concurrency=2,
                       rate_limit={"1m": 40, "24h": 5000})
    rate = SlaRate(rate_in="0.000001", rate_out="0.000001")
    day = Throttle.from_model(ModelConfig(model="m", slas={"24h": rate}, backend=be), clock=Clock())
    assert day.bounded and day.capacity == 5000 and day.claimable() == 5000
    hour = Throttle.from_model(ModelConfig(model="m", slas={"1h": rate, "24h": rate}, backend=be),
                               clock=Clock())
    assert hour.capacity == 2 and hour.claimable() == 2   # no window at the hour: start now


def test_a_window_longer_than_the_sla_bounds_the_claim_too():
    """A held job must start inside its SLA, which lies inside every longer
    window — so a spent day stops the claims of an hour-held entry."""
    clock = Clock()
    t = Throttle(concurrency=4, windows={60: 100, 3600: 10, 86400: 12}, sla_s=3600, clock=clock)
    assert t.capacity == 10
    for _ in range(12):
        t.acquire(); t.release()
        clock.advance(400)                                   # 12 starts over 80 min
    assert t.claimable() == 0                                # the day is spent
    assert t.wait_s() > 0
    clock.advance(86400)
    assert t.claimable() == 10


# --- the breaker ---------------------------------------------------------------


def test_the_breaker_trips_on_the_nth_consecutive_fault_and_not_before():
    clock = Clock()
    breaker = Breaker(trip_after=3, cooldown_s=60, clock=clock)
    assert breaker.record_failure() is False
    assert breaker.record_failure() is False
    assert not breaker.tripped()
    assert breaker.record_failure() is True
    assert breaker.tripped()
    assert breaker.open_until == clock() + 60


def test_a_success_clears_the_streak():
    breaker = Breaker(trip_after=3, cooldown_s=60, clock=Clock())
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.record_failure() is False
    assert not breaker.tripped()


def test_faults_while_open_do_not_retrip():
    clock = Clock()
    breaker = Breaker(trip_after=1, cooldown_s=60, clock=clock)
    assert breaker.record_failure() is True
    opened = breaker.open_until
    clock.advance(10)
    assert breaker.record_failure() is False   # one push per trip, not per fault
    assert breaker.open_until == opened


def test_one_fault_after_the_cooldown_retrips():
    """Re-listed on trust, not on evidence: the streak survives the cooldown."""
    clock = Clock()
    breaker = Breaker(trip_after=3, cooldown_s=60, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(61)
    assert not breaker.tripped()
    assert breaker.record_failure() is True
    assert breaker.tripped()


def test_a_disabled_breaker_never_trips():
    breaker = Breaker(trip_after=0, cooldown_s=60, clock=Clock())
    assert all(breaker.record_failure() is False for _ in range(10))
    assert not breaker.tripped()


def test_the_breaker_comes_from_the_backend_limits():
    be = BackendConfig(preset="openai-chat", trip_after=2, trip_cooldown_s=10)
    breaker = Breaker.from_backend(be, clock=Clock())
    breaker.record_failure()
    assert breaker.record_failure() is True
    assert breaker.open_until == 1010.0
