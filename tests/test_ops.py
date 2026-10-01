"""Ops server, Prometheus metrics, structured JSON logging."""

from __future__ import annotations

import json
import logging

import pytest

from vorqd.ops import JsonFormatter, Metrics, OpsServer, configure_logging

COLLECTORS = [
    "vorqd_jobs_claimed_total",
    "vorqd_jobs_settled_total",
    "vorqd_jobs_failed_total",
    "vorqd_job_duration_seconds",
    "vorqd_backend_latency_seconds",
    "vorqd_capacity_free",
    "vorqd_asks_published",
    "vorqd_backend_load",
    "vorqd_bid_floor_discount_pct",
    "vorqd_load_probe_failures_total",
]


def dump(metrics: Metrics) -> str:
    from prometheus_client import generate_latest

    return generate_latest(metrics.registry).decode()


def test_metrics_exposes_every_collector():
    m = Metrics()
    m.on_claim()
    m.on_settle(1.5)
    m.on_fail("backend_error")
    m.on_backend_latency("m:fp8", 0.2)
    m.set_capacity_free(3)
    m.set_asks_published(2)
    m.set_backend_load("m:fp8", 0.42)
    m.set_floor_discount("m:fp8", 15)
    m.on_probe_failure("m:fp8")
    text = dump(m)
    for name in COLLECTORS:
        assert name in text


def test_the_pricing_gauges_carry_the_current_reading():
    m = Metrics()
    m.set_backend_load("m:fp8", 0.42)
    m.set_floor_discount("m:fp8", 15)
    m.on_probe_failure("m:fp8")
    text = dump(m)
    assert 'vorqd_backend_load{model="m:fp8"} 0.42' in text
    assert 'vorqd_bid_floor_discount_pct{model="m:fp8"} 15.0' in text
    assert 'vorqd_load_probe_failures_total{model="m:fp8"} 1.0' in text


def test_counters_move():
    m = Metrics()
    m.on_claim()
    m.on_settle(2.0)
    text = dump(m)
    assert "vorqd_jobs_claimed_total 1.0" in text
    assert "vorqd_jobs_settled_total 1.0" in text
    m.on_fail("sla_abandon")
    assert 'vorqd_jobs_failed_total{reason="sla_abandon"} 1.0' in dump(m)


async def test_healthz_reflects_health_fn():
    ok = OpsServer(Metrics(), health_fn=lambda: True, port=0)
    status, _, _ = await ok.handle("GET", "/healthz")
    assert status == 200

    bad = OpsServer(Metrics(), health_fn=lambda: False, port=0)
    status, _, _ = await bad.handle("GET", "/healthz")
    assert status == 503


async def test_healthz_supports_async_health_fn():
    async def healthy():
        return True

    srv = OpsServer(Metrics(), health_fn=healthy, port=0)
    status, _, _ = await srv.handle("GET", "/healthz")
    assert status == 200


async def test_metrics_endpoint_serves_exposition():
    m = Metrics()
    m.on_claim()
    srv = OpsServer(m, health_fn=lambda: True, port=0)
    status, ctype, body = await srv.handle("GET", "/metrics")
    assert status == 200
    assert "text/plain" in ctype
    assert "vorqd_jobs_claimed_total" in body.decode()


async def test_real_socket_round_trip():
    import httpx

    m = Metrics()
    srv = OpsServer(m, health_fn=lambda: True, port=0)
    await srv.start()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://127.0.0.1:{srv.port}/healthz")
            assert r.status_code == 200
            r2 = await client.get(f"http://127.0.0.1:{srv.port}/metrics")
            assert r2.status_code == 200
            assert "vorqd_" in r2.text
    finally:
        await srv.stop()


def test_json_log_carries_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord("vorqd", logging.INFO, __file__, 1, "claimed", None, None)
    record.job_id = "job_1"
    record.model = "m:fp8"
    payload = json.loads(formatter.format(record))
    assert payload["event"] == "claimed"
    assert payload["job_id"] == "job_1"
    assert payload["model"] == "m:fp8"
    assert payload["level"] == "info"


def test_configure_logging_sets_json_handler():
    logger = configure_logging("debug")
    assert logger.level == logging.DEBUG
    assert any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers)
