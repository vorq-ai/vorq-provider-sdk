"""The ops surface: Prometheus ``/metrics``, ``/healthz``, and structured logs.

``Metrics`` owns the ten ``vorqd_*`` collectors in a private registry (so the
process can host several without clashing in tests). ``OpsServer`` serves both
endpoints over a small asyncio HTTP responder. ``configure_logging`` installs a
JSON formatter on the ``vorqd`` logger, one line per state transition.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.jobs_claimed = Counter("vorqd_jobs_claimed_total", "Jobs claimed", registry=self.registry)
        self.jobs_settled = Counter("vorqd_jobs_settled_total", "Jobs settled", registry=self.registry)
        self.jobs_failed = Counter("vorqd_jobs_failed_total", "Jobs failed", ["reason"], registry=self.registry)
        self.job_duration = Histogram("vorqd_job_duration_seconds", "Claim->settle wall-clock", registry=self.registry)
        self.backend_latency = Histogram("vorqd_backend_latency_seconds", "Backend latency", ["model"], registry=self.registry)
        self.capacity_free = Gauge("vorqd_capacity_free", "Free capacity slots", registry=self.registry)
        self.capacity_granted = Gauge("vorqd_capacity_granted", "Slots the network grants this provider on its record; the poll never offers more", registry=self.registry)
        self.asks_published = Gauge("vorqd_asks_published", "Asks on the order book", registry=self.registry)
        self.backend_load = Gauge("vorqd_backend_load", "Backend load 0..1 as the model's load probe reports it", ["model"], registry=self.registry)
        # NOT the published ask, which never moves: this is how far UNDER the
        # published ask this daemon is currently willing to claim.
        self.floor_discount = Gauge("vorqd_bid_floor_discount_pct", "How far under the published ask a bid may be priced and still be claimed, in percent", ["model"], registry=self.registry)
        self.probe_failures = Counter("vorqd_load_probe_failures_total", "Load probe scrapes that did not yield a reading", ["model"], registry=self.registry)
        self.backend_retries = Counter("vorqd_backend_retries_total", "Backend attempts retried after a retryable failure", ["model"], registry=self.registry)
        # What the last poll told the coordinator this model could start: the
        # tightest of the global slots and the model's own limits.
        self.model_free = Gauge("vorqd_model_free", "Slots offered to the coordinator for this model on the last poll", ["model"], registry=self.registry)

    def on_claim(self) -> None:
        self.jobs_claimed.inc()

    def on_settle(self, duration: float) -> None:
        self.jobs_settled.inc()
        self.job_duration.observe(duration)

    def on_fail(self, reason: str) -> None:
        self.jobs_failed.labels(reason=reason).inc()

    def on_backend_latency(self, model: str, seconds: float) -> None:
        self.backend_latency.labels(model=model).observe(seconds)

    def set_capacity_free(self, n: int) -> None:
        self.capacity_free.set(n)

    def set_capacity_granted(self, n: int) -> None:
        self.capacity_granted.set(n)

    def set_asks_published(self, n: int) -> None:
        self.asks_published.set(n)

    def set_backend_load(self, model: str, value: float) -> None:
        self.backend_load.labels(model=model).set(value)

    def set_floor_discount(self, model: str, pct: int) -> None:
        self.floor_discount.labels(model=model).set(pct)

    def on_probe_failure(self, model: str) -> None:
        self.probe_failures.labels(model=model).inc()

    def on_retry(self, model: str) -> None:
        self.backend_retries.labels(model=model).inc()

    def set_model_free(self, model: str, n: int) -> None:
        self.model_free.labels(model=model).set(n)


class OpsServer:
    def __init__(self, metrics: Metrics, health_fn, port: int, host: str = "0.0.0.0") -> None:
        self._metrics = metrics
        self._health_fn = health_fn
        self._port = port
        self._host = host
        self._server: asyncio.AbstractServer | None = None

    @property
    def port(self) -> int:
        if self._server is not None:
            return self._server.sockets[0].getsockname()[1]
        return self._port

    async def _health_ok(self) -> bool:
        result = self._health_fn()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def handle(self, method: str, path: str) -> tuple[int, str, bytes]:
        if path == "/healthz":
            ok = await self._health_ok()
            return (200, "text/plain", b"ok") if ok else (503, "text/plain", b"unhealthy")
        if path == "/metrics":
            return 200, CONTENT_TYPE_LATEST, generate_latest(self._metrics.registry)
        return 404, "text/plain", b"not found"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._on_conn, self._host, self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _on_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode("latin1").split()
            method, target = (parts[0], parts[1]) if len(parts) >= 2 else ("GET", "/")
            while True:  # drain headers
                header = await reader.readline()
                if header in (b"\r\n", b"\n", b""):
                    break
            status, ctype, body = await self.handle(method, target.split("?", 1)[0])
            reason = {200: "OK", 404: "Not Found", 503: "Service Unavailable"}.get(status, "OK")
            head = (
                f"HTTP/1.1 {status} {reason}\r\n"
                f"Content-Type: {ctype}\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(head.encode("latin1") + body)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):  # pragma: no cover
            pass
        finally:
            writer.close()


class JsonFormatter(logging.Formatter):
    _FIELDS = ("job_id", "model", "request_id", "result_cid")

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in self._FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "info") -> logging.Logger:
    logger = logging.getLogger("vorqd")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def configure_error_reporting() -> None:
    """Sentry, on when ``SENTRY_DSN`` is set and a no-op otherwise.

    ``ERROR`` log lines and unhandled exceptions, including those of asyncio
    tasks, become events; ``INFO`` and up ride along as breadcrumbs. Called
    inside the running loop, which the asyncio integration needs. Local
    variables stay off: a frame holding the config or the signer would ship
    the wallet key. No trace headers go out: the backend is someone else's API.
    """
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import AsyncioIntegration

    sentry_sdk.init(
        include_local_variables=False,
        trace_propagation_targets=[],
        integrations=[AsyncioIntegration()],
    )


def report_job_failed(model: str, reason: str, job_id: str, detail: str = "") -> None:
    """One Sentry event for a job the daemon gave back, grouped by model.

    The fingerprint is the model and nothing else, so every failure on one
    model lands in one issue whatever the reason or the wording — the reason,
    which is the ``vorqd_jobs_failed_total`` label, and the job id are tags to
    filter that issue by. ``detail`` is the daemon's own account of the failure
    and never the client's payload. A no-op without ``SENTRY_DSN``.
    """
    import sentry_sdk

    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = ["job-failed", model]
        scope.set_tag("model", model)
        scope.set_tag("reason", reason)
        scope.set_tag("job_id", job_id)
        if detail:
            scope.set_extra("detail", detail)
        sentry_sdk.capture_message(f"job failed on {model}: {reason}", level="error")
