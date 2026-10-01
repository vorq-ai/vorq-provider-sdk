---
title: Monitor the daemon
description: Wire up the health endpoint, scrape metrics, read the logs and report errors to Sentry.
---

## Health

`GET /healthz` on `provider.metrics_port` (default `9090`) answers `200 ok` while the daemon is
running and holds a valid coordinator session, and `503 unhealthy` otherwise: the session
cannot be renewed (the coordinator is unreachable or refuses the wallet), or the daemon is
shutting down. Use it as the liveness probe of your supervisor or orchestrator.

The daemon exits if the coordinator or its model catalog cannot be reached at startup, so run
it under something that restarts it.

Per-model backend health is separate: a failing `backend.health` probe or a tripped breaker
withdraws that model's asks without affecting `/healthz`. Watch `vorqd_asks_published` for that.

## Metrics

`GET /metrics` serves Prometheus metrics. Useful signals:

| Symptom | Likely cause |
|---|---|
| `vorqd_jobs_failed_total{reason="sla_abandon"}` or `{reason="deadline_wait"}` rising | You claim more than the backend finishes inside the SLA. Lower `capacity` or the entry's limits. |
| `vorqd_capacity_free` pinned at `0` | Fully used. You may be leaving work on the table. |
| `vorqd_capacity_granted` below your `capacity` | The network grants fewer slots than you asked for. The grant grows with reputation. See [Capacity](../concepts/capacity.md). |
| `vorqd_asks_published` drops | A backend health check is failing or a breaker tripped. |
| `vorqd_model_free{model}` at `0` while `vorqd_capacity_free` is not | That model's own limits are full. |
| `vorqd_jobs_failed_total{reason="backend_exhausted"}` rising | The backend fails every attempt. Check its logs. |
| `vorqd_jobs_failed_total` with a `settle_…` or `session_http_…` reason | Settles are not landing. See the reason list. |

Every metric and failure reason is listed in the [metrics reference](../reference/metrics.md).

Port `9090` has no authentication, and `vorqd_bid_floor_discount_pct` shows how far under your
published price you will go. Do not expose it publicly.

## Logs

The daemon writes one JSON object per line to stderr, at `provider.log_level` (default
`info`). Lines about a job carry its `job_id`, so you can follow one job end to end:

```bash
docker logs vorqd 2>&1 | grep '"0x7c65'
```

Alert on:

- a `claimed` with no `settled`, `abandoned (SLA)`, `abandoned (backend): …` or
  `refusing claimed job (…)` after the job's SLA window: a stuck job;
- `backend tripped after N consecutive faults`;
- `the network grants N of the M slots configured`.

See the [log events reference](../reference/logs.md).

## Error reporting

Set `SENTRY_DSN` to send errors to Sentry. Every `error` log line and every unhandled
exception becomes an event, with preceding `info` and `warning` lines attached as
breadcrumbs. `SENTRY_ENVIRONMENT` and `SENTRY_RELEASE` tag the events. Stack frames carry no
local variables, so keys never leave the process, and no trace headers are added to backend
requests. Without `SENTRY_DSN` nothing is sent.

## Related

- [Metrics reference](../reference/metrics.md)
- [Log events reference](../reference/logs.md)
