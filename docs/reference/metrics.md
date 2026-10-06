---
title: Metrics
description: Every Prometheus metric vorqd exports and every failure reason it reports.
---

`GET /metrics` on `provider.metrics_port` serves these metrics in Prometheus text format.
`GET /healthz` answers `200 ok` or `503 unhealthy`.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `vorqd_jobs_claimed_total` | counter | | Jobs claimed. |
| `vorqd_jobs_settled_total` | counter | | Jobs settled. |
| `vorqd_jobs_failed_total` | counter | `reason` | Jobs that ended without a settle. See [failure reasons](#failure-reasons). |
| `vorqd_job_duration_seconds` | histogram | | From the start of a job's run to its settle. |
| `vorqd_backend_latency_seconds` | histogram | `model` | Time in the backend phase, including waits and retries, for jobs the backend completed. |
| `vorqd_capacity_free` | gauge | | Slots (capacity or grant, whichever is smaller) minus jobs held. |
| `vorqd_capacity_granted` | gauge | | Slots the network grants this provider. |
| `vorqd_asks_published` | gauge | | Asks currently on the book. |
| `vorqd_model_free` | gauge | `model` | The `free` sent on the model's last poll. |
| `vorqd_backend_load` | gauge | `model` | Last load reading, 0..1. Only for models with a `load` block. |
| `vorqd_bid_floor_discount_pct` | gauge | `model` | Current discount under the published ask, in percent. `0` means claims only at the configured rates. |
| `vorqd_load_probe_failures_total` | counter | `model` | Load probe scrapes that produced no reading. |
| `vorqd_backend_retries_total` | counter | `model` | Backend attempts retried. |

## Failure reasons

Values of the `reason` label on `vorqd_jobs_failed_total`. Except where noted, the daemon also
sends a signed `fail`, which refunds the client immediately.

**Backend and SLA**

| Reason | Meaning |
|---|---|
| `backend_error` | The backend refused the job (a non-retryable error). |
| `backend_exhausted` | Every attempt failed. |
| `backend_gone` | The backend answered `404` or `410`: the endpoint is not there. Not retried. |
| `deadline_wait` | The next wait or attempt would end past the SLA margin. |
| `media_input_refused` | A media request or reference did not fit what was paid for or what the backend takes. |
| `sla_abandon` | The backend finished after the SLA margin. No `fail` is sent; the job is left to be reclaimed. |
| `internal_error` | An unexpected error in the daemon. No `fail` is sent. |

**Before running** (the payload or its key)

| Reason | Meaning |
|---|---|
| `too_short`, `bad_version`, `commitment_mismatch`, `bad_container` | The payload is not the container the job commits to. |
| `undecryptable` | The payload did not decrypt. |
| `malformed_envelope` | Wrong envelope version, no input, or a missing or invalid result key. |
| `owner_mismatch` | The envelope names a different owner than the job. |
| `units_in_short` | The payload is larger than the declared input units allow (jobs recovered after a restart). |
| `unseal_failed` | The daemon's box key could not open the payload key. |
| `not_our_bid`, `no_cipher` | The job is designated to another provider, or no box key is loaded. |
| `escrow_unavailable`, `escrow_unreachable` | No escrow client, or the escrow did not answer. |
| Escrow refusal codes | Reported verbatim, e.g. `escrow_key_lost`, `not_claimed`, `wrong_wallet`, `stale_issued_at`. |
| `modality_unknown` | The claimed job's modality could not be established. |
| `task_unresolved` | A recovered job's payload could not be fetched. |
| `claim_expired` | A recovered job's SLA window had closed. |
| `unservable` | A recovered job is for a model this daemon no longer serves. |

**Settlement**

| Reason | Meaning |
|---|---|
| `settle_<reason>` | The chain refused the settle, e.g. after the deadline. No `fail` is sent: the job is no longer this provider's. |
| `settle_rejected` | The registries did not recognise the signer. |
| `settle_reverted` | The settle was mined but reverted, on every retry until the SLA closed. |
| `settle_upload_invalid` | The result upload answered without a content id. |
| `settle_result_too_large` | The result exceeds the coordinator's size limit (`413`). |
| `settle_http_<status>` | The coordinator refused the settle with that HTTP status. A `429` or `5xx` is retried until the SLA closes. |
| `session_http_<status>` | Sign-in failed, so the settle was never sent. |
| `settle_transport_error` | The coordinator did not answer, on every retry until the SLA closed. |
