---
title: Log events
description: The vorqd log format and the events it writes.
---

`vorqd` writes one JSON object per line to stderr, at `provider.log_level` (default `info`).

| Field | Meaning |
|---|---|
| `level` | `debug`, `info`, `warning` or `error`. |
| `logger` | Always `vorqd`. |
| `event` | The message. |
| `job_id` | The job, when the line is about one. |
| `model` | The network model name (not `backend.model`), when the line is about one. |
| `result_cid` | The result's content id, on `settled`. |
| `exc` | The traceback, on lines logged with an exception. |

Rendered backend requests are never logged, at any level. Backend responses appear only in
truncated form, in `warning` lines about a failed call.

## Job events

| Event | Level | When |
|---|---|---|
| `claimed` | info | A claim landed and the SLA clock started. |
| `backend_submitted` | info | A resumable backend accepted the job and its handle was recorded. |
| `backend_wait` | info | An attempt waits for a `rate_limit` window. |
| `backend_queued` | info | An attempt waits for a `concurrency` slot. |
| `backend_retry` | info | An attempt failed retryably and another is scheduled. |
| `settled` | info | The settle landed. |
| `abandoned (SLA)` | warning | The backend finished after the SLA margin; the job is left to be reclaimed. |
| `abandoned (backend): …` | warning | The backend could not complete the job; it was failed back. |
| `refusing claimed job (<reason>): …` | warning | A claimed job was handed back before running. |
| `settle refused (<reason>)` | warning | The chain refused the settle. |
| `job not settled: …` | warning | The settle could not be delivered; the job was failed back. |
| `could not report failure` | warning | A `fail` could not be sent; the refund falls back to reclaim at SLA expiry. |
| `suspended` | info | On shutdown, a resumable job was parked for the next start. |
| `recovered`, `resumed` | info | On the first sweep after a start, a job claimed earlier was re-run or its poll resumed. |
| `job <id> crashed` | error | An unexpected error ended a job. |

## Claim decisions

| Event | Level | When |
|---|---|---|
| `not claiming <job>: its declared N input units do not cover the payload …` | info | The bid is oversized for its declared input. |
| `not claiming <job>: …` | info or warning | The payload could not be fetched or does not match the job. |
| `claim would not land for <job> (…)` | info | The claim simulation refused. |
| `lost race for job <job> (…)` | info | Another provider claimed it first. |
| `claim rejected for <job>: …` | warning | The registries did not recognise the signer. |

## Daemon events

| Event | Level | When |
|---|---|---|
| `not registered with the coordinator; waiting for admin provisioning` | info | The wallet is not registered yet; the daemon retries. |
| `the network grants N slots` | info | The granted capacity changed and covers `capacity`. |
| `the network grants N of the M slots configured; …` | warning | The grant is below `capacity`. |
| `model X is not permitted by the network; excluded from asks` | warning | The provider record does not allow a configured model. |
| `model X has no known modality …` | warning | Nothing is claimed for the model this sweep. |
| `backend tripped after N consecutive faults; asks withdrawn for Ns` | warning | The breaker withdrew a model. |
| `load probe … failed …`, `load probe … carries no usable sample …` | warning | A load probe produced no reading. |
| `poll failed for model X; will retry` | warning | The poll for one model failed. |
| `the model catalog could not be read (…); nothing is polled this sweep` | warning | The catalog read failed. |
| `provider.bid_filter.min_age_s is ignored …` | warning | The config sets an inert key. |
