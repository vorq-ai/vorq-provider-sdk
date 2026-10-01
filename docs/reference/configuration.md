---
title: Configuration
description: Every key of vorqd.yaml — the provider block, model entries, backend limits and load-time validation.
---

`vorqd.yaml` has two top-level keys: `provider` and `models`. Any string value may contain
`env:NAME`, which is replaced by the environment variable `NAME` at load. A referenced variable
that is not set stops the daemon with an error naming it.

```yaml
provider:
  wallet_key: env:VORQ_WALLET_KEY
  box_key: env:VORQ_BOX_KEY
  api_url: https://api.vorq.co
  capacity: 4

models:
  - model: deepseek-ai/deepseek-v4-pro:fp8
    modality: text
    slas:
      "24h": { rate_in: "0.16", rate_out: "0.55" }
    backend:
      preset: openai-chat
      base_url: http://localhost:8000/v1
      model: deepseek-ai/DeepSeek-V4-Pro
```

## provider

| Key | Default | Meaning |
|---|---|---|
| `api_url` | `https://api.vorq.co` | Base URL of the VORQ coordinator. |
| `wallet_key` | `$VORQ_WALLET_KEY` | Operator wallet private key (secp256k1, hex). Signs the session and every operation. The daemon exits if no key is available. |
| `box_key` | required; forbidden when confidential | Curve25519 private key (64 hex characters) that opens designated payloads. Its public half must match your provider record. Must be absent when any model sets `confidential: true`. |
| `capacity` | derived | Jobs the daemon requests to hold at once; a positive integer. Unset, it is the sum over models of each entry's `rate_limit` window at its shortest SLA, else its `concurrency`; an entry with neither makes it required. |
| `metrics_port` | `9090` | Port of the ops server (`/healthz`, `/metrics`), bound on all interfaces. |
| `log_level` | `info` | `debug`, `info`, `warn` or `error`. |
| `poll_interval_s` | `5` | Seconds between poll sweeps. |
| `safety_margin_s` | `60` | Seconds before the SLA deadline after which no attempt runs and no settle is tried. |
| `fail_grace_s` | `300` | Window after a claim in which a `fail` is penalty-free, mirroring the job registry. Used for logging only. |
| `max_input_bytes_per_unit` | `16` | For `text` and `embedding` bids with a nonzero `rate_in`: most payload bytes one declared input unit may buy (plus 4096 bytes of slack). Larger bids are not claimed. `0` disables the check. |
| `state_db` | `vorqd-state.sqlite` | SQLite file for the handles of jobs in flight at resumable backends. Relative paths resolve against the working directory. Must be writable. |
| `escrow_url` | `api_url` | Base URL of the escrow that releases payload keys for open bids. |
| `pricing` | off | Private acceptance floor. See [pricing](#pricing). |
| `bid_filter.min_age_s` | none | Accepted for compatibility and ignored, with a warning at startup. |

There is no provider id key: the daemon learns its id at sign-in.

### pricing

| Key | Default | Meaning |
|---|---|---|
| `max_discount_pct` | `0` | Discount at load 0, shrinking linearly to 0 at load 1. Integer, 0–90. |
| `bid_tolerance_pct` | `0` | Extra discount while load is below `low_load_pct`. Integer, 0–90. |
| `low_load_pct` | `30` | Load threshold in percent, 0–100. |

The total discount is capped at 90%. It applies only to models with a `load` block. Unknown
keys are refused.

## Model entry

Each item of `models`:

| Key | Default | Meaning |
|---|---|---|
| `model` | required | Network model name clients submit against. Must be in the coordinator's catalog. |
| `slas` | required | Map of SLA window to `{rate_in?, rate_out}`. See [slas](#slas). |
| `backend` | required | How to run a job: exactly one of `preset` ([presets](./presets.md)) or `request` ([raw mappings](./raw-mappings.md)), plus the [shared backend keys](#backend-keys). |
| `modality` | none | `text`, `embedding`, `image` or `video`. Used when the catalog names none. The daemon claims nothing for a model whose modality is unknown. |
| `confidential` | `false` | `true` makes the daemon a confidential provider. See [Run a confidential provider](../guides/run-a-confidential-provider.md). |
| `load` | none | Load source for dynamic pricing. See [load](#load). A sibling of `backend`; a `load` key inside `backend` is refused. |
| `sla_backends` | none | Map of SLA window to a backend patch. See [sla_backends](#sla_backends). |

### slas

- Keys are windows: a whole number and a unit `s`, `m`, `h` or `d` (`"1h"`, `"24h"`).
- `rate_out` is required on every window; `rate_in` is optional.
- Rates are USD per 10<sup>6</sup> units of work, as quoted decimal strings (`"0.16"`): digits,
  optionally one point and fraction digits, no sign or exponent. An unquoted YAML number is
  refused. A rate with more fraction digits than the payment token's decimals is refused when
  the daemon reads the chain, never rounded.
- One ask is published per window. The rates are also the claim floor for that window.
- At most 64 asks (windows across all models) fit in one ask book.

### load

Either a Prometheus probe:

| Key | Default | Meaning |
|---|---|---|
| `url` | required | Absolute `http(s)` URL of a Prometheus text endpoint. |
| `metric` | required | Series name. `:` and `_` are treated as equal. The busiest sample wins. |
| `scale` | `1.0` | Positive divisor that maps the series to 0..1. |

or `{source: occupancy}`, which reads the entry's own fullness and takes no other key. It needs
`concurrency` or `rate_limit` on the backend.

The probe is scraped once per sweep with a 2-second timeout. A reading older than 30 seconds,
a failed scrape, or a series with no finite non-negative sample counts as unknown, which earns
no discount.

### sla_backends

Each value is merged key by key over `backend` and validated as a backend of its own. Patch
keys win; `null` removes a key; `preset` in a patch drops an inherited `request`/`response`,
and `request` drops an inherited `preset`. Windows must exist in `slas`. Limit keys are refused
in a patch. The model's asks are published only while every one of its backends passes its
health check.

## Backend keys

These keys are valid on every `backend` block, preset or raw.

### health

| Key | Meaning |
|---|---|
| `health.path` | Probe URL, requested with `GET` once per sweep. A path starting with `http` is used as is; otherwise it is appended to `base_url` (presets) or to the scheme and host of `request.url` (raw mappings). A status below 400 is healthy. While unhealthy, the model's asks are withdrawn. |

### Backend limits

| Key | Default | Meaning |
|---|---|---|
| `retries` | `0` | Attempts after the first. A `429`, any `5xx`, a transport error or timeout, and statuses in `retry_statuses` are retried; any other `4xx` fails the job. Non-negative integer. |
| `retry_statuses` | none | Extra `4xx` codes to retry. |
| `retry_backoff_s` | `30` | First retry delay, doubling after each retry. A `429`'s `Retry-After` (in seconds) wins when longer. Positive number. |
| `retry_backoff_max_s` | `900` | Ceiling on one delay. Must be at least `retry_backoff_s`. |
| `timeout_s` | remaining SLA budget | Read timeout of one attempt, never beyond `deadline − safety_margin_s`. With `stream: true`, the longest silence between chunks. |
| `concurrency` | unbounded | Most requests in flight for this model. Positive integer. |
| `rate_limit` | none | Map of window to the most attempts started in it, retries included. The window equal to the shortest SLA is also how many jobs the entry may hold. The loader refuses a hold budget that the shorter windows cannot start inside the SLA. |
| `trip_after` | `3` | Consecutive jobs failed with the backend at fault before the model's asks are withdrawn. `0` disables. |
| `trip_cooldown_s` | `60` | Seconds the model stays withdrawn after tripping. Positive number. |

No wait or attempt is scheduled past `deadline − safety_margin_s`; the job is failed back
instead. `queue` is not a key and is refused.

### Media policy

`param_caps`, `reference`, `resolutions`, `durations`, `auto_duration` and `adaptive_aspect`
apply to image and video models on any backend shape. See
[Raw mappings: media policy](./raw-mappings.md#media-policy).

## Load-time validation

The daemon validates the whole file before it contacts anything and exits with status 2 and a
message naming the key and model on the first violation. Beyond the types and ranges above:

- `models` is a non-empty list; each `backend` holds exactly one of `preset` or `request`.
- A raw `response.result` holds exactly one of `text`, `media_urls` or `media_b64`, and a
  `text` result also maps `completion_tokens`.
- A raw poll holds exactly one of `status_url` or `request`; `handle` requires `request`, and
  a poll request with `handle` cannot use `{submit.…}`. A poll request using `{handle}` needs
  `handle`.
- `max_polls` is a positive integer.
- `stream` is a boolean and `true` only on `openai-chat`.
- `headers` on a preset is a string map and is refused on `openai-batch`.
- `service_tier` is `flex` or `priority`; `completion_window` is a non-empty string;
  `endpoint` starts with `/`.
- `param_map`, `reasoning`, `param_caps` and `params_supported` follow the shapes in
  [Presets](./presets.md).
- `prepare` steps, `response.ok`, `poll.failure_code` and `poll.failure_message` follow the
  shapes in [Raw mappings](./raw-mappings.md).
- `reference`, `resolutions`, `durations`, `auto_duration` and `adaptive_aspect` follow the
  shapes in [media policy](./raw-mappings.md#media-policy).

Further checks run as the daemon starts, and stop it on failure: `preset` must be one of
`openai-chat`, `openai-embeddings`, `openai-responses` or `openai-batch`; every model must be in
the coordinator's catalog; and a regular provider's box key must match its provider record.

## Examples

- [`vllm.yaml`](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/docs/examples/vllm.yaml):
  a self-hosted OpenAI-compatible text model.
- [`queue-backend.yaml`](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/docs/examples/queue-backend.yaml):
  an image model on a submit-then-poll API.
- [`confidential.yaml`](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/docs/examples/confidential.yaml):
  a confidential provider.
