---
title: Raw mappings
description: The request and response fields of a raw backend mapping, the template language, media results and the media policy keys.
---

A raw mapping describes a backend call directly. It takes the
[shared backend keys](./configuration.md#backend-keys) too.

```yaml
backend:
  request:
    method: POST
    url: https://queue.example.com/v1/generate
    headers: { Authorization: "Key env:QUEUE_BACKEND_KEY" }
    body: { prompt: "{input.prompt}", num_images: "{input.num_images}" }
  response:
    mode: poll
    poll:
      status_url: "$.status_url"
      status_field: "$.status"
      done_values: [COMPLETED]
      failed_values: [FAILED]
      interval_s: 2
    result:
      media_urls: "$.images[*].url"
```

## request

| Key | Meaning |
|---|---|
| `method` | HTTP method. Default `POST`. |
| `url` | Templated URL. |
| `headers` | Templated header map. |
| `body` | Templated JSON object or top-level array. Not sent with `GET`. |
| `prepare` | Requests to run before the submit. See [prepare](#prepare). |

## prepare

A list of steps, run in order before the submit. Each step is a templated request.

| Key | Meaning |
|---|---|
| `name` | Required. A plain identifier, unique among the steps. |
| `url` | Required. |
| `method`, `headers`, `body` | As in `request`. |
| `extract` | Required. Map of key to a `$.` JSONPath into the step's answer. Available later as `{prepare.<name>.<key>}`. |
| `when` | Dot-path such as `input.image`. The step is skipped when it resolves to nothing or `null`. |
| `for_each` | Dot-path to a list such as `input.reference_images`. The step runs once per element, in scope as `{item}`, and `{prepare.<name>}` is the list of results. Replaces `when`. |

A step that runs but whose answer lacks an `extract` value fails the job. Steps are not run
again when a poll is resumed after a restart.

## response

| Key | Meaning |
|---|---|
| `mode` | Required. `sync`: the submit answer is the result. `poll`: poll until done. |
| `poll` | Required with `mode: poll`. See [poll](#poll). |
| `result` | Required. See [result](#result). |
| `ok` | For backends that report failure in the body. See [ok](#ok). |

### poll

| Key | Meaning |
|---|---|
| `status_url` | JSONPath into the submit answer yielding a URL to `GET`. Exactly one of `status_url` or `request`. |
| `request` | Templated poll request. `{submit.…}` reads the submit answer. |
| `handle` | With `request` only. JSONPath into the submit answer naming the job at the backend, in scope as `{handle}`. Recorded in `state_db` so a restart resumes the poll. A poll request with `handle` cannot use `{submit.…}`. |
| `status_field` | JSONPath into each poll answer giving the status. |
| `done_values` | Statuses that end the poll with the result. |
| `failed_values` | Statuses that fail the job. |
| `failure_code` | JSONPath to the backend's error code on a failed status; added to the failure reason. |
| `failure_message` | JSONPath to the backend's error message on a failed status; logged only. |
| `interval_s` | Fixed seconds between polls. |
| `max_polls` | Polls per SLA window when `interval_s` is unset: interval = `sla / max_polls`, at least 1 second. Default `60`. |
| `timeout_s` | Give up polling after this many seconds (retryable). Default: the job's deadline. |

A poll tick that fails with a retryable error is absorbed and polled again; it never causes a
second submit.

### result

Exactly one of `text`, `media_urls` or `media_b64`.

| Key | Meaning |
|---|---|
| `text` | JSONPath to the output text. Requires `completion_tokens`. |
| `completion_tokens` | JSONPath to the output token count that is settled. |
| `media_urls` | JSONPath to output URLs (absolute `http(s)`); the daemon downloads each. |
| `media_b64` | JSONPath to base64 outputs; the daemon decodes each. |
| `content_type` | Media only. Scalar JSONPath to the output media type. |
| `seed` | Media only. JSONPath to the seed the backend used. Default: the request's seed. |

A media result's type is, in order: the `content_type` value, the type detected from the bytes,
the type the file was served with, `application/octet-stream`.

### ok

```yaml
response:
  ok: { field: "$.code", values: [200], retry: [429, 503], message: "$.msg" }
```

`field` is a JSONPath evaluated on every JSON answer (prepare steps, submit, poll ticks);
`values` is the non-empty list meaning success; `retry` lists values that are retried; `message`
is a JSONPath logged, never sent. Other values fail the job. An answer without the field passes.

## Template language

Strings in `request`, `prepare` and `poll.request` are rendered per job.

| Token | Value |
|---|---|
| `{input.<path>}` | A field of the job's decoded input, e.g. `{input.prompt}`, `{input.image.b64}`. |
| `{job.id}`, `{job.owner}` | The job id and the address that posted it. |
| `{job.units_out}` | The job's output cap in its unit (tokens, pixels or pixel-seconds). |
| `{job.modality}` | The job's modality. |
| `{submit.<path>}` | The submit answer, in a poll request. |
| `{handle}` | The recorded handle, in a poll request. |
| `{prepare.<name>.<key>}` | A value extracted by a prepare step. |
| `{item}` | The current element, in a `for_each` step. |
| `{uuid}` | A new random UUIDv4. |

- A string that is exactly one token keeps the value's JSON type; embedded tokens are
  stringified.
- A trailing `?` (`{input.end_image.b64?}`) marks a token as optional. If it resolves to nothing,
  the key or list element holding it is omitted. An embedded optional token that is missing
  drops the whole string. A list whose elements all drop is omitted.
- A token without `?` that resolves to nothing fails the job, naming the token.
- A wildcard token (`{prepare.clips[*].url}`) renders a list; as a list element it is spliced in.
- `env:NAME` works inside any string and is resolved at load.
- Presets also expose their params (`{base_url}`, `{model}`, `{api_key}`) in `headers`. Raw
  mappings have no such params.

**JSONPath.** `$.` values are evaluated against answers. A wildcard (`[*]`) returns a list,
otherwise the first match. A path that continues past a string field parses that string as
JSON, for APIs that return JSON inside a string:

```yaml
# { "data": { "result": "{\"urls\": [\"https://…/out.mp4\"]}" } }
media_urls: "$.data.result.urls[*]"
```

## Media results

The sealed result of an image job:

```json
{"images": [{"b64": "…", "content_type": "image/png", "width": 1024, "height": 1024}],
 "units": 1048576, "seed": 42, "vorq": {"job_id": "0x…"}}
```

A video job carries one `video` object with `duration_secs` instead of `images`. `width`,
`height` and `duration_secs` are read from the delivered file's header (PNG, JPEG, WebP, MP4);
if it cannot be read, the priced dimensions are used. `units` is the settled count: delivered
pixels (times seconds), capped at the order's output limit.

## Media policy

These keys sit on the `backend` block (preset or raw) and apply to image and video jobs. Every
refusal happens before any upload or submit, and hands the job back for a refund.

| Key | Meaning |
|---|---|
| `param_caps` | Map of param to numeric ceiling, e.g. `{steps: 40, fps: 24}`. |
| `resolutions` | Non-empty list of tiers the backend renders, from `480p`, `720p`, `1080p`, `4k`. A request naming another tier, or raw pixels, is handed back. |
| `durations` | Non-empty list of whole-second clip lengths. A clip is lowered to the longest listed length its order covers, or handed back if none fits. |
| `auto_duration` | Value sent as the duration when the client asks the model to choose. Without it such requests are handed back. |
| `adaptive_aspect` | Value sent as `aspect_ratio` when the client asks to keep the reference's shape. Without it such requests are handed back. |
| `reference.accept` | Non-empty list narrowing the reference types taken, from `image/png`, `image/jpeg`, `image/webp`, `video/mp4`, `audio/mpeg`, `audio/wav`. Omit `reference` to take all. |
| `reference.still`, `reference.clip`, `reference.audio` | Bounds per kind of reference: `min_side`, `max_side`, `min_ratio`, `max_ratio` (width ÷ height), `min_pixels`, `max_pixels`, `min_secs`, `max_secs`, `max_total_secs`, `max_bytes`, `max_count`. Audio is only bounded by `max_bytes` and `max_count`. |

`image`, `end_image` and `reference_images` are stills; `video` and `reference_videos` are
clips; `reference_audios` is audio. Each reference is decoded and must have no more pixels and
no longer runtime than the client declared, and together they must not exceed the order's input
units. Areas are compared, not sides, so a rotated frame is accepted. Clip lengths are rounded
up to whole seconds after a 100 ms allowance.
