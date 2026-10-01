---
title: Presets
description: The four built-in backend presets, their params, the forwarded param sets, param mapping and reasoning dialects.
---

A preset is a built-in mapping for an OpenAI-compatible endpoint. Every preset sends
`Content-Type: application/json` and, when `api_key` is set, `Authorization: Bearer <api_key>`.
Every preset also takes the [shared backend keys](./configuration.md#backend-keys).

## openai-chat

`POST {base_url}/chat/completions`, synchronous.

| Param | Default | Meaning |
|---|---|---|
| `base_url` | required | Runtime base URL, e.g. `http://localhost:8000/v1`. |
| `model` | required | Runtime's own model name, sent as `model`. |
| `api_key` | none | Bearer token. |
| `params_supported` | standard set | Exact list of params to forward. Replaces the standard set; may name nonstandard params. `[]` forwards none. |
| `param_map` | none | How forwarded params are renamed or value-mapped. See [param mapping](#param-mapping). |
| `reasoning` | none | Reasoning dialect. See [reasoning dialects](#reasoning-dialects). |
| `param_caps` | none | Map of param to numeric ceiling, applied to image and video jobs. |
| `extra_params` | none | Object merged into the body last; overrides any key, `model` and `max_tokens` included. |
| `headers` | none | Extra headers, templated per job (e.g. `{ X-Request-Id: "{job.id}" }`), merged over the preset's own. |
| `stream` | `false` | Stream the completion with `stream_options.include_usage` and reassemble it. `timeout_s` then bounds the silence between chunks. A stream that ends without a finish reason or usage block, or reports an error mid-stream, is retried. |

**Body.** The job's `messages` are forwarded. If the job has `input` instead, a string becomes
one user message and a list is used as the messages. Params in the forwarded set are added
through `param_map`; `model` is set to the runtime's name, `max_tokens` to the job's output
cap, then `extra_params` is merged.

**Standard forwarded set:** `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`,
`max_tokens`, `min_tokens`, `seed`, `stop`, `reasoning_effort`, `reasoning_max_tokens`,
`frequency_penalty`, `presence_penalty`, `logit_bias`, `logprobs`, `top_logprobs`,
`response_format`, `tools`, `tool_choice`, `parallel_tool_calls`. Other params are dropped and
logged at `info`.

**Never forwarded**, whatever `params_supported` says: `n`, `best_of`, `stream`,
`stream_options`, `service_tier` (they break single-choice, usage-bearing billing at the priced
tier), and `user`, `metadata` (caller identifiers that would link a client's jobs).

**Clamps.** `reasoning_max_tokens` is clamped to below the job's output cap and `min_tokens`
to at most the cap. A value that is not a non-negative integer is dropped.

**Result.** The response object is returned as is. The settled count is
`usage.completion_tokens`; a response without it fails the job.

## openai-embeddings

`POST {base_url}/embeddings`, synchronous.

| Param | Default | Meaning |
|---|---|---|
| `base_url`, `model`, `api_key`, `headers` | | As for `openai-chat`. |
| `input_type` | none | `query` or `passage`: the side an asymmetric retrieval model embeds into. |
| `input_type_overridable` | `false` | Let a caller's `input_type` (`query` or `passage`) replace the configured one. Other values fall back to the configured one. |

**Body:** `model`, the job's `input`, `encoding_format` (the job's, default `base64`),
`dimensions` if the job names one, and `input_type` when one applies. Chat params are not
forwarded. The response object is returned as is, and the job settles with no output count.

## openai-responses

`POST {base_url}/responses` with `background: true`, then `GET {base_url}/responses/{id}`
until `completed`. `failed`, `cancelled` and `incomplete` fail the job. Resumable after a
restart.

| Param | Default | Meaning |
|---|---|---|
| `base_url`, `model`, `api_key`, `params_supported`, `param_map`, `reasoning`, `param_caps`, `extra_params` | | As for `openai-chat`. |
| `headers` | none | As for `openai-chat`; sent on the submit only. |
| `service_tier` | none | `flex` or `priority`, sent as `service_tier`. |
| `max_polls` | `60` | Status polls per SLA window: interval = `sla / max_polls`, at least 1 second. |

**Body.** Built as for `openai-chat`, then respelled:

| Chat field | Responses field |
|---|---|
| `messages` | `input` |
| `max_tokens` | `max_output_tokens` |
| `reasoning_effort` | `reasoning.effort` (merged into an existing `reasoning`) |
| `response_format` | `text.format` (merged into an existing `text`; a `json_schema` wrapper is flattened) |
| function `tools` and `tool_choice` | `{type: function, …}` without the `function` nesting |

`extra_params` is respelled the same way and merged last; `reasoning` and `text` objects merge
key by key. A job with no messages or input fails before any request.

**Default forwarded set:** `temperature`, `top_p`, `max_tokens`, `stop`, `frequency_penalty`,
`presence_penalty`, `reasoning_effort`, `response_format`, `tools`, `tool_choice`,
`parallel_tool_calls`.

**Result.** The finished object is converted to a chat completion: `choices[0].message.content`
joins the `output_text` parts, `reasoning_content` carries the reasoning summary (or the raw
reasoning text), `function_call` items become `tool_calls` with `finish_reason: "tool_calls"`,
and `usage.output_tokens` becomes `usage.completion_tokens`.

## openai-batch

One job is one batch of one line. The daemon uploads the chat-completions body as a one-line
JSONL file (`POST {base_url}/files`, `purpose=batch`, `custom_id` = the job id without `0x`,
at most 64 characters), creates the batch (`POST {base_url}/batches`), and polls
`GET {base_url}/batches/{id}` until `completed`. `failed`, `cancelled` and `expired` fail the
job. The line is read from `GET {base_url}/files/{output_file_id}/content`; a non-200 line, a
line in the error file, or no line fails the job without a retry. Resumable after a restart.

| Param | Default | Meaning |
|---|---|---|
| `base_url`, `model`, `api_key`, `params_supported`, `param_map`, `reasoning`, `param_caps`, `extra_params` | | As for `openai-chat`; the line body is a chat-completions body. |
| `completion_window` | `"24h"` | Window named on the batch. Use one no longer than the SLA window it serves. |
| `endpoint` | `"/v1/chat/completions"` | Path each line names and the batch is created for. |
| `max_polls` | `60` | As for `openai-responses`. |

`headers` and `stream` are refused. The input file is deleted after collection (best effort);
output and error files are kept. A retried submit repeats both the upload and the create, so
keep `retries: 0` unless a duplicate batch is acceptable.

## Param mapping

`param_map` maps a canonical param name to how it lands in the body.

- **String:** rename. A dot nests one level: `chat_template_kwargs.reasoning_budget` produces
  `{"chat_template_kwargs": {"reasoning_budget": …}}`.
- **Object** `{to, values, unmapped}`: `to` is the target key (default: the same name, dotted to
  nest). `values` maps the client's exact value to what is sent; a `null` mapping drops the
  param. `unmapped` is `pass` (default: send the value unchanged) or `drop`.

When a mapped value is an object and the target already holds one, they merge, so several
params can build one backend object.

## Reasoning dialects

Clients use `reasoning_effort` (`none`, `low`, `medium`, `high`, `xhigh`, `max`) and
`reasoning_max_tokens`. `reasoning` maps them onto the backend's dialect, and resolves to a
`param_map`; an inline `param_map` overrides it key by key.

**Mapping form:** `reasoning: { efforts: [...], budget: <field>, default_effort: <effort> }`

| Key | Meaning |
|---|---|
| `efforts` | Levels the backend accepts, in canonical terms. `[]`: no reasoning control; both params are dropped. |
| `budget` | The backend's thinking-budget field. Without it, `reasoning_max_tokens` is dropped. |
| `default_effort` | Effort injected when a job carries no reasoning control. Not allowed with `efforts: []`. |

A requested level maps to the nearest accepted level not above it; below the lowest accepted
level, it rises to that level. If `none` is not accepted, `none` rises to the lowest level.

When a `budget` field exists and a job sends an effort without a budget, the budget is derived
from the job's output cap: `low` 20%, `medium` 50%, `high` 80%, `xhigh` and `max` 95%, at least
1,024 and at most 128,000 tokens, and always below the cap. `none` derives nothing.

**Named dialects** for chat-template switches:

| Name | Sends |
|---|---|
| `thinking_bool` | `chat_template_kwargs.enable_thinking`: `false` for `none`, `true` otherwise. |
| `thinking_bool_low_effort` | As above plus `low_effort`: `true` for `low` and `medium`, `false` for `high`, `xhigh`, `max`. |

Both drop `reasoning_max_tokens`. Use `reasoning: thinking_bool` or, with a default,
`reasoning: { preset: thinking_bool, default_effort: medium }`.
