---
title: Connect an OpenAI-compatible backend
description: Serve a chat or embeddings model from vLLM or any OpenAI-compatible server through a preset.
---

A runtime that speaks the OpenAI chat-completions or embeddings wire format needs no mapping
of its own. Point a preset at it and decide which request fields reach it.

## Serve a chat model

```yaml
models:
  - model: deepseek-ai/deepseek-v4-pro:fp8     # the network name clients submit against
    modality: text
    slas:
      "1h": { rate_in: "0.22", rate_out: "0.75" }
    backend:
      preset: openai-chat
      base_url: http://localhost:8000/v1
      model: deepseek-ai/DeepSeek-V4-Pro       # your runtime's own model name
      api_key: env:RUNTIME_API_KEY              # optional; sent as a Bearer token
      health: { path: http://localhost:8000/health }
```

The daemon sends `POST {base_url}/chat/completions` with the job's messages, your `model`,
and `max_tokens` set to the job's output cap. The whole response object, usage block included,
is sealed and returned to the client. The settled output count is `usage.completion_tokens`.

## Gate the asks on a health check

`health.path` is probed once per poll sweep. A `2xx` or `3xx` answer is healthy; anything
else, or no answer, takes the model's asks off the order book until the probe passes again.
Jobs already claimed keep running.

A path that starts with `http` is used as is. Any other path is appended to `base_url`, so
`{ path: /health }` with `base_url: http://localhost:8000/v1` probes
`http://localhost:8000/v1/health`. vLLM serves its health endpoint at the server root, so give
it the absolute URL.

## Choose which params reach the runtime

By default the preset forwards a standard set of chat-completions params and strips everything
else (see [the forwarded set](../reference/presets.md#openai-chat)). If your runtime publishes
an exact schema for the model, declare it. The list replaces the default set, and may include
the runtime's own knobs:

```yaml
    backend:
      preset: openai-chat
      base_url: http://localhost:8000/v1
      model: deepseek-ai/DeepSeek-V4-Pro
      params_supported: [temperature, top_p, max_tokens, seed, stop, reasoning_effort]
```

A field the served model does not accept can make the backend reject the request, and a
rejected request is a job you fail at your own cost. Declaring the schema avoids that.

To send a param under the runtime's own spelling, map it:

```yaml
      param_map:
        reasoning_max_tokens: chat_template_kwargs.reasoning_budget
```

To pin a field on every request, set it in `extra_params`. It is merged last and overrides
anything else with the same name:

```yaml
      extra_params: { chat_template_kwargs: { enable_thinking: false } }
```

## Map reasoning controls

Clients ask for reasoning with `reasoning_effort` (`none`, `low`, `medium`, `high`, `xhigh`,
`max`) and `reasoning_max_tokens`. Declare the levels your runtime accepts and the name of its
thinking-budget field, and the daemon maps every request onto them:

```yaml
      reasoning: { efforts: [none, low, high], budget: reasoning_budget, default_effort: low }
```

A requested level the runtime lacks runs at the nearest accepted level below it. For a
runtime whose thinking is a chat-template switch, use `reasoning: thinking_bool`. The
[reasoning dialects reference](../reference/presets.md#reasoning-dialects) has the full rules.

## Stream long generations

Some gateways close a request after a fixed wall time. With `stream: true` the daemon streams
the completion and reassembles it, and `timeout_s` then bounds the silence between two chunks
rather than the whole answer:

```yaml
      stream: true
      timeout_s: 120
```

The attempt as a whole still ends at the job's SLA deadline. `stream` works on `openai-chat`
only.

## Serve an embeddings model

```yaml
  - model: example-org/embed-large
    modality: embedding
    slas:
      "1h": { rate_in: "0.02", rate_out: "0" }
    backend:
      preset: openai-embeddings
      base_url: http://localhost:8001/v1
      model: embed-large
      input_type: passage
```

The daemon sends `POST {base_url}/embeddings` with the job's `input`, `encoding_format`
(default `base64`) and `dimensions` if the job names one. The response is returned verbatim.
Embeddings settle with no output count, so the job is paid on `rate_in` alone.

For an asymmetric retrieval model, `input_type` pins the side (`query` or `passage`) this
listing embeds into. Set `input_type_overridable: true` to let a caller choose the side
instead.

## Complete example

[`vllm.yaml`](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/docs/examples/vllm.yaml)
is a complete configuration for a self-hosted vLLM model, with the optional blocks commented
out.

## Related

- [Presets reference](../reference/presets.md)
- [Serve Responses and Batch endpoints](./serve-responses-and-batch-endpoints.md)
- [Respect a backend's rate limits](./respect-backend-rate-limits.md)
