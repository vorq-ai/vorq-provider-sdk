---
title: Serve Responses and Batch endpoints
description: Run long jobs through a background Responses endpoint or a Files + Batches endpoint, and give each SLA window its own backend.
---

Hour- and day-long SLA windows suit backends that accept work in the background and deliver
it later, often at a lower price. Two presets cover the common OpenAI-compatible shapes. Both
record the backend's id for the job, so a restart resumes the job instead of submitting it
again.

## Use a background Responses endpoint

```yaml
    backend:
      preset: openai-responses
      base_url: https://inference.example.com/v1
      model: example-model
      api_key: env:INFERENCE_API_KEY
      service_tier: flex          # optional: flex or priority
      max_polls: 60               # status checks per SLA window
```

The daemon submits `POST {base_url}/responses` with `background: true`, then polls
`GET {base_url}/responses/{id}` every `sla / max_polls` seconds (never under one second) until
the status is `completed`. `failed`, `cancelled` and `incomplete` fail the job.

The request body is built like an `openai-chat` body and then respelled into Responses field
names (`messages` becomes `input`, `max_tokens` becomes `max_output_tokens`, and so on). The
finished object is converted back into a chat completion, so clients read the same shape from
every text backend.

The default forwarded set is smaller than the chat one. Declare `params_supported` from the
endpoint's own schema: some gateways drop an unknown Responses field silently, and the client
then believes it set a control that never reached the model.

## Use a Files + Batches endpoint

```yaml
    backend:
      preset: openai-batch
      base_url: https://inference.example.com/v1
      model: example-model
      api_key: env:INFERENCE_API_KEY
      completion_window: 24h
      max_polls: 288              # a status check every five minutes over 24h
```

Each job becomes a batch of one line. The daemon uploads the chat-completions body as a
one-line JSONL file, creates the batch, polls it until `completed`, and reads the job's line
from the output file. The input file is deleted afterwards. Output and error files are left in
place for you to inspect.

Keep `retries: 0` (the default) on a model served through `openai-batch`. The batch id is only
known once the create call answers, so a retry after a create that timed out can start a
second batch for the same job, and both are billed. Limit keys are per model, so this applies
to every window of that model.

A batch job holds one of the model's slots from upload to collection, which can be the whole
window. Size `capacity` and `concurrency` for that.

## Give one SLA window its own backend

`sla_backends` serves one window through a different backend. Each entry is a patch merged
over `backend`: its keys win, `null` removes a key, and naming a `preset` drops an inherited
raw mapping (and the reverse).

```yaml
    slas:
      "1h":  { rate_in: "0.22", rate_out: "0.75" }
      "24h": { rate_in: "0.16", rate_out: "0.55" }
    backend:
      preset: openai-responses
      base_url: https://inference.example.com/v1
      model: example-model
      api_key: env:INFERENCE_API_KEY
    sla_backends:
      "24h":
        preset: openai-batch
        max_polls: 288
```

Here `1h` jobs run in the background tier and `24h` jobs through the batch tier. Limit keys
(`retries`, `concurrency`, `rate_limit` and the rest) stay on `backend` and are refused in a
patch. The model is offered only while every one of its backends passes its health check.

## Related

- [Presets reference](../reference/presets.md#openai-responses)
- [Stop and restart safely](./stop-and-restart-safely.md)
