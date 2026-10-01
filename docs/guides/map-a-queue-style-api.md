---
title: Map a queue-style API
description: Describe a submit-then-poll backend with a raw mapping, and make its jobs survive a restart.
---

Many media backends accept a task, answer with an id or a status URL, and deliver the output
later. No preset covers these shapes, so you describe the calls directly in a raw mapping. The
daemon renders the request for each job, polls until the task finishes, and extracts the
result.

## Write the mapping

```yaml
models:
  - model: black-forest-labs/flux-2-dev:fp8
    modality: image
    slas:
      "24h": { rate_out: "0.012" }          # USD per 1M output pixels
    backend:
      request:
        method: POST
        url: https://queue.example.com/v1/models/flux-dev
        headers: { Authorization: "Key env:QUEUE_BACKEND_KEY" }
        body:
          prompt: "{input.prompt}"
          width: "{input.width}"
          height: "{input.height}"
          num_images: "{input.num_images}"
      response:
        mode: poll
        poll:
          status_url: "$.status_url"        # JSONPath into the submit answer
          status_field: "$.status"          # JSONPath into each poll answer
          done_values: [COMPLETED]
          failed_values: [FAILED]
          interval_s: 2
        result:
          media_urls: "$.images[*].url"
      retries: 2
```

- `{input.…}` tokens read the job's decoded input. A token that is the whole string keeps the
  value's JSON type, so `num_images` arrives as a number.
- `env:NAME` works inside any string. Raw mappings have no `base_url` or `api_key` params:
  write the URL out and pull secrets from the environment.
- `$.…` values are JSONPath expressions evaluated against the backend's answers.
- `result` names exactly one output: `media_urls` (the daemon downloads each URL),
  `media_b64` (the daemon decodes each value), or `text` together with `completion_tokens`.

## Forward the metered fields

Image jobs are billed in output pixels (`num_images × width × height`) and video jobs in
pixel-seconds (`width × height × duration_secs`). Before rendering the request, the daemon
lowers `num_images` or `duration_secs` so the job fits the output cap the client paid for.
That only reaches the backend if the body forwards those fields. If you leave `num_images`
out, the backend renders the full request while you are paid for the reduced one.

## Poll with a request instead of a URL

Some APIs poll by `POST`ing a task id. Replace `status_url` with a templated `request`. The
submit answer is in scope as `{submit.…}`:

```yaml
        poll:
          request:
            method: POST
            url: https://api.example.com/tasks
            body: [ { taskType: getResponse, taskUUID: "{submit.data[0].taskUUID}" } ]
          status_field: "$.data[0].status"
          done_values: [success]
          failed_values: [error]
```

## Survive a restart

A job running at a queue backend keeps running if the daemon restarts. Name the field that
identifies it with `handle`, and poll with `{handle}` instead of `{submit.…}`:

```yaml
        poll:
          request: { method: GET, url: "https://queue.example.com/v1/tasks/{handle}" }
          handle: "$.task_id"
          status_field: "$.status"
          done_values: [COMPLETED]
          failed_values: [FAILED]
```

The daemon records the handle in `provider.state_db` as soon as the submit answers. After a
restart it resumes polling instead of submitting again, and on `SIGTERM` it parks the job
instead of waiting for it. A mapping that sets `handle` cannot use `{submit.…}` in its poll
request. See [Stop and restart safely](./stop-and-restart-safely.md).

## Handle APIs that answer 200 for everything

If the backend reports failures in the body rather than the HTTP status, tell the daemon where
to look:

```yaml
      response:
        ok: { field: "$.code", values: [200], retry: [429, 503], message: "$.msg" }
```

Every JSON answer (submit, poll ticks and `prepare` steps) is checked. A value in `retry` is
retried like an HTTP `429`; any other value outside `values` fails the job. The `message` is
written to your log only, never sent to the network.

For a task that ends in one of `failed_values`, `poll.failure_code` adds the backend's own
error code to the failure reason, and `poll.failure_message` logs its message.

## Upload references first

A backend that takes an input image by URL rather than inline needs an upload before the
submit. Add `prepare` steps:

```yaml
      request:
        prepare:
          - name: first_frame
            when: input.image                   # skipped when the job has no image
            method: POST
            url: https://files.example.com/upload
            headers: { Authorization: "Bearer env:BACKEND_KEY" }
            body: { data: "data:{input.image.media_type};base64,{input.image.b64}" }
            extract: { url: "$.data.downloadUrl" }
        method: POST
        url: https://api.example.com/tasks
        body:
          prompt: "{input.prompt}"
          first_frame_url: "{prepare.first_frame.url?}"
```

The trailing `?` marks an optional token. When it resolves to nothing, the whole key is left
out of the request. Uploading sends the client's reference to that storage, so point `prepare`
only at storage that belongs to the backend that will read it.

## Complete example

[`queue-backend.yaml`](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/docs/examples/queue-backend.yaml)
is a complete image model served through a poll mapping.

## Related

- [Raw mappings reference](../reference/raw-mappings.md)
- [Serve image and video models](./serve-image-and-video-models.md)
