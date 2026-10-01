---
title: Serve image and video models
description: Price media output, bound what a media backend accepts, and pass reference images and clips through.
---

Media models are billed in output pixels. The daemon checks every media request against what
the order paid for before it reaches your backend, and settles on what the backend actually
delivered.

## Price media output

Declare the modality and quote `rate_out` in USD per million output units. An image model's unit is one output
pixel (`num_images × width × height`), a video model's is one pixel-second
(`width × height × duration_secs`). A model that takes no reference input quotes no `rate_in`:

```yaml
  - model: example-org/image-model
    modality: image
    slas:
      "24h": { rate_out: "0.012" }
```

A model that takes reference images or clips quotes `rate_in` too. The input side is billed in
pixel-seconds of reference: `width × height × max(1, duration_secs)` per asset, summed, with a
still counting as one second.

## Forward the fields the request is clamped on

Before the request is rendered, the daemon:

- lowers `num_images` (image) or `duration_secs` (video) so the job fits the output cap the
  client paid for, and hands the job back if even one image or one second does not fit;
- writes the priced `width`, `height` and `duration_secs` into the request, whichever way the
  client spelled them. A request that named a resolution tier also gets the `aspect_ratio` it
  was priced at.

Your mapping must forward those fields, for example `num_images: "{input.num_images}"`, or the
backend renders more than you are paid for.

## Bound compute the pixel price does not cover

Steps, frame rate and similar knobs cost you compute without changing the bill. Cap them:

```yaml
    backend:
      param_caps: { steps: 40, fps: 24 }
```

Any of these params in a job is lowered to the ceiling.

## Declare what the backend renders

```yaml
    backend:
      resolutions: [480p, 720p]      # tiers it renders; other tiers, or raw pixels, are handed back
      durations: [5, 10]             # clip lengths it renders
      auto_duration: -1              # value sent when the client lets the model choose the length
      adaptive_aspect: adaptive      # value sent when the client asks to keep the reference's shape
```

With `durations`, a clip is lowered to the longest listed length the order covers: seven paid
seconds render five and settle five. A job that cannot cover even the shortest is handed back.
A backend with no `auto_duration` or `adaptive_aspect` hands back requests that need them.

## Accept reference images and clips

References travel inline, as base64, inside the sealed payload. After decrypting, the daemon
reads each reference's own header and compares it with what the client declared and paid for.
A reference with more pixels or a longer runtime than declared, one that cannot be read as its
stated type, or one past the network's caps is handed back, and the client is refunded.

Narrow what a backend takes, and bound each kind of reference:

```yaml
    backend:
      reference:
        accept: [image/png, image/jpeg, video/mp4]
        still: { min_side: 300, max_side: 6000, max_bytes: 31457280, max_count: 9 }
        clip:  { min_secs: 2, max_secs: 15, max_total_secs: 15, max_count: 3 }
        audio: { max_bytes: 15728640, max_count: 3 }
```

`accept` can only narrow the types the daemon reads: `image/png`, `image/jpeg`, `image/webp`,
`video/mp4`, plus `audio/mpeg` and `audio/wav` for sound references. Every refusal happens
before anything is uploaded or submitted.

To pass a reference to the backend, compose a data URI. Mark optional ones with `?` so the key
is dropped when the job has none:

```yaml
      request:
        body:
          image_url:     "data:{input.image.media_type};base64,{input.image.b64}"
          end_image_url: "data:{input.end_image.media_type?};base64,{input.end_image.b64?}"
```

The daemon never logs a rendered request, so references do not end up in your logs.

## What the client receives

The result carries the frames inline, labelled with the dimensions read from the delivered
files, and the settled unit count, which is never above the order's cap:

```json
{"images": [{"b64": "…", "content_type": "image/png", "width": 1024, "height": 1024}],
 "units": 1048576, "seed": 42}
```

A result larger than the coordinator's upload limit cannot be delivered. The job is handed back
under `settle_result_too_large` and the client is refunded. If you see it, narrow the
`resolutions` or `durations` the backend accepts.

## Related

- [Raw mappings reference](../reference/raw-mappings.md#media-policy)
- [Map a queue-style API](./map-a-queue-style-api.md)
