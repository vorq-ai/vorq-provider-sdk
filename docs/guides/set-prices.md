---
title: Set prices
description: Quote one ask per SLA window, and optionally accept cheaper bids while your backend is idle.
---

## Quote one ask per SLA window

Each entry under `slas` publishes one ask: a window and its rates.

```yaml
    slas:
      "24h": { rate_in: "0.16", rate_out: "0.55" }
      "1h":  { rate_in: "0.22", rate_out: "0.75" }
```

- A window is a count and a unit: `s`, `m`, `h` or `d`.
- A rate is USD per million units of work, written as a quoted decimal string. `"0.55"` on a
  text model is 0.55 USD per million output tokens. An unquoted number is refused at startup,
  and so is a rate with more fraction digits than the payment token holds (6 for USDC).
- `rate_out` is required on every window. `rate_in` is omitted on a model that meters no input
  side, such as a text-to-image model.

The units depend on the modality: tokens for `text` and `embedding`, output pixels for
`image`, output pixel-seconds for `video`. See
[Pricing and asks](../concepts/pricing-and-asks.md).

These rates are also your floor. The daemon claims a job only when the rates the client signed
are at least the configured rates for that job's window. Jobs priced below are left for other
providers.

Price changes take effect on restart. Stop the daemon with `SIGTERM`: a graceful shutdown
withdraws every ask it published, and the new process publishes the new book. After a hard
kill, a window you removed from the config stays on the book at its old price, because the new
process does not know it was ever published.

## Accept cheaper bids while idle

Dynamic pricing is off by default. When configured, the daemon may claim bids priced **under**
its published ask while the backend is quiet. The published ask does not change, and nothing
about the discount is published or signed.

```yaml
provider:
  pricing:
    max_discount_pct: 20     # the discount at a fully idle backend, shrinking to 0 at full load
    bid_tolerance_pct: 10    # an extra discount while load is under low_load_pct
    low_load_pct: 30

models:
  - model: deepseek-ai/deepseek-v4-pro:fp8
    # ...
    load:
      url: http://localhost:8000/metrics
      metric: "vllm:kv_cache_usage_perc"
      scale: 1.0
```

At load 0.5 this accepts bids 10% under the ask; below 30% load it adds another 10%. The total
is capped at 90%. A model without a `load` block always claims at exactly its configured rates.

### Choose a load source

- **A Prometheus endpoint.** Name the URL and the series. `scale` is the divisor that turns the
  series into 0..1: `1.0` for a ratio, `100` for a percentage such as a GPU exporter's
  utilization. When the series has several samples (one per GPU), the busiest one counts.
- **The entry's own occupancy.** For a backend that exports nothing, such as a hosted API
  behind a quota, use `load: { source: occupancy }`. The load is the fuller of jobs held over
  what the entry may hold, and requests started over the budget of its longest `rate_limit`
  window. It needs a `concurrency` or a `rate_limit` on the backend.

A probe that fails, returns no usable sample, or is older than 30 seconds counts as busy: the
daemon then takes no discount. A failing probe never stops claiming or settling.

### Check it

`vorqd_backend_load{model}` shows the last reading and `vorqd_bid_floor_discount_pct{model}`
the discount it currently earns. A discount of `0` while load looks low usually means the probe
is failing: check `vorqd_load_probe_failures_total{model}`.

The discount gauge reveals how far under your ask you will go. Keep the metrics port off the
public internet.

## Related

- [Pricing and asks](../concepts/pricing-and-asks.md)
- [Configuration reference: pricing](../reference/configuration.md#pricing)
