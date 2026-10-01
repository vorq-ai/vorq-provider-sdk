---
title: Pricing and asks
description: How rates are expressed, how the ask book works, and why the discount floor stays private.
---

## Rates and units

A rate is USD per million units of work, written as a quoted decimal string such as `"0.55"`.
A string keeps every party's arithmetic exact, which is why the loader refuses an unquoted
number. The daemon converts it to the payment token's atomic units (6 decimals for USDC) before
it signs; a rate with more fraction digits than the token holds is refused, never rounded. What
a unit of work is depends on the model's modality:

| Modality | `rate_in` per 10<sup>6</sup> | `rate_out` per 10<sup>6</sup> |
|---|---|---|
| `text` | input tokens | output tokens |
| `embedding` | input tokens | none: settles with no output count |
| `image` | reference pixel-seconds, if the model takes references | output pixels (`num_images × width × height`) |
| `video` | reference pixel-seconds, if the model takes references | output pixel-seconds (`width × height × duration_secs`) |

Pricing media by the pixel lets one model id price every size, and lets a per-image,
per-megapixel or per-second backend be quoted on the same scale.

The client signs the rates and the unit counts into its order. At settlement the chain charges
from the signed rates, the input units the client declared and the output units the provider
reports, never above the order's cap.

## The ask book

An ask is one row per model and SLA window. The daemon publishes its whole book as one signed
snapshot, and only when the book changes. The book has no expiry and needs no heartbeat.

Because the on-chain write is an upsert, a row a snapshot leaves out keeps its old price. So
withdrawal is explicit: a withdrawn row is published with both rates set to 0. That is how an
unhealthy model leaves the book, and why a graceful shutdown republishes every row at 0 rather
than simply going quiet.

Asks advertise you to clients. They do not bind the chain: a claim charges the rates the client
signed, whatever your ask says. That is what lets the daemon apply its own floor.

## The private floor

By default the floor is the configured rates: a job below them is not claimed. With
[dynamic pricing](../guides/set-prices.md#accept-cheaper-bids-while-idle), the floor drops
below the published ask while the backend is idle, so a cheap fill can beat an idle GPU.

The published ask never moves with load, on purpose. A price the market can see is a price bids
converge on, so a published discount would simply reprice your book downwards for every future
client. The floor is private: it is never signed or published, costs no transaction, and is
sent to the coordinator only as a filter on your own poll. Every job that comes back is
checked against the same floor locally.

The discount is linear in load, from `max_discount_pct` at an idle backend to nothing at a full
one, plus `bid_tolerance_pct` while load is under `low_load_pct`, capped at 90%. The arithmetic
rounds toward the undiscounted price. An unknown or stale load reading earns no discount: a
monitoring outage must never sell capacity cheap.

## Related

- [Set prices](../guides/set-prices.md)
- [Capacity](./capacity.md)
