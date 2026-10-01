---
title: Respect a backend's rate limits
description: Describe a backend's quota, concurrency and timeouts so the daemon never claims work it cannot start or finish.
---

A hosted or shared backend usually limits one key: so many requests per minute or day, so many
in flight, a ceiling on one request's wall time. Declare those limits on the model's `backend`
block. The daemon then keeps to them before it claims, so it never holds a job its quota cannot
run.

## Declare the limits

```yaml
    backend:
      preset: openai-chat
      base_url: https://inference.example.com/v1
      model: example-model
      api_key: env:INFERENCE_API_KEY
      concurrency: 4                       # requests in flight at once
      rate_limit: { "1m": 40, "24h": 5000 }  # requests started per rolling window
      timeout_s: 300                       # ceiling on one request
      retries: 3                           # attempts after the first
      retry_backoff_s: 30
      retry_backoff_max_s: 900
```

Every limit counts per model entry. Retries count against `rate_limit` too, so a backend that
fails often eats into the quota for new work.

Each poll tells the coordinator how many jobs the model can take (`free`). It is the smallest
of the daemon's free capacity slots and what the entry's own limits allow. A model whose
minute is spent reports `free=0` and is offered nothing until the window frees up. Other
models on the same daemon are unaffected.

## Hold a day's work for a long SLA window

The `rate_limit` window equal to the entry's shortest SLA window is also how many jobs the
entry may hold at once. With a `24h` SLA and `rate_limit: {"1m": 36, "24h": 2880}`, the entry
may hold up to 2,880 claimed jobs, start at most 36 a minute, and run `concurrency` of them at
a time. Shorter windows only pace the starts. Without a window at the SLA, every claim must be
startable immediately.

Set that number to what the backend can **finish** inside the window, roughly
`concurrency × window / job time`, not to the quota's headline. A held job that cannot start
before its deadline is handed back at your cost. The loader refuses a budget the shorter
windows cannot even start inside the SLA.

Window counts live in memory. After a restart the daemon recovers the jobs it held, but not the
count of requests already started, so it may start a fresh budget on top of the old one.

## Retry inside the deadline

A `429`, any `5xx` or a transport error (a timeout included) is retried up to `retries` times.
The wait starts at `retry_backoff_s`, doubles each time and is capped at
`retry_backoff_max_s`. A `429` with a longer `Retry-After` (in seconds) wins. Any other `4xx`
fails the job at once; add codes to `retry_statuses` for a gateway that answers, say, `404`
while a pool scales up.

Nothing is scheduled past the job's deadline less `provider.safety_margin_s`. A wait or
attempt that would end later fails the job back immediately, so the client is refunded now.
Size `retries` and `retry_backoff_max_s` against the shortest SLA window the model publishes.

## Withdraw a failing backend

After `trip_after` consecutive jobs fail with the backend at fault (default `3`), the model's
asks are withdrawn at once and it is offered nothing. After `trip_cooldown_s` (default `60`)
it is listed again, provided its health check passes. One more fault withdraws it again; one
job the backend completes resets the count. A `4xx` refusal does not count.
`trip_after: 0` disables this.

## Let capacity follow the limits

Leave `provider.capacity` unset and it is the sum of what each entry may hold: its `rate_limit`
window at the SLA, else its `concurrency`. An entry with neither makes `capacity` required.

## Check it

- `vorqd_model_free{model}` is the `free` the coordinator last saw.
- `vorqd_backend_retries_total{model}` counts retries.
- `backend_wait` and `backend_queued` log lines mark a job waiting for a rate-limit window or
  a concurrency slot.
- `backend tripped after N consecutive faults` marks a withdrawal.

## Related

- [Configuration reference: backend limits](../reference/configuration.md#backend-limits)
- [Capacity](../concepts/capacity.md)
