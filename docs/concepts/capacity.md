---
title: Capacity
description: How many jobs a provider may hold, how the network grants it, and how per-model limits shape what the daemon offers.
---

## Requested and granted capacity

`provider.capacity` is the number of jobs the daemon asks to hold at once. It sends the request
at startup. Left unset, it is the sum of what each model entry may hold (see below).

The network grants:

```
granted = max(1, min(requested, ceiling) × reputation / 1000)
```

Onboarding sets your ceiling and your starting reputation when it registers you. Reputation
ranges from 100 to 1000. It rises by 5 for every settled job and falls by 40 for every job
failed after the 300-second grace window or reclaimed after a missed SLA. A provider registered
at 1000 fills its whole capacity from its first claim. One registered at 200 and asking for 4
slots holds 1 slot until about 60 settled jobs, 2 until about 110, and all 4 at full
reputation.

The daemon reads the grant every sweep and never offers more than it. When the grant is below
`capacity`, it logs `the network grants N of the M slots configured`, and
`vorqd_capacity_granted` shows the grant.

## What each poll offers

For each model, the poll's `free` is the smallest of:

- the daemon's slots (capacity or grant, whichever is smaller) minus jobs held;
- what the model entry's own limits admit.

Without a `rate_limit` window at the entry's shortest SLA, the entry admits what it could start
now: `concurrency` minus jobs held, and the slack in its tightest `rate_limit` window. With such
a window, the entry may hold that window's budget, less the attempts already started in it and
one reserved for every job still waiting. Shorter windows then only pace the starts. This is
how a `24h` entry can take a day's work in the morning and run it `concurrency` at a time.

A tripped breaker makes `free` zero for that model. Jobs recovered after a restart are resumed
regardless of free capacity: they were already sold.

## Sizing

A missed SLA costs reputation, and reputation costs capacity. Set capacity and limits below
what the backend sustains inside the SLA window, not at its peak. Use
`vorqd_backend_latency_seconds` and `vorqd_job_duration_seconds` to measure real throughput and
leave headroom. Settling fewer jobs cleanly beats claiming more and missing some.

## Related

- [Respect a backend's rate limits](../guides/respect-backend-rate-limits.md)
- [Configuration reference](../reference/configuration.md#provider)
