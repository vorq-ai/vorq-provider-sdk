---
title: Job lifecycle
description: What the daemon does with a job from the open book to settlement, and how it behaves when something fails.
---

Every `poll_interval_s` (default 5 seconds) the daemon runs one sweep. It reads the load of each
backend, refreshes the capacity the network grants it, republishes its asks if they changed,
and then polls the open book once per served model.

## Before the claim

The poll tells the coordinator how many jobs the model can take right now (`free`) and the
lowest rates the daemon accepts. The coordinator leases the daemon at most `free` of the oldest
open jobs that clear those rates. The poll is also the daemon's heartbeat, so a full or
withdrawn model still polls, with `free=0`.

Each returned job is then checked, cheapest check first. Nothing is claimed until all pass:

- **Price.** The job's signed rates must clear the floor for its SLA window: the configured
  rates, or less with [dynamic pricing](./pricing-and-asks.md#the-private-floor). Only the
  cleartext order terms are read; nothing is decrypted to price a job.
- **Capacity.** A free slot, within the network's grant and the model's own limits.
- **Payload.** The sealed container is downloaded by its content id from a public IPFS gateway
  and checked against the job's own commitment (see [Encryption](./encryption.md)). Bytes that
  do not match are never claimed, whichever gateway served them. A job whose bytes cannot be
  found is skipped and retried later with a growing delay.
- **Declared input size.** For text and embedding models, the payload must not be far larger
  than the input units the client paid for (`max_input_bytes_per_unit`, default 16 bytes per
  unit). Oversized bids are skipped.

Declining here is free: nothing is spent, and the job stays on the book for other providers.

## Claim

The daemon simulates the claim, then signs and submits it. The coordinator simulates the
signed operation again and relays it. A refusal usually means another provider claimed the job
first, and the daemon moves on. A successful claim starts the SLA clock: the deadline is
`claimed_at + sla`.

## Open and run

The daemon recovers the payload key, decrypts the payload and checks the envelope inside: its
version, its owner (which must be the job's owner) and the key the result must be sealed to. A
payload that fails any check is handed back at once with a signed `fail`, which refunds the
client. Within 300 seconds of the claim a fail costs no reputation.

The job then runs against the backend for its SLA window. Media requests are first clamped to
the output the client paid for. Each attempt is bounded by the time left before
`deadline − safety_margin_s`, and retryable failures are retried inside that budget (see
[Respect a backend's rate limits](../guides/respect-backend-rate-limits.md)).

## Settle

The result is sealed to the client's key and settled with a signed `settle` carrying the
backend's output count (tokens for text, delivered pixels for media, capped at the order's
limit). A sealed result up to 15,679,488 bytes travels inline with the settle; a larger one is
uploaded to the coordinator first. The coordinator stores the bytes, names them by content id
and records that id on-chain.

The chain computes the charge from the rates the client signed. For text, the settled count is
the one number the provider supplies, and the correct value is defined as the retokenization
of the delivered output, so an inflated count is disputable.

## The SLA guard

The deadline is enforced on-chain: a late settle is refused. If the backend finishes after
`deadline − safety_margin_s`, the daemon does not try to settle. It logs `abandoned (SLA)`,
frees the slot and leaves the job to be reclaimed, which refunds the client and costs the
provider reputation.

## Failure handling

| What happens | What the daemon does |
|---|---|
| Backend health check fails | Withdraws the model's asks; jobs already claimed keep running. Republishes when the check passes. |
| Backend error during a job | Retries a `429`, `5xx` or timeout within the SLA budget. Otherwise, or once retries run out, sends `fail` so the client is refunded now. |
| Backend fails `trip_after` jobs in a row | Withdraws the model's asks immediately and lists it again after `trip_cooldown_s`. |
| Payload or envelope invalid, payload key unavailable | Sends `fail` immediately. |
| Settle refused by the chain (`409`) | Logs it and drops the job: it is no longer this provider's to settle. |
| Settle not delivered (HTTP error, upload failure, reverted) | Sends `fail` so the client is refunded now. |
| Coordinator unreachable during a sweep | Skips that sweep or model and tries again on the next one. Running jobs continue. |
| Coordinator or catalog unreachable at startup | Exits. Run the daemon under a supervisor. |
| Modality unknown for a model | Claims nothing for it until the catalog or the config names one. |

Every failure path frees its capacity slot. If a `fail` itself cannot be delivered, the client
is still refunded when the SLA expires and the job is reclaimed.

## Related

- [Architecture](./architecture.md)
- [Stop and restart safely](../guides/stop-and-restart-safely.md)
