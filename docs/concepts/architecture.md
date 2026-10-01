---
title: Architecture
description: The parts of vorqd, what each one talks to, how the daemon starts, and how it proves who it is.
---

`vorqd` is one process driven by one configuration file. The configuration and the inference
backend it points at are the whole integration surface: there is no provider-written code.

## Parts

| Part | Job | Talks to |
|---|---|---|
| Session | Signs the daemon in with the operator wallet and keeps the session fresh. | Coordinator `/auth/*` |
| Node client | Reads the job book, the model catalog and the provider record; submits signed operations and the ask book. | Coordinator `/evm/*` |
| Signer | Signs each operation as EIP-712 typed data for the registry contract that verifies it. | Nothing: it runs locally |
| Blob fetch | Downloads a job's sealed payload by its content id and checks it against the job's commitment. | A public IPFS gateway |
| Escrow client | Obtains the payload key for an open bid. | The coordinator's escrow |
| Backend driver | Runs one job against your backend as the configuration describes. | Your inference backend |
| Scheduler | Runs the poll loop: capacity, pricing, claims, the SLA guard, retries, settlement. | All of the above |
| Ops server | Serves `/healthz` and Prometheus `/metrics`. | Your monitoring |

The daemon holds no blockchain connection. It never builds a transaction, pays gas or tracks
a nonce. It signs typed messages; the coordinator simulates each one, relays it on-chain and
pays for it. Because the operations are signed, the coordinator cannot alter them, and
anything it relays is attributable to your wallet.

## Startup

The daemon first **signs in**: it fetches a nonce, signs it with the operator wallet (EIP-712
`VorqSession`) and exchanges the signature for a session token. The coordinator resolves the
wallet to your provider id. If the wallet is not registered yet, the daemon logs it and retries
every `poll_interval_s`, so it can start before onboarding completes.

It then **binds the catalog**, resolving every configured model name to the numeric id the
chain uses. A model the catalog does not carry stops the daemon, because it could never be
priced or claimed. The catalog may also state each model's modality; where it does not, the
configured `modality` is used.

Next it **checks its identity** against its provider record. A regular provider's box public
key must match the local `box_key`, or the daemon exits, since jobs sealed to the key on record
would be unreadable. A [confidential provider](../guides/run-a-confidential-provider.md)
instead publishes its freshly generated key and waits for the record to show it. If the record
restricts which models you may serve, other configured models are left out of the ask book
with a warning.

Finally it **requests capacity** (`provider.capacity` slots) and **publishes its asks** as one
signed snapshot of the whole book. Then the poll loop starts; see
[Job lifecycle](./job-lifecycle.md).

## Identity

Your provider id is issued when onboarding registers your wallet. The daemon never configures
or sends it: it learns the id at sign-in, and every operation it signs recovers to your wallet,
which the registry maps to the id. The one exception is the ask book, which names its provider
id so the registry can refuse a book signed by someone else's wallet.

Sessions refresh automatically before they expire. A `401` from the coordinator triggers one
fresh sign-in.

## Where state lives

- **On-chain and at the coordinator:** your provider record, reputation, granted capacity, the
  ask book and every job.
- **In the daemon's memory:** jobs in flight, rate-limit counters, breaker state, the last load
  reading.
- **On disk:** only `provider.state_db`, which holds the backend handles of jobs in flight at
  resumable backends.

## Related

- [Job lifecycle](./job-lifecycle.md)
- [Encryption](./encryption.md)
