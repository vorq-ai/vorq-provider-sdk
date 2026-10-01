---
title: Provider daemon
description: What vorqd is, who runs it, what it needs, and where to go next.
---

`vorqd` (PyPI package `vorq-provider`) is the VORQ provider daemon. It turns an inference
backend you already operate into a provider on the VORQ network. You describe the models you
serve, their prices and how to call your backend in one `vorqd.yaml`; the daemon does the rest
and needs no code from you.

A running daemon:

- publishes one price (an **ask**) per model and SLA window;
- polls the coordinator for open jobs it can serve at or above those prices;
- fetches each job's sealed payload, checks it, and claims the job;
- runs the job against your backend, seals the result to the client's key, and settles it
  on-chain;
- hands back any job it cannot finish in time, so the client is refunded at once.

The daemon opens no inbound port for work. It makes outbound calls only, and the coordinator
relays and pays gas for everything the daemon signs.

## Who runs it

Operators with inference capacity: a self-hosted runtime such as vLLM, a GPU cluster behind an
internal API, or a hosted API you have access to. Any backend reachable over HTTP works:
OpenAI-compatible chat, embeddings, Responses and Batch endpoints through built-in presets, and
other APIs, including submit-then-poll task queues, through a declarative mapping.

## Requirements

- Python 3.11 or newer, or Docker.
- An inference backend the daemon can reach over HTTP.
- The URL of a VORQ coordinator.
- A registered provider identity: an operator wallet and a payload-decryption key (box key)
  that VORQ provider onboarding has on record. See the [Quickstart](./quickstart.md#2-create-and-register-your-keys).
- Outbound HTTPS access to the coordinator, to a public IPFS gateway, and to your backend.

## Next steps

- [Quickstart](./quickstart.md): install, register, configure one model and settle a first job.
- [Guides](./guides/connect-an-openai-compatible-backend.md): connect a backend, set prices,
  respect rate limits, deploy, monitor, rotate keys.
- [Concepts](./concepts/architecture.md): architecture, job lifecycle, pricing, capacity,
  encryption.
- [Reference](./reference/configuration.md): every configuration key, the presets, raw
  mappings, the CLI, metrics and log events.
