---
title: Quickstart
description: Install vorqd, register your provider identity, configure one model and settle a first job.
---

This tutorial takes you from nothing to a daemon that has settled one job. It serves a text
model from a local [vLLM](https://github.com/vllm-project/vllm) server through the
`openai-chat` preset.

You need Python 3.11 or newer, a machine that can run your model, and the URL of a VORQ
coordinator.

## 1. Install

```bash
python -m venv .venv
.venv/bin/pip install vorq-provider
```

This installs the `vorqd` command. The rest of this page assumes the virtualenv is active
(`source .venv/bin/activate`).

## 2. Create and register your keys

A provider holds two keys:

- the **operator wallet** (secp256k1). The daemon signs its session and every claim, settle and
  price update with it. Use a dedicated wallet for this.
- the **box key** (Curve25519). Clients seal job payloads to its public half, and the daemon
  decrypts with the private half.

Generate both:

```bash
python - <<'EOF'
from eth_account import Account
from nacl.public import PrivateKey

wallet = Account.create()
box = PrivateKey.generate()
print("VORQ_WALLET_KEY=" + wallet.key.hex())
print("VORQ_BOX_KEY=" + box.encode().hex())
print("wallet address:", wallet.address)
print("box public key:", box.public_key.encode().hex())
EOF
```

Keep the two private values secret. Send the **wallet address** and the **box public key** to
VORQ provider onboarding. Onboarding registers them and issues your provider id. You never
configure that id: the daemon learns it when it signs in.

You can start the daemon before registration is complete. Until then it logs
`not registered with the coordinator; waiting for admin provisioning` and retries.

## 3. Start a backend

`vorqd` does not run models. It sends jobs to a backend you operate:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro --port 8000
```

This serves an OpenAI-compatible API at `http://localhost:8000/v1`. Any server that speaks the
same protocol works the same way.

## 4. Write `vorqd.yaml`

```yaml
provider:
  wallet_key: env:VORQ_WALLET_KEY
  box_key: env:VORQ_BOX_KEY
  capacity: 4

models:
  - model: deepseek-ai/deepseek-v4-pro:fp8
    modality: text
    slas:
      "24h": { rate_in: "0.16", rate_out: "0.55" }
      "1h":  { rate_in: "0.22", rate_out: "0.75" }
    backend:
      preset: openai-chat
      base_url: http://localhost:8000/v1
      model: deepseek-ai/DeepSeek-V4-Pro
      health: { path: http://localhost:8000/health }
```

- `env:NAME` reads a value from the environment, so no secret sits in the file.
- `capacity` is how many jobs the daemon may hold at once.
- `models[].model` is the name clients submit against. It must be a model in the
  coordinator's catalog, or the daemon refuses to start. `backend.model` is your runtime's own
  name for it.
- `slas` publishes one ask per SLA window. Rates are USD per million units of work, written as
  quoted decimal strings. For a text model that is per million input tokens (`rate_in`) and
  per million output tokens (`rate_out`), so `"0.55"` is 0.55 USD per million output tokens.
  The daemon claims only jobs whose rates are at least these.
- `health` gates the asks: while the probe fails, the model is off the order book.

The full schema is in the [configuration reference](./reference/configuration.md).

## 5. Run

```bash
export VORQ_WALLET_KEY=<your wallet key>
export VORQ_BOX_KEY=<your box key>
vorqd --config vorqd.yaml
```

The daemon signs in, binds your models to the catalog, checks that your box key matches the
one on record, requests capacity, publishes its asks and starts polling. It logs one JSON
object per line:

```json
{"level": "info", "logger": "vorqd", "event": "the network grants 4 slots"}
```

Check that it is healthy:

```bash
curl -s localhost:9090/healthz    # ok
```

## 6. Settle a first job

Submit a job for the same model with the
[Python client SDK](https://github.com/vorq-ai/vorq-client-sdk-python), using a separate,
funded client wallet:

```python
import asyncio

import vorq


async def main():
    # Reads the client's own VORQ_WALLET_KEY from the environment.
    client = vorq.Client()
    handle = await client.submit(
        model="deepseek-ai/deepseek-v4-pro:fp8",
        input="Summarize the plot of Hamlet in three bullet points.",
    )
    result = await handle.result()
    print(result.text)
    print(result.usage)


asyncio.run(main())
```

Within a poll interval the daemon logs the claim and then the settle:

```json
{"level": "info", "logger": "vorqd", "event": "claimed", "job_id": "0x7c65…", "model": "deepseek-ai/deepseek-v4-pro:fp8"}
{"level": "info", "logger": "vorqd", "event": "settled", "job_id": "0x7c65…", "model": "deepseek-ai/deepseek-v4-pro:fp8", "result_cid": "bafkrei…"}
```

`handle.result()` returns the text your backend produced.

## Next steps

- [Connect an OpenAI-compatible backend](./guides/connect-an-openai-compatible-backend.md) to tune
  what is forwarded to your runtime.
- [Deploy with Docker](./guides/deploy-with-docker.md) to run the daemon as a service.
- [Monitor the daemon](./guides/monitor-the-daemon.md) before you take real traffic.
- [Job lifecycle](./concepts/job-lifecycle.md) to see what happens between claim and settle.
