# vorq-provider

Run your own inference backend as a provider on the VORQ network.

`vorqd` is the VORQ provider daemon. It is a single process, driven by one `vorqd.yaml`, that
authenticates with a VORQ coordinator, publishes your prices, polls for jobs it can profitably
serve, runs them against your inference backend, seals each result to the client's key and
settles on-chain. You write configuration, not code.

## Features

- **Any HTTP backend** — built-in presets for OpenAI-compatible chat, embeddings, Responses and
  Batch endpoints, plus a declarative raw mapping for task-queue APIs (submit → poll → fetch).
- **Pricing per SLA window** — one ask per window (`1h`, `24h`, …); jobs are claimed only when
  their signed rates clear your floor. Optional load-aware discounting that never changes your
  published ask.
- **Backend limits** — concurrency, rolling rate-limit windows, retries with backoff, and a
  circuit breaker that withdraws a failing model from the order book.
- **SLA-safe** — every attempt is bounded by the job's deadline; a job that cannot finish is
  handed back so the client is refunded immediately.
- **Restart-safe** — jobs in flight at an async backend are resumed after a restart instead of
  resubmitted.
- **Operable** — `/healthz`, Prometheus `/metrics`, structured JSON logs, optional Sentry.

## Installation

```bash
pip install vorq-provider
```

Requires Python 3.11+. A container image can be built from the included `Dockerfile`.

## Quick start

Your operator wallet address and box public key must first be registered with the VORQ network
(see the [quickstart](https://docs.vorq.co/docs/provider/quickstart)). Then write a `vorqd.yaml`:

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
```

and start the daemon:

```bash
export VORQ_WALLET_KEY=<operator-wallet-private-key>
export VORQ_BOX_KEY=<box-private-key>
vorqd --config vorqd.yaml
```

Without `--config`, `vorqd` reads the YAML document from the `VORQD_CONFIG` environment variable,
then from `/etc/vorqd/vorqd.yaml`.

With Docker:

```bash
docker build -t vorqd .
docker run -d --restart unless-stopped \
  -v "$PWD/vorqd.yaml:/etc/vorqd/vorqd.yaml:ro" \
  -v vorqd-state:/home/vorqd \
  --env-file .env \
  -p 127.0.0.1:9090:9090 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  vorqd
```

The `vorqd-state` volume holds the state file of jobs in flight; with `--read-only` the daemon
cannot start without it. Port `9090` serves `/healthz` and `/metrics` only; the daemon needs
no inbound port for work.

## Documentation

Full documentation is at **[docs.vorq.co/docs/provider](https://docs.vorq.co/docs/provider)**:

- [Quickstart](https://docs.vorq.co/docs/provider/quickstart)
- [Guides](https://docs.vorq.co/docs/provider/guides/connect-an-openai-compatible-backend):
  backends, prices, rate limits, Docker, monitoring, key rotation
- [Concepts](https://docs.vorq.co/docs/provider/concepts/architecture): architecture, job
  lifecycle, pricing, capacity, encryption
- [Configuration reference](https://docs.vorq.co/docs/provider/reference/configuration)

Sources are in [`docs/`](https://github.com/vorq-ai/vorq-provider-sdk/tree/main/docs), with complete example configs in
[`docs/examples/`](https://github.com/vorq-ai/vorq-provider-sdk/tree/main/docs/examples).

## Contributing

```bash
python -m venv .venv
.venv/bin/pip install -e . --group dev   # needs pip >= 25.1
.venv/bin/pytest
```

`dev` is a PEP 735 dependency group, so install it with `--group dev`, not `.[dev]`.

## License

[FSL-1.1-ALv2](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/LICENSE.md) (Functional Source License). Any use is permitted except offering a competing product or service. Each version converts to Apache-2.0 two years after its release.
