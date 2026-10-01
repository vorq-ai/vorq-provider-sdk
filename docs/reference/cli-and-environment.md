---
title: CLI and environment
description: The vorqd command line, where it reads its configuration, its signals and exit codes, and the environment variables it reads.
---

## Command

```
vorqd [--config FILE]
```

| Option | Meaning |
|---|---|
| `--config FILE` | Path to `vorqd.yaml`. |
| `-h`, `--help` | Print usage and exit. |

The configuration is read from the first of:

1. the file given by `--config`;
2. the `VORQD_CONFIG` environment variable, which must hold the YAML document itself (a
   single-line value is refused as a path);
3. `/etc/vorqd/vorqd.yaml`.

## Signals

| Signal | Effect |
|---|---|
| `SIGTERM`, `SIGINT` | Drain: stop claiming, suspend resumable jobs, withdraw all asks, let other jobs settle, exit. |
| `SIGKILL` | Immediate stop; jobs are recovered on the next start. |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Stopped after a drain. |
| `2` | The configuration could not be found or loaded, or no wallet key is available. The message is printed to stderr as `vorqd: …`. |
| other | A startup failure after loading, such as an unreachable coordinator, a model missing from the catalog or a box key that does not match the provider record. |

## Environment variables

| Variable | Meaning |
|---|---|
| `VORQD_CONFIG` | The configuration document, when no `--config` is given. |
| `VORQ_WALLET_KEY` | Operator wallet key, used when `provider.wallet_key` is not set. |
| `VORQ_PIN_GATEWAY` | IPFS gateway for job payloads, read as `GET {gateway}/ipfs/{cid}`. Defaults to `https://ipfs.filebase.io`. Set it empty to disable fetching (no job can then be claimed). |
| `SENTRY_DSN` | Enables error reporting to Sentry. |
| `SENTRY_ENVIRONMENT`, `SENTRY_RELEASE` | Tags on Sentry events. |

Any other variable is read only where the configuration names it with `env:NAME`, such as
`VORQ_BOX_KEY` or a backend API key.

## Ports and paths

| Item | Default |
|---|---|
| Ops server (`/healthz`, `/metrics`) | `0.0.0.0:9090` (`provider.metrics_port`) |
| State file | `vorqd-state.sqlite` in the working directory (`provider.state_db`) |
| Default config path | `/etc/vorqd/vorqd.yaml` |
| Container user and working directory | uid `10001`, `/home/vorqd` |
