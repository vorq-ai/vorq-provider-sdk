---
title: Deploy with Docker
description: Build the vorqd image and run it with a mounted config, a state volume and a health check.
---

The repository's `Dockerfile` builds a small image that runs `vorqd` as an unprivileged user
(uid `10001`) with a built-in health check.

## Build the image

```bash
git clone https://github.com/vorq-ai/vorq-provider-sdk.git
cd vorq-provider-sdk
docker build -t vorqd .
```

## Run it

Put your keys and backend secrets in an env file, then:

```bash
docker run -d --name vorqd --restart unless-stopped \
  -v "$PWD/vorqd.yaml:/etc/vorqd/vorqd.yaml:ro" \
  -v vorqd-state:/home/vorqd \
  --env-file .env \
  -p 127.0.0.1:9090:9090 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  vorqd
```

- **Config.** Started with no arguments, the daemon reads the YAML document from the
  `VORQD_CONFIG` variable if it is set, and otherwise `/etc/vorqd/vorqd.yaml`. On a platform
  that cannot mount files, put the whole document in `VORQD_CONFIG`. It must be the document
  itself, not a path.
- **State.** The state file (`provider.state_db`, default `vorqd-state.sqlite`) resolves against
  the working directory `/home/vorqd`. It holds the backend handles of jobs in flight, so the
  volume keeps them across container replacement. With `--read-only` the daemon cannot start
  without a writable volume there.
- **Ports.** The daemon needs no inbound port for work. Port `9090` serves only `/healthz` and
  `/metrics`; publish it on localhost or a private network only.

## Health check

The image's `HEALTHCHECK` calls `http://localhost:9090/healthz` every 30 seconds, so
`docker ps` shows `healthy` or `unhealthy`. The image's `EXPOSE` and `HEALTHCHECK` both use the
default port. If you change `provider.metrics_port`, change both lines of the `Dockerfile` too,
or the container reports `unhealthy` while the daemon is fine.

## Stop it

`docker stop` sends `SIGTERM`, and the daemon drains: it stops claiming, withdraws its asks,
parks resumable jobs and lets other in-flight jobs settle. Docker's default grace period is
10 seconds. Raise it to cover your longest synchronous job, or the drain turns into a hard
kill:

```bash
docker stop --time 600 vorqd
```

See [Stop and restart safely](./stop-and-restart-safely.md).

## Reproducible images

The Dockerfile pins the base image by digest, but not the Python dependencies: rebuilding the
same commit later can resolve newer versions. Build once and deploy that image by its digest if
several hosts must run identical bytes.

## Related

- [Monitor the daemon](./monitor-the-daemon.md)
- [CLI and environment reference](../reference/cli-and-environment.md)
