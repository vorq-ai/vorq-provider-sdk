# syntax=docker/dockerfile:1

# The daemon, as a container. Two stages so that the build backend and the
# source tree stay out of the image that ships: the builder installs into a
# virtualenv, and the runtime receives only that directory.
#
# The base is pinned by digest rather than by tag, so the OS and the interpreter
# under a long-running daemon cannot move without an edit to this line. That pin
# is the only thing frozen here: the dependency closure is NOT pinned. The
# project declares lower bounds, so rebuilding this same commit later can
# resolve newer versions of the daemon's dependencies and produce a different
# image. Build once and deploy that image by its own digest if you need two
# hosts to be running the same bytes.
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e AS build

WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pyproject names README.md as the readme, so the build needs it.
COPY pyproject.toml README.md ./
COPY vorqd ./vorqd
RUN pip install --no-cache-dir .


FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

# The daemon opens no listening socket for work — it polls the API outbound and
# signs ops the coordinator relays — so it needs no privileged port and nothing
# it writes has to outlive the container.
RUN useradd --create-home --uid 10001 vorqd

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER vorqd

# The state file (`provider.state_db`) is a relative path, so it is resolved
# against the working directory — and this is the one directory the daemon's own
# user owns. Mount a volume here to keep it across container replacement.
WORKDIR /home/vorqd

# The ops server only: /healthz and Prometheus /metrics. There is no inbound job
# traffic to expose. Both this and the healthcheck below name the default
# `metrics_port`; a config that moves the ops server has to move them too.
EXPOSE 9090

# /healthz answers 200 only when the scheduler loop is running and the API is
# reachable under an authenticated session — the two conditions the daemon needs
# to make any progress — so it is the right liveness signal, not a process check.
# An unhealthy daemon answers 503 and a stopped one refuses the connection; both
# raise here, so the probe catches everything and reports it as a plain exit 1
# rather than a traceback in `docker inspect`. The timeout is the probe's own, a
# second inside the HEALTHCHECK deadline, so it fails rather than being killed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request\ntry:\n    sys.exit(0 if urllib.request.urlopen('http://localhost:9090/healthz', timeout=4).status == 200 else 1)\nexcept Exception:\n    sys.exit(1)"]

# No CMD: started bare, the daemon reads the document from `VORQD_CONFIG` when
# that is set — a platform that mounts no files hands the config over that way —
# and otherwise the file mounted at /etc/vorqd/vorqd.yaml. Either, and nothing
# to override.
ENTRYPOINT ["vorqd"]
