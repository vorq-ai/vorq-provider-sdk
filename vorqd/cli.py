"""``vorqd`` command-line entry point and daemon wiring.

Builds the seams from config, registers the provider, publishes asks, and runs
the scheduler alongside the ops server. ``SIGTERM``/``SIGINT`` drains: stop
claiming, withdraw asks, suspend jobs that can be resumed at the next boot, let
the rest finish and settle, then exit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from ._crypto import WalletSigner
from .backend import BackendDriver
from .blob import default_resolver
from .config import DEFAULT_CONFIG_PATH, VorqdConfig, load_config, load_config_text
from .coordinator import CoordinatorClient
from .errors import ConfigError
from .escrow import HttpEscrowRelease
from .node import NodeClient
from .ops import Metrics, OpsServer, configure_error_reporting, configure_logging
from .opsig import OpSigner
from .pricing import LoadMonitor, LoadProbe, NullLoadMonitor
from .scheduler import Scheduler
from .state import InflightStore


#: The daemon's one HTTP client speaks to two kinds of door, and httpx's default
#: five seconds is wrong for both.
#:
#: `POST /evm/ops` is not a database write: on the settle branch the coordinator
#: files the sealed result with an object store over the public internet and
#: answers with the name that store minted, so this is a wait on somebody else's
#: storage network. The blob surface is the same wait from the other side — a
#: read gateway serving a name that may have been minted seconds ago. Abandoning
#: either at five seconds does not retry it; it abandons a settle that is still
#: in flight and loses the job at this daemon's own cost.
#:
#: `read` and `write` are sized once for the largest thing this daemon sends: a
#: result filed through the files door, which stores up to 200 MiB and answers
#: only once the object store has taken the whole body. At a poor uplink that is
#: minutes, and a budget sized for reading a JSON answer would abort a rendered,
#: paid-for job mid-upload. One static number rather than one scaled per result —
#: the cost of being generous here is a stalled transfer noticed late, and the
#: cost of being tight is a delivered job failed back.
#:
#: `connect` and `pool` stay short on purpose. A coordinator that is not
#: listening is a different fact from a door that is working, and it is not worth
#: a minute — let alone the body budget, which would leave a settle blocked well
#: inside the job's SLA window.
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=900.0, pool=10.0)


class Daemon:
    def __init__(self, config: VorqdConfig, client: httpx.AsyncClient, scheduler: Scheduler, ops: OpsServer):
        self._config = config
        self._client = client
        self._sched = scheduler
        self._ops = ops
        self._stopping = False

    async def _healthy(self) -> bool:
        if self._stopping:
            return False
        try:  # reachable includes authenticated
            await self._sched._coord.token()
            return True
        except Exception:
            return False

    async def start(self) -> None:
        await self._ops.start()
        await self._sched.startup()

    async def run_once(self) -> None:
        await self._sched.run_once()

    async def join(self) -> None:
        await self._sched.join()

    async def serve(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self._sched.run_once()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._config.provider.poll_interval_s)
            except (asyncio.TimeoutError, TimeoutError):
                pass
        await self.shutdown()

    async def shutdown(self) -> None:
        self._stopping = True
        await self._sched.stop()
        # Withdraw the book: the ask registry has no TTL, so prices left standing
        # keep matching this provider with work it is no longer running. Only if
        # we ever held an identity — a daemon that never got past provisioning
        # published nothing and has no id to sign a snapshot for.
        if self._sched._coord.provider_id is not None:
            try:
                await self._sched.shutdown()
            except Exception:
                pass
        await self._sched.join()  # let in-flight jobs settle
        self._sched.close()
        await self._ops.stop()
        await self._client.aclose()


def build_daemon(
    config: VorqdConfig,
    *,
    transport: httpx.BaseTransport | None = None,
    clock=time.time,
    driver_kwargs: dict | None = None,
) -> Daemon:
    client = (
        httpx.AsyncClient(transport=transport, timeout=HTTP_TIMEOUT) if transport is not None
        else httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    )
    api_url = config.provider.api_url
    signer = WalletSigner(config.provider.wallet_key)
    coord = CoordinatorClient(client, api_url, signer)
    # The node is the daemon's one chain collaborator: it reads the book, the
    # catalog and the provider directory, relays what this daemon signs and pays
    # the gas for it. One session, shared with the handshake the coordinator
    # client already holds — a second handshake for the same wallet would mint a
    # second token for nothing.
    node = NodeClient(client, api_url, signer, session=coord)

    async def chain_id() -> int:
        return (await node.chain_context()).chain_id

    # An open bid's DEK comes from the coordinator's attested escrow, against a
    # request this wallet signs. The URL defaults to the node's own — the node
    # serves the escrow today — and the chain id is read once, lazily, because
    # `GET /evm/chain` cannot be read while the wiring is being built.
    escrow = HttpEscrowRelease(client, config.provider.escrow_url or api_url, signer, chain_id)
    kw = driver_kwargs or {}

    cipher = evidence = None
    if config.confidential:
        # The one import seam: an unflagged config never executes a line of vorqd/tee/.
        from .tee import MockAttestationAgent, boot_identity

        # Hardware attestation ships only inside the measured CVM image; this
        # package carries the mock agent (mock-tagged, refused by production
        # verifiers), so the whole dev/CI loop runs on any machine.
        ident = boot_identity(signer.address, MockAttestationAgent())
        cipher, evidence = ident.cipher, ident.evidence

    # A confidential model runs through the same driver as any other: what makes
    # it confidential is this daemon's attested identity, not its backend.
    drivers = {m.model: BackendDriver.from_config(client, m.backend, **kw) for m in config.models}
    # A model may serve one SLA window through a backend of its own — a second
    # driver beside the model's default, chosen per job by its window.
    sla_drivers = {
        (m.model, window): BackendDriver.from_config(client, be, **kw)
        for m in config.models for window, be in m.sla_backends.items()
    }
    metrics = Metrics()
    # Containers are fetched by CID and checked against the job's commitment, so
    # any source is safe to read from — the daemon reads the storage network's
    # public gateway and the coordinator is off the byte path entirely. The
    # gateway and its propagation-sized retry budget are chosen in one place
    # (`blob.default_resolver`), which is also what the scheduler falls back to.
    blobs = default_resolver(client)
    # One probe per model that says where its backend reports load, read off the
    # `/metrics` the runtime already serves — the daemon asks the backend for
    # nothing and holds no state on it. No `load:` block anywhere is the null
    # monitor: every model's load is unknown, unknown prices exactly like busy,
    # and the private floor stays at the configured rates.
    probes = {
        m.model: LoadProbe(m.load.url, m.load.metric, m.load.scale, client, clock=clock)
        for m in config.models if m.load is not None and m.load.source == "probe"
    }
    load = LoadMonitor(probes, metrics) if probes else NullLoadMonitor()
    # The handles of jobs in flight at an async backend, on disk, so a
    # restart resumes them instead of submitting them again. A path the daemon's
    # user cannot open is a config error, named here: sqlite's own message says
    # "unable to open database file" and nothing about which setting chose it.
    try:
        store = InflightStore(config.provider.state_db)
    except sqlite3.OperationalError as exc:
        raise ConfigError(
            f"provider.state_db: cannot open {config.provider.state_db!r} ({exc}); "
            "point it at a writable path"
        ) from exc
    scheduler = Scheduler(config, node, coord, drivers, metrics, ops=OpSigner(signer),
                          clock=clock, http=client, cipher=cipher, evidence=evidence, blobs=blobs,
                          escrow=escrow, load=load, store=store, sla_drivers=sla_drivers)
    daemon = Daemon(config, client, scheduler, ops=None)  # type: ignore[arg-type]
    daemon._ops = OpsServer(metrics, daemon._healthy, config.provider.metrics_port)
    return daemon


async def serve(config: VorqdConfig, *, transport: httpx.BaseTransport | None = None) -> None:
    configure_logging(config.provider.log_level)
    configure_error_reporting()
    daemon = build_daemon(config, transport=transport)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, ValueError):  # pragma: no cover
            pass
    await daemon.start()
    await daemon.serve(stop)


#: The configuration document itself, for a platform that mounts no files.
CONFIG_ENV = "VORQD_CONFIG"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vorqd", description="VORQ provider daemon")
    parser.add_argument(
        "--config", metavar="FILE",
        help=f"Path to the vorqd.yaml configuration file. Without it the document is read from "
             f"${CONFIG_ENV}, and failing that from {DEFAULT_CONFIG_PATH}",
    )
    return parser


def _load(args: argparse.Namespace) -> VorqdConfig:
    """Where the configuration comes from: `--config`, else the document in
    `VORQD_CONFIG`, else the file at the image's default path.

    The variable carries the document, never a path: a path is what `--config`
    is for, and a one-line value is refused by name rather than failing later
    as "config root must be a mapping".
    """
    if args.config:
        if not Path(args.config).is_file():
            raise ConfigError(f"config file not found: {args.config}")
        return load_config(args.config)
    doc = os.environ.get(CONFIG_ENV)
    if doc and doc.strip():
        if "\n" not in doc.strip():
            raise ConfigError(
                f"{CONFIG_ENV} must hold the configuration document itself, not a path "
                f"— a path goes to --config"
            )
        return load_config_text(doc)
    if Path(DEFAULT_CONFIG_PATH).is_file():
        return load_config(DEFAULT_CONFIG_PATH)
    raise ConfigError(
        f"no configuration: pass --config FILE, set {CONFIG_ENV} to the document, "
        f"or place it at {DEFAULT_CONFIG_PATH}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load(args)
    except ConfigError as exc:
        print(f"vorqd: {exc}", file=sys.stderr)
        raise SystemExit(2)
    try:
        asyncio.run(serve(config))
    except ValueError as exc:  # e.g. missing wallet key — required to start
        print(f"vorqd: {exc}", file=sys.stderr)
        raise SystemExit(2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
