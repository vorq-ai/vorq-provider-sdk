"""Delivery survives a provider restart, because there is no delivery.

Discovery is a poll of Open chain state: nothing is pushed to the daemon, so
there is no message to miss while it is down. A job posted mid-restart simply
rests in Open — held by its order's own deadline — and the first sweep after
boot finds, claims, and settles it like any other. The window this buys is the
order's TTL: a provider back before ``expires_at`` loses nothing.

Both restart timings are covered here. A job posted while the daemon is down
waits in Open until the next life claims it; a job already claimed when the
daemon died is not stranded either — the first sweep after boot lists what this
provider still owns and finishes it, so the client's spent order still delivers.
"""

from __future__ import annotations

import base64
import json

import pytest

from vorqd.cli import build_daemon

from .conftest import CLIENT, Emulator, load_example, no_sleep, pinned_job, transport_for
from .test_queue_path import CATALOG, media_job, queue_backend


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VORQ_WALLET_KEY", "0x" + "4a" * 32)
    monkeypatch.setenv("QUEUE_BACKEND_KEY", "sekret")


async def boot_and_sweep(emu, backend):
    """One daemon lifetime: boot a fresh process over the shared world, run one
    sweep, drain, shut down. Two calls = a restart."""
    daemon = build_daemon(load_example("queue-backend.yaml"),
                          transport=transport_for(emu, backend),
                          clock=lambda: 1001.0, driver_kwargs={"sleep": no_sleep})
    emu.box_public_key = daemon._sched._cipher.public_key
    await daemon.start()
    await daemon.run_once()
    await daemon.join()
    await daemon.shutdown()


async def test_job_posted_while_the_daemon_is_down_is_settled_after_reboot():
    statuses = [
        {"status": "QUEUED"},
        {"status": "COMPLETED", "images": [{"url": "https://cdn.example.com/a.png"},
                                           {"url": "https://cdn.example.com/b.png"}]},
    ]
    emu = Emulator([], CATALOG)
    backend = queue_backend(statuses=statuses)

    # First life: the world is empty; the daemon sweeps nothing and goes down.
    await boot_and_sweep(emu, backend)
    assert emu.settled == {}

    # No daemon is running when the client posts a job designated to THIS
    # provider. Nobody notifies anybody: the job rests in Open state.
    spec = {**media_job(), "designated": Emulator.PROVIDER_ID}
    job, raw = pinned_job(spec, spec["input_body"])
    emu.jobs[job["job_id"]] = job
    emu.blobs[job["task_cid"]] = raw

    # Second life: a fresh process over the same world. Its first sweep of Open
    # state finds the resting job — designated to it — and runs it to settle.
    await boot_and_sweep(emu, backend)

    assert job["state"] == 2   # Settled
    (settled,) = emu.settled.values()
    sealed = json.loads(base64.b64decode(settled["result"]))
    payload = json.loads(CLIENT.decrypt(base64.b64decode(sealed["ciphertext"])))
    assert [base64.b64decode(f["b64"]) for f in payload["images"]] == [b"\x89PNG-bytes"] * 2


async def test_job_claimed_by_a_previous_life_is_settled_after_reboot():
    statuses = [
        {"status": "QUEUED"},
        {"status": "COMPLETED", "images": [{"url": "https://cdn.example.com/a.png"},
                                           {"url": "https://cdn.example.com/b.png"}]},
    ]
    emu = Emulator([{**media_job(), "designated": Emulator.PROVIDER_ID}], CATALOG)
    backend = queue_backend(statuses=statuses)

    # A previous life claimed this job and died before settling: the order is
    # already spent, and the row names THIS provider as its owner. No daemon is
    # running, so nothing is in flight — the claim is orphaned, not racing.
    (job,) = emu.jobs.values()
    job.update(state=1, provider_id=str(Emulator.PROVIDER_ID), claimed_at="1000")

    # Next life. The job is no longer Open, so no claim sweep would ever see it;
    # only the boot read of Claimed-by-me rows recovers it.
    await boot_and_sweep(emu, backend)

    assert job["state"] == 2   # Settled
    (settled,) = emu.settled.values()
    # It genuinely ran: the sealed result opens with the client's key, frames and
    # all — the recovered job went through the backend, not just a state flip.
    sealed = json.loads(base64.b64decode(settled["result"]))
    payload = json.loads(CLIENT.decrypt(base64.b64decode(sealed["ciphertext"])))
    assert [base64.b64decode(f["b64"]) for f in payload["images"]] == [b"\x89PNG-bytes"] * 2
