"""End-to-end queue path: raw submit/poll media backend (mirrors
examples/queue-backend.yaml) — poll until COMPLETED, seal the frames, settle."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from vorqd.cli import build_daemon

from .conftest import CLIENT, Emulator, load_example, no_sleep, transport_for


def queue_backend(*, statuses):
    state = {"polls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v1/models/flux-dev":  # submit
            assert req.headers["authorization"] == "Key sekret"  # env: resolved
            return httpx.Response(200, json={"status_url": "https://queue.example.com/status/1"})
        if path == "/status/1":  # poll
            i = min(state["polls"], len(statuses) - 1)
            state["polls"] += 1
            return httpx.Response(200, json=statuses[i])
        if path.endswith(".png"):  # the rendered frame the daemon seals
            return httpx.Response(200, content=b"\x89PNG-bytes", headers={"content-type": "image/png"})
        return httpx.Response(404, json={})

    return handler


MODEL = "black-forest-labs/flux-2-dev:fp8"
MODEL_ID = 9
# The catalog answers an id and a name; modality is NOT a member of the on-chain
# model record, so the example config declares it and the daemon plans pixel
# units off that. The order never asserts it either way.
CATALOG = [{"id": MODEL, "object": "model", "owned_by": "vorq",
            "vorq": {"model_id": str(MODEL_ID), "enabled": True}}]


def media_job():
    return {
        "model_id": str(MODEL_ID), "state": 0,
        "sla_secs": "86400", "provider_id": "0",
        "rate_in": "0", "rate_out": "0.02",
        "expires_at": None, "claimed_at": "0", "ended_because": 0,
        "input_body": {"prompt": "a cat", "width": 1024, "height": 1024, "num_images": 2},
        # units_out is output pixels (a cap): 2 × 1024 × 1024 = 2_097_152.
        "units_in": None, "units_out": str(2 * 1024 * 1024), "result_cid": None,
        "completion_tok": "0",
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VORQ_WALLET_KEY", "0x" + "4a" * 32)
    monkeypatch.setenv("QUEUE_BACKEND_KEY", "sekret")


async def run(emulator, backend):
    daemon = build_daemon(load_example("queue-backend.yaml"), transport=transport_for(emulator, backend),
                          clock=lambda: 1001.0, driver_kwargs={"sleep": no_sleep})
    emulator.box_public_key = daemon._sched._cipher.public_key   # record must match the local box key
    await daemon.start()
    await daemon.run_once()
    await daemon.join()
    await daemon.shutdown()


async def test_media_job_polls_and_settles_frames_inside_the_sealed_result():
    statuses = [
        {"status": "QUEUED"},
        {"status": "COMPLETED", "images": [{"url": "https://cdn.example.com/a.png"},
                                           {"url": "https://cdn.example.com/b.png"}]},
    ]
    emu = Emulator([media_job()], CATALOG)
    await run(emu, queue_backend(statuses=statuses))
    (settled,) = emu.settled.values()
    # media settles the actual delivered units (output pixels), ≤ the units_out cap
    assert settled["completion_tok"] == 2 * 1024 * 1024
    assert list(emu.jobs.values())[0]["state"] == 2   # Settled
    assert emu.uploads == []   # the pixels never touch the coordinator's file surface

    # Both frames ride base64 inside the one sealed result, each labelled with the
    # dimensions the job was billed on.
    sealed = json.loads(base64.b64decode(settled["result"]))
    assert sealed["enc"] == "vorq-sealed-v1"
    payload = json.loads(CLIENT.decrypt(base64.b64decode(sealed["ciphertext"])))
    assert [base64.b64decode(f["b64"]) for f in payload["images"]] == [b"\x89PNG-bytes"] * 2
    assert all(f["content_type"] == "image/png" and f["width"] == 1024 and f["height"] == 1024
               for f in payload["images"])


async def test_media_backend_failure_abandons():
    statuses = [{"status": "FAILED"}]
    emu = Emulator([media_job()], CATALOG)
    await run(emu, queue_backend(statuses=statuses))
    assert emu.settled == {}
    # abandoned after claim; a signed fail op goes out for an immediate refund
    assert list(emu.jobs.values())[0]["state"] == 3   # Cancelled
    assert set(emu.failed) == set(emu.jobs)
