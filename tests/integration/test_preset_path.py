"""End-to-end preset path: openai-chat backend (mirrors examples/vllm.yaml)."""

from __future__ import annotations

import httpx
import pytest

from vorqd.cli import build_daemon

from .conftest import Emulator, load_example, no_sleep


def vllm_backend(*, completion="4", tokens=3, fail=False):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/health"):  # readiness probe gating ask publishing
            return httpx.Response(200, json={"status": "ok"})
        assert req.url.path == "/v1/chat/completions"
        if fail:
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": completion}}],
                                         "usage": {"completion_tokens": tokens}})

    return handler


MODEL = "deepseek-ai/deepseek-v4-pro:fp8"
MODEL_ID = 7
#: What a node's catalog answers: an id, a name and an enabled flag. No modality
#: — the example config declares that, because nothing on chain carries it.
CATALOG = [{"id": MODEL, "object": "model", "owned_by": "vorq",
            "vorq": {"model_id": str(MODEL_ID), "enabled": True}}]


def text_job(**over):
    job = {
        "model_id": str(MODEL_ID), "state": 0,
        "sla_secs": "3600", "provider_id": "0",
        "rate_in": "0.3", "rate_out": "0.8",
        "expires_at": None, "claimed_at": "0", "ended_because": 0,
        "input_body": {"messages": [{"role": "user", "content": "2+2?"}], "temperature": 0.7},
        "units_in": "5", "units_out": "64", "result_cid": None, "completion_tok": "0",
    }
    job.update(over)
    return job


def only(mapping: dict):
    """The single job/settle record in the store — its id is content-derived."""
    (value,) = mapping.values()
    return value


@pytest.fixture(autouse=True)
def _wallet(monkeypatch):
    # A fixed, valid secp256k1 test key so the provider address is deterministic.
    monkeypatch.setenv("VORQ_WALLET_KEY", "0x" + "4a" * 32)


async def run(emulator, backend):
    from .conftest import transport_for

    daemon = build_daemon(load_example("vllm.yaml"), transport=transport_for(emulator, backend),
                          clock=lambda: 1001.0, driver_kwargs={"sleep": no_sleep})
    emulator.box_public_key = daemon._sched._cipher.public_key   # record must match the local box key
    await daemon.start()
    await daemon.run_once()
    await daemon.join()
    await daemon.shutdown()
    return daemon


async def test_text_job_settles_with_completion_tokens():
    emu = Emulator([text_job()], CATALOG)
    await run(emu, vllm_backend(completion="4", tokens=7))
    assert only(emu.settled)["completion_tok"] == 7
    # The sealed result rides in the settle itself; nothing is filed as a /v1 file.
    assert only(emu.settled)["result"]
    assert emu.uploads == []
    assert only(emu.jobs)["state"] == 2   # Settled
    # Asks were published on start (one quote per SLA window, in ids and seconds)
    # and withdrawn on drain — as quotes at rate_out 0, since a slot the snapshot
    # omits keeps its price on chain.
    assert {q["sla"] for q in emu.asks_history[0]["quotes"]} == {3600, 86400}
    assert all(q["model_id"] == MODEL_ID for q in emu.asks_history[0]["quotes"])
    assert emu.live_quotes() == []
    assert {q["rate_out"] for q in emu.asks_history[-1]["quotes"]} == {"0"}


async def test_unprofitable_text_job_not_claimed():
    # Below the configured 1h floor of "0.75".
    emu = Emulator([text_job(rate_out="0.1")], CATALOG)
    await run(emu, vllm_backend())
    assert emu.settled == {}
    assert only(emu.jobs)["state"] == 0   # Open


async def test_lost_race_moves_on():
    emu = Emulator([text_job()], CATALOG)
    emu.claim_conflicts = 1  # first claim returns 409
    await run(emu, vllm_backend())
    assert emu.settled == {}
    assert only(emu.jobs)["state"] == 0   # Open


async def test_backend_failure_abandons_without_settle():
    emu = Emulator([text_job()], CATALOG)
    await run(emu, vllm_backend(fail=True))
    assert emu.settled == {}  # abandoned, never settled
    # the failure is reported by a signed fail op, cancelling the job so the client's
    # escrow refunds immediately instead of waiting out the SLA
    assert only(emu.jobs)["state"] == 3   # Cancelled
    assert set(emu.failed) == set(emu.jobs)


def no_usage_backend():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        # 2xx answer but no usage block → no billable count
        return httpx.Response(200, json={"choices": [{"message": {"content": "4"}}]})

    return handler


async def test_missing_usage_job_never_settles():
    # A backend that answers without a usage count has no billable quantity; the
    # job is abandoned before upload and never reaches settle — the escrow's
    # full-cap fallback can never fire.
    emu = Emulator([text_job()], CATALOG)
    await run(emu, no_usage_backend())
    assert emu.settled == {}
    assert only(emu.jobs)["state"] == 3   # Cancelled: reported by a fail op
    assert set(emu.failed) == set(emu.jobs)


async def test_overcount_settles_at_the_cap():
    # units_out=64 is the charged cap; a backend counter above it settles at 64.
    emu = Emulator([text_job()], CATALOG)
    await run(emu, vllm_backend(completion="4", tokens=100))
    assert only(emu.settled)["completion_tok"] == 64
