"""Backend driver against a DeepSeek-format chat backend (mock, CI-safe).

Validates that the openai-chat preset shapes an outbound request matching the
DeepSeek chat-completions reference — the runtime model, the completion budget,
allowlisted sampling params, and the DeepSeek-native fields (``thinking``,
``reasoning_effort``, ``response_format``) carried through ``extra_params`` — and
that it extracts text + ``completion_tokens`` from the response. No network.
"""

from __future__ import annotations

import json

import httpx

from vorqd.backend import BackendDriver
from vorqd.types import EvmJob
from vorqd.config import BackendConfig


def deepseek_backend():
    """openai-chat preset pointed at the DeepSeek chat API, in DeepSeek's format."""
    return BackendConfig(preset="openai-chat", params={
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-ai/deepseek-v4-pro",
        "api_key": "test-key",
        "params_supported": ["temperature", "top_p", "max_tokens", "stop"],
        "extra_params": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "response_format": {"type": "text"},
        },
    })


async def test_builds_deepseek_request_and_extracts_result():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url == "https://api.deepseek.com/chat/completions"
        assert req.headers["authorization"] == "Bearer test-key"
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"completion_tokens": 8},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, deepseek_backend())
    job = EvmJob(job_id="job_1", model="deepseek-ai/deepseek-v4-pro:fp8", modality="text",
                 state="Claimed", sla="1h", created_at=0, units_out=4096)

    result = await driver.run(job, {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hi"},
        ],
        "temperature": 1,
        "top_p": 1,
    })

    assert result.kind == "text"
    assert result.text == "Hello!"
    assert result.completion_tokens == 8

    body = captured["body"]
    assert body["model"] == "deepseek-ai/deepseek-v4-pro"          # runtime model, not the network id
    assert body["max_tokens"] == 4096                  # = job.units_out
    assert body["temperature"] == 1 and body["top_p"] == 1
    assert body["messages"][0]["role"] == "system"
    # DeepSeek-native fields carried through extra_params
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["response_format"] == {"type": "text"}


async def test_disallowed_param_is_dropped_before_backend():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, deepseek_backend())
    job = EvmJob(job_id="j", model="deepseek-ai/deepseek-v4-pro:fp8", modality="text", state="Claimed", sla="1h", created_at=0, units_out=256)

    await driver.run(job, {"messages": [{"role": "user", "content": "Hi"}], "made_up_knob": 5, "best_of": 2})
    # made_up_knob is outside this backend's declared params_supported -> stripped
    # before it reaches the backend; best_of is billing-unsafe and always stripped
    assert "made_up_knob" not in captured["body"]
    assert "best_of" not in captured["body"]
