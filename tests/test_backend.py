"""BackendDriver: preset expansion, sync/poll execution, result extraction."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from vorqd.backend import (BackendDriver, DEFAULT_DIM, ENVELOPE_SLACK_BYTES, Normalized,
                           _frame_pixels, expand_preset, input_shortfall, plan_media_units,
                           responses_to_chat)
from vorqd.types import EvmJob
from vorqd.config import BackendConfig
from vorqd.errors import BackendError


def text_job():
    return EvmJob(job_id="job_1", model="m:fp8", modality="text", state="Claimed", sla="1h",
                  created_at=0, units_out=128)


def media_job():
    return EvmJob(job_id="job_2", model="img:fp8", modality="image", state="Claimed", sla="24h",
                  created_at=0, units_out=1)


async def no_sleep(_seconds):
    return None


def embedding_job():
    # units_out 0 is the whole point: an embedding is priced on its input side and settles at
    # completionTok 0, so the client escrows rate_in * units_in and nothing more.
    return EvmJob(job_id="job_3", model="emb:fp8", modality="embedding", state="Claimed", sla="24h",
                  created_at=0, units_out=0)


# --- openai-embeddings preset ------------------------------------------------


async def test_openai_embeddings_preset_reports_no_completion_tokens():
    """The billing asymmetry, at the one place it originates.

    An embeddings backend answers with `usage.prompt_tokens` and no completion count — there is
    no output token to report, because the output size is a property of the model. So the
    normalizer must carry `completion_tokens=None`, which is what makes the scheduler settle at
    `completion_tok = 0` and the charge collapse to the input leg.
    """
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/embeddings"
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": "c29tZS1iYXNlNjQ="}],
            "model": "runtime-embed",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        })

    be = BackendConfig(preset="openai-embeddings", params={
        "base_url": "http://runtime/v1", "model": "runtime-embed", "api_key": "sk-x",
    })
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, be, sleep=no_sleep)

    result = await driver.run(embedding_job(), {"input": "what is a vector?"})

    assert isinstance(result, Normalized)
    assert result.kind == "embedding"
    assert result.completion_tokens is None, "an embedding has no output token count to bill"
    assert result.raw["usage"]["prompt_tokens"] == 6
    assert captured["body"]["input"] == "what is a vector?"
    assert captured["body"]["model"] == "runtime-embed"


async def test_openai_embeddings_preset_asks_for_base64_by_default():
    """Vectors are ~4x smaller base64 than as JSON floats, and a multi-input request
    seals every one of them into the result."""
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "object": "list", "data": [], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    be = BackendConfig(preset="openai-embeddings",
                       params={"base_url": "http://runtime/v1", "model": "m"})
    driver = BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)

    await driver.run(embedding_job(), {"input": ["a", "b"]})

    assert captured["body"]["encoding_format"] == "base64"


async def test_openai_embeddings_preset_forwards_dimensions_but_not_prompt_params():
    """`dimensions` is the one knob an embeddings request has; chat sampling params are not
    part of this surface and a backend would 400 on them."""
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "object": "list", "data": [], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    be = BackendConfig(preset="openai-embeddings",
                       params={"base_url": "http://runtime/v1", "model": "m"})
    driver = BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)

    await driver.run(embedding_job(), {"input": "x", "dimensions": 512, "temperature": 0.7})

    assert captured["body"]["dimensions"] == 512
    assert "temperature" not in captured["body"]


async def test_openai_embeddings_preset_sends_the_operators_input_type():
    """Asymmetric embedding models refuse a request that does not say which side it is.

    A retrieval model embeds a *query* and a *passage* into different points, and
    the ones that do it publish that as a required request field: without it the
    backend answers `400 'input_type' parameter is required for asymmetric
    models`, verified live against a hosted endpoint on 2026-08-14. Three of the
    five reachable embedding models there refuse on exactly that.

    It is **operator** configuration and not a caller's field: it is a property of
    how this deployment is being served — an index builder pins `passage`, a
    search path pins `query` — and the network id the caller submits against is
    what distinguishes the two. The vendor's own OpenAI-compatibility escape hatch
    is a `-query` / `-passage` suffix on the model name, which is the same idea and
    which their hosted endpoint 404s on, so this is the only thing that works
    everywhere.
    """
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "object": "list", "data": [], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    be = BackendConfig(preset="openai-embeddings", params={
        "base_url": "http://runtime/v1", "model": "m", "input_type": "passage",
    })
    driver = BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)

    await driver.run(embedding_job(), {"input": "x"})

    assert captured["body"]["input_type"] == "passage"


async def test_openai_embeddings_preset_omits_input_type_when_the_operator_named_none():
    """Absent, not defaulted. A symmetric model has no notion of a side, and two of
    the reachable hosted models answer 200 to a body without the field — sending a
    guess would be a 400 on a model that was working."""
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "object": "list", "data": [], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    be = BackendConfig(preset="openai-embeddings",
                       params={"base_url": "http://runtime/v1", "model": "m"})
    driver = BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)

    await driver.run(embedding_job(), {"input": "x", "input_type": "query"})

    # Not even from the caller: the request body is model-owned, and a caller that
    # could set this could silently halve retrieval accuracy on a served index.
    assert "input_type" not in captured["body"]


def _embeddings_driver(captured: dict, **extra_params):
    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "object": "list", "data": [], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    be = BackendConfig(preset="openai-embeddings", params={
        "base_url": "http://runtime/v1", "model": "m", **extra_params,
    })
    return BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)


async def test_pinned_input_type_ignores_the_caller_unless_the_operator_opens_it():
    """The default stays the default. A network id that names a side — the `:query`
    / `:passage` convention — promises which space the vector lands in, and a caller
    able to override it would make that id a lie."""
    captured = {}
    driver = _embeddings_driver(captured, input_type="passage")

    await driver.run(embedding_job(), {"input": "x", "input_type": "query"})

    assert captured["body"]["input_type"] == "passage"


async def test_overridable_input_type_lets_the_caller_pick_a_side():
    """One published id, an operator-pinned default, and a caller that knows which
    side it needs able to say so. Only meaningful where the id claims no side."""
    captured = {}
    driver = _embeddings_driver(captured, input_type="passage", input_type_overridable=True)

    await driver.run(embedding_job(), {"input": "x", "input_type": "query"})

    assert captured["body"]["input_type"] == "query"


async def test_overridable_input_type_still_defaults_when_the_caller_is_silent():
    """The operator's side is the floor, not merely a suggestion the caller replaces."""
    captured = {}
    driver = _embeddings_driver(captured, input_type="passage", input_type_overridable=True)

    await driver.run(embedding_job(), {"input": "x"})

    assert captured["body"]["input_type"] == "passage"


@pytest.mark.parametrize("bogus", ["document", "QUERY", "", 1, True, None, ["query"]])
async def test_overridable_input_type_drops_a_value_outside_the_two_sides(bogus):
    """A sealed payload is untrusted, and the job is already claimed by the time this
    body is built: an unrecognised side falls back to the operator's rather than being
    forwarded into a `400` the provider would eat. `None` is the caller saying nothing."""
    captured = {}
    driver = _embeddings_driver(captured, input_type="passage", input_type_overridable=True)

    await driver.run(embedding_job(), {"input": "x", "input_type": bogus})

    assert captured["body"]["input_type"] == "passage"


# --- openai-chat preset (sync) ----------------------------------------------


async def test_openai_chat_preset_extracts_text_and_tokens():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        captured["body"] = json.loads(req.content)
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi there"}}],
            "usage": {"completion_tokens": 5},
        })

    be = BackendConfig(preset="openai-chat", params={
        "base_url": "http://runtime/v1", "model": "runtime-model", "api_key": "sk-x",
        "params_supported": ["temperature", "top_p", "max_tokens"],
        "extra_params": {"chat_template_kwargs": {"thinking": False}},
    })
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, be, sleep=no_sleep)

    job = text_job()
    result = await driver.run(job, {"messages": [{"role": "user", "content": "hey"}], "temperature": 1, "top_p": 0.95})
    assert isinstance(result, Normalized)
    assert result.kind == "text"
    assert result.text == "hi there"
    assert result.completion_tokens == 5

    body = captured["body"]
    assert body["model"] == "runtime-model"           # swapped to runtime name
    assert body["max_tokens"] == 128                   # = job.units_out
    assert body["temperature"] == 1 and body["top_p"] == 0.95
    assert body["chat_template_kwargs"] == {"thinking": False}  # extra_params merged
    assert captured["auth"] == "Bearer sk-x"


async def test_preset_response_without_usage_yields_none_count():
    # A 2xx response with no usage block extracts a None count; the scheduler's
    # runtime guard (not the extractor) is what abandons such a job.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    be = BackendConfig(preset="openai-chat", params={"base_url": "http://runtime/v1", "model": "m"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, be, sleep=no_sleep)
    result = await driver.run(text_job(), {"messages": []})
    assert result.kind == "text"
    assert result.text == "hi"
    assert result.completion_tokens is None


async def test_run_applies_per_request_timeout():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["timeout"] = req.extensions.get("timeout")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}})

    be = BackendConfig(preset="openai-chat", params={"base_url": "http://runtime/v1", "model": "m"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, be, sleep=no_sleep)

    await driver.run(text_job(), {"messages": []}, timeout_s=4500)
    # SLA-derived read ceiling reaches httpx; connect stays short so an
    # unreachable backend still fails fast.
    assert captured["timeout"]["read"] == 4500
    assert captured["timeout"]["connect"] == 10.0


def test_expand_preset_shape():
    be = BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "m"})
    mapping = expand_preset(be)
    assert mapping.request["url"] == "http://r/v1/chat/completions"
    assert mapping.response["mode"] == "sync"
    assert mapping.response["result"]["text"] == "$.choices[0].message.content"


# --- openai-chat preset: param filter (base set, allowlist-extends, param_map) ---


def _preset_driver(params_extra=None):
    params = {"base_url": "http://b", "api_key": "k", "model": "m", **(params_extra or {})}
    be = BackendConfig(preset="openai-chat", params=params)
    return BackendDriver.from_config(httpx.AsyncClient(), be)


class _Job:
    units_out = 64
    modality = "text"


def test_openai_chat_body_strips_billing_unsafe_params():
    d = _preset_driver()
    body = d._openai_chat_body({"input": "hi", "n": 4, "stream": True, "stream_options": {}}, _Job(), d.mapping.params)
    assert "n" not in body and "stream" not in body and "stream_options" not in body
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_chat_body_strips_unknown_params_keeps_base():
    d = _preset_driver()
    body = d._openai_chat_body({"input": "hi", "temperature": 0.5, "made_up": 1}, _Job(), d.mapping.params)
    assert body["temperature"] == 0.5
    assert "made_up" not in body


def test_params_supported_replaces_the_base_set():
    """A backend that publishes a per-model schema forwards that schema only:
    a base param the served model does not accept is stripped, not sent."""
    d = _preset_driver({"params_supported": ["temperature", "max_tokens"]})
    body = d._openai_chat_body(
        {"input": "hi", "temperature": 0.5, "top_k": 40, "repetition_penalty": 1.1}, _Job(), d.mapping.params
    )
    assert body["temperature"] == 0.5
    assert "top_k" not in body and "repetition_penalty" not in body


def test_params_supported_carries_nonstandard_knobs():
    """One list covers both jobs: standard fields and a runtime's own extension."""
    d = _preset_driver({"params_supported": ["temperature", "thinking_budget"]})
    body = d._openai_chat_body(
        {"input": "hi", "temperature": 0.5, "thinking_budget": 512, "top_p": 0.9}, _Job(), d.mapping.params
    )
    assert body["temperature"] == 0.5 and body["thinking_budget"] == 512
    assert "top_p" not in body                 # outside the declared schema


def test_params_supported_cannot_resurrect_never_forwarded():
    d = _preset_driver({"params_supported": ["temperature", "service_tier", "n", "stream"]})
    body = d._openai_chat_body(
        {"input": "hi", "service_tier": "priority", "n": 2, "stream": True}, _Job(), d.mapping.params
    )
    assert "service_tier" not in body and "n" not in body and "stream" not in body


def test_negative_budget_sentinel_is_dropped_not_forwarded():
    """Several backends read a negative reasoning budget as an *unlimited*
    sentinel. min(-1, units_out) is -1, so a naive clamp would forward it and
    bill the backend for unbounded thinking settlement cannot pass on — the
    exact hole the clamp exists to close."""
    d = _preset_driver()
    body = d._openai_chat_body({"input": "hi", "reasoning_max_tokens": -1}, _Job(), d.mapping.params)
    assert "reasoning_max_tokens" not in body


def test_non_integer_budget_is_dropped_not_coerced():
    """A sealed payload is untrusted: floats, bools, and strings skip the clamp's
    type expectations, so they are dropped rather than forwarded unclamped."""
    d = _preset_driver()
    for bad in (512.5, 1e9, True, "4096"):
        body = d._openai_chat_body({"input": "hi", "reasoning_max_tokens": bad}, _Job(), d.mapping.params)
        assert "reasoning_max_tokens" not in body, repr(bad)


def test_min_tokens_clamped_to_units_out():
    """A floor above the forced max_tokens is a guaranteed backend rejection the
    provider would eat — clamp it into the paid budget like the thinking cap."""
    d = _preset_driver()
    body = d._openai_chat_body({"input": "hi", "min_tokens": 999999}, _Job(), d.mapping.params)
    assert body["min_tokens"] == 64                              # = job.units_out
    body = d._openai_chat_body({"input": "hi", "min_tokens": -5}, _Job(), d.mapping.params)
    assert "min_tokens" not in body


def test_empty_params_supported_forwards_nothing():
    """`params_supported: []` means exactly that — the declared schema is empty,
    not "fall back to the default set"."""
    d = _preset_driver({"params_supported": []})
    body = d._openai_chat_body({"input": "hi", "temperature": 0.5, "top_p": 0.9}, _Job(), d.mapping.params)
    assert "temperature" not in body and "top_p" not in body
    assert body["max_tokens"] == 64                              # the forced cap still stands


def test_caller_identity_params_never_reach_the_backend():
    """The catalog forbids `user`/`metadata`, so a client never legitimately sends
    one; the daemon strips them anyway rather than hand a backend a stable id that
    links one client's jobs across providers."""
    d = _preset_driver({"params_supported": ["temperature", "user", "metadata"]})
    body = d._openai_chat_body(
        {"input": "hi", "temperature": 0.5, "user": "customer-42", "metadata": {"tenant": "acme"}},
        _Job(), d.mapping.params,
    )
    assert body["temperature"] == 0.5
    assert "user" not in body and "metadata" not in body


def test_param_map_renames_forwarded_key():
    d = _preset_driver({"params_supported": ["reasoning"], "param_map": {"reasoning": "reasoning_effort"}})
    body = d._openai_chat_body({"input": "hi", "reasoning": "high"}, _Job(), d.mapping.params)
    assert body["reasoning_effort"] == "high"
    assert "reasoning" not in body


def test_reasoning_params_forward_by_default():
    d = _preset_driver()
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "low"}, _Job(), d.mapping.params)
    assert body["reasoning_effort"] == "low"


def test_no_reasoning_key_injected_when_client_sets_none():
    """Silent client → backend default stands; nothing reasoning-shaped appears."""
    d = _preset_driver({"param_map": {"reasoning_effort": {"to": "chat_template_kwargs", "values": {"none": {"enable_thinking": False}}}}})
    body = d._openai_chat_body({"input": "hi", "temperature": 0.2}, _Job(), d.mapping.params)
    assert "reasoning_effort" not in body and "chat_template_kwargs" not in body


def test_param_map_value_form_maps_values():
    d = _preset_driver({"param_map": {"reasoning_effort": {
        "values": {"none": "none", "low": "high", "medium": "high", "high": "max"}}}})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "low"}, _Job(), d.mapping.params)
    assert body["reasoning_effort"] == "high"


def test_param_map_value_form_retargets_key():
    d = _preset_driver({"param_map": {"reasoning_effort": {
        "to": "chat_template_kwargs",
        "values": {"none": {"enable_thinking": False}, "high": {"enable_thinking": True}}}}})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "none"}, _Job(), d.mapping.params)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body


def test_param_map_object_values_merge_into_target():
    """Two canonical params can both write into chat_template_kwargs: an object
    value merges into the target key; a dotted `to` path nests a scalar in it."""
    d = _preset_driver({"param_map": {
        "reasoning_effort": {"to": "chat_template_kwargs", "values": {"high": {"enable_thinking": True}}},
        "reasoning_max_tokens": {"to": "chat_template_kwargs.reasoning_budget"},
    }})
    body = d._openai_chat_body(
        {"input": "hi", "reasoning_effort": "high", "reasoning_max_tokens": 32}, _Job(), d.mapping.params
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 32}


def test_param_map_unmapped_drop_and_pass():
    dropping = _preset_driver({"param_map": {"reasoning_effort": {"values": {"none": "none"}, "unmapped": "drop"}}})
    body = dropping._openai_chat_body({"input": "hi", "reasoning_effort": "max"}, _Job(), dropping.mapping.params)
    assert "reasoning_effort" not in body

    passing = _preset_driver({"param_map": {"reasoning_effort": {"values": {"none": "none"}}}})
    body = passing._openai_chat_body({"input": "hi", "reasoning_effort": "max"}, _Job(), passing.mapping.params)
    assert body["reasoning_effort"] == "max"       # default unmapped: pass


def test_param_map_null_mapped_value_drops():
    d = _preset_driver({"param_map": {"reasoning_effort": {"values": {"none": None, "high": "high"}}}})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "none"}, _Job(), d.mapping.params)
    assert "reasoning_effort" not in body


def test_reasoning_max_tokens_clamped_strictly_below_units_out_before_mapping():
    """The clamp runs on the canonical key, so it holds in every dialect — and
    strictly below the cap: the budget spends units_out from the inside, and
    the final answer must fit in what remains."""
    d = _preset_driver({"param_map": {"reasoning_max_tokens": "reasoning_budget"}})
    body = d._openai_chat_body({"input": "hi", "reasoning_max_tokens": 100000}, _Job(), d.mapping.params)
    assert body["reasoning_budget"] == 63          # _Job.units_out - 1

    plain = _preset_driver()
    body = plain._openai_chat_body({"input": "hi", "reasoning_max_tokens": 100000}, _Job(), plain.mapping.params)
    assert body["reasoning_max_tokens"] == 63


# --- reasoning defaults and the derived thinking budget -----------------------


class _BigJob:
    units_out = 10_000
    modality = "text"


def test_default_effort_injected_when_payload_is_silent():
    d = _preset_driver({"default_effort": "medium"})
    body = d._openai_chat_body({"input": "hi"}, _Job(), d.mapping.params)
    assert body["reasoning_effort"] == "medium"


def test_default_effort_never_overrides_an_explicit_control():
    d = _preset_driver({"default_effort": "medium"})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "none"}, _Job(), d.mapping.params)
    assert body["reasoning_effort"] == "none"
    # An explicit budget alone is also a reasoning control the client chose.
    body = d._openai_chat_body({"input": "hi", "reasoning_max_tokens": 32}, _Job(), d.mapping.params)
    assert "reasoning_effort" not in body
    assert body["reasoning_max_tokens"] == 32


def test_default_effort_respects_the_served_schema():
    d = _preset_driver({"default_effort": "medium", "params_supported": ["temperature", "max_tokens"]})
    body = d._openai_chat_body({"input": "hi"}, _Job(), d.mapping.params)
    assert "reasoning_effort" not in body


def test_effort_derives_budget_as_share_of_cap():
    """On a dialect with a budget field, an effort without an explicit budget
    derives one as that effort's share of the charged cap."""
    d = _preset_driver({"param_map": {"reasoning_max_tokens": "reasoning_budget"}})
    for effort, budget in (("low", 2000), ("medium", 5000), ("high", 8000), ("xhigh", 9500), ("max", 9500)):
        body = d._openai_chat_body({"input": "hi", "reasoning_effort": effort}, _BigJob(), d.mapping.params)
        assert body["reasoning_budget"] == budget, effort


def test_derived_budget_floors_then_stays_below_a_small_cap():
    d = _preset_driver({"param_map": {"reasoning_max_tokens": "reasoning_budget"}})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "low"}, _Job(), d.mapping.params)
    assert body["reasoning_budget"] == 63          # 20% of 64 rises to the floor, then bounds at cap-1


def test_no_budget_derived_for_off_or_alongside_an_explicit_one():
    d = _preset_driver({"param_map": {"reasoning_max_tokens": "reasoning_budget"}})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "none"}, _BigJob(), d.mapping.params)
    assert "reasoning_budget" not in body
    body = d._openai_chat_body(
        {"input": "hi", "reasoning_effort": "high", "reasoning_max_tokens": 2048}, _BigJob(), d.mapping.params
    )
    assert body["reasoning_budget"] == 2048        # the client's own budget wins


def test_no_budget_derived_where_the_dialect_drops_the_cap():
    d = _preset_driver({"param_map": {"reasoning_max_tokens": {"values": {}, "unmapped": "drop"}}})
    body = d._openai_chat_body({"input": "hi", "reasoning_effort": "high"}, _BigJob(), d.mapping.params)
    assert "reasoning_max_tokens" not in body and "reasoning_budget" not in body


def test_default_effort_flows_into_the_derived_budget():
    d = _preset_driver({"default_effort": "medium", "param_map": {"reasoning_max_tokens": "reasoning_budget"}})
    body = d._openai_chat_body({"input": "hi"}, _BigJob(), d.mapping.params)
    assert body["reasoning_effort"] == "medium"
    assert body["reasoning_budget"] == 5000


def test_input_sugar_key_is_not_logged_as_dropped(caplog):
    d = _preset_driver()
    with caplog.at_level("INFO"):
        d._openai_chat_body({"input": "hi", "temperature": 0.5}, _Job(), d.mapping.params)
    assert not any("dropped params" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO"):
        d._openai_chat_body({"input": "hi", "made_up": 1}, _Job(), d.mapping.params)
    assert any("dropped params" in r.message and "made_up" in r.message for r in caplog.records)


# --- raw poll mapping (media) -----------------------------------------------


def queue_backend(*, statuses):
    """A submit/poll server. `statuses` is the sequence returned by successive polls."""
    state = {"polls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/submit":
            return httpx.Response(200, json={"status_url": "http://q/status/1"})
        if req.url.path == "/status/1":
            i = min(state["polls"], len(statuses) - 1)
            state["polls"] += 1
            return httpx.Response(200, json=statuses[i])
        return httpx.Response(404, json={})

    return handler


def raw_media_config():
    return BackendConfig(
        preset=None,
        request={"method": "POST", "url": "http://q/submit", "body": {"prompt": "{input.prompt}"}},
        response={
            "mode": "poll",
            "poll": {"status_url": "$.status_url", "status_field": "$.status",
                     "done_values": ["COMPLETED"], "failed_values": ["FAILED"],
                     "interval_s": 0, "timeout_s": 100},
            "result": {"media_urls": "$.images[*].url"},
        },
        retries=0,
    )


async def test_raw_poll_media_extracts_urls():
    statuses = [
        {"status": "QUEUED"},
        {"status": "COMPLETED", "images": [{"url": "http://cdn/a.png"}, {"url": "http://cdn/b.png"}]},
    ]
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat", "width": 768, "height": 512, "seed": 7})
    assert result.kind == "media"
    assert result.media_urls == ["http://cdn/a.png", "http://cdn/b.png"]
    # The frames are labelled with the dimensions the job was priced against, and
    # the seed the request pinned rides back with them.
    assert (result.width, result.height, result.seed) == (768, 512, 7)
    assert result.duration_secs is None          # an image has no length
    assert result.content_type is None           # this mapping names no type field


async def test_media_dimensions_default_exactly_as_the_billed_pixels_do():
    # A request that names no size is priced at DEFAULT_DIM² per frame, so that is
    # the size the frame must claim — the label and the bill read one source.
    statuses = [{"status": "COMPLETED", "images": [{"url": "http://cdn/a.png"}]}]
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat"})
    assert (result.width, result.height) == (DEFAULT_DIM, DEFAULT_DIM)
    assert result.width * result.height == _frame_pixels({"prompt": "a cat"})


async def test_a_mapping_may_name_the_backends_content_type_and_seed():
    statuses = [{"status": "COMPLETED", "mime": "image/webp", "used_seed": 99,
                 "images": [{"url": "http://cdn/a.webp"}]}]
    config = raw_media_config()
    config.response["result"].update(content_type="$.mime", seed="$.used_seed")
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    # The request pins a different seed; what the backend reports is what produced
    # the pixels, so it wins.
    result = await driver.run(media_job(), {"prompt": "a cat", "seed": 7})
    assert (result.content_type, result.seed) == ("image/webp", 99)


async def test_a_wildcard_content_type_path_states_nothing():
    # `[*]` evaluates to a list, and one frame carries one media type as a string.
    # An operator who writes the wildcard (the media_urls line above it needs one)
    # gets the fallback, never a list sealed into the frame where a string belongs.
    statuses = [{"status": "COMPLETED", "images": [{"url": "http://cdn/a.png", "mime": "image/png"}]}]
    config = raw_media_config()
    config.response["result"]["content_type"] = "$.images[*].mime"
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat"})
    assert result.content_type is None


@pytest.mark.parametrize("images, why", [
    ([{"url": "http://cdn/a.png"}], "a list of objects, not of URLs"),
    ([None], "a null where a URL belongs"),
    (["/relative/a.png"], "a path with no host to fetch it from"),
    (["  "], "blank"),
])
async def test_a_media_urls_path_that_is_not_a_url_fails_the_job(images, why):
    # A JSONPath resolves against whatever the backend returned. A value that cannot
    # be fetched is a backend error — which reports the job and refunds the client —
    # rather than an unhandled error in the fetch loop that strands the escrow.
    statuses = [{"status": "COMPLETED", "images": images}]
    config = raw_media_config()
    config.response["result"]["media_urls"] = "$.images"   # the whole list, unwrapped
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    with pytest.raises(BackendError) as exc:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert "media_urls" in str(exc.value)
    # The offending value never rides along: the reason reaches the network.
    assert "cdn" not in str(exc.value) and "relative" not in str(exc.value)


async def test_a_scalar_media_path_is_one_output_not_a_string_of_characters():
    # `media_urls: "$.images[0].url"` is a legitimate mapping for a backend that
    # returns exactly one frame; iterating its characters would fetch nonsense.
    statuses = [{"status": "COMPLETED", "images": [{"url": "http://cdn/a.png"}]}]
    config = raw_media_config()
    config.response["result"]["media_urls"] = "$.images[0].url"
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat"})
    assert result.media_urls == ["http://cdn/a.png"]


async def test_a_media_b64_path_that_is_not_base64_fails_the_job():
    statuses = [{"status": "COMPLETED", "data": [{"b64": "not base64 at all"}]}]
    config = raw_media_config()
    del config.response["result"]["media_urls"]
    config.response["result"]["media_b64"] = "$.data"
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    with pytest.raises(BackendError) as exc:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert "media_b64" in str(exc.value)


async def test_media_b64_that_decodes_to_nothing_fails_the_job():
    # b64decode silently drops characters outside the alphabet, so a value made
    # only of them yields b"" — an empty frame sealed as if it were a render.
    statuses = [{"status": "COMPLETED", "data": ["###"]}]
    config = raw_media_config()
    del config.response["result"]["media_urls"]
    config.response["result"]["media_b64"] = "$.data[*]"
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    with pytest.raises(BackendError) as exc:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert "no bytes" in str(exc.value)


@pytest.mark.parametrize("status", [404, 503])
async def test_a_backend_failure_reason_carries_the_status_and_not_the_url(status, caplog):
    # The reason becomes the error the coordinator records on the client's job, so
    # the backend's URL — credentials and all — and the upstream body stay in the
    # operator's log, which is where someone debugging the backend is looking.
    def handler(req):
        return httpx.Response(status, text="upstream said: model qwen-x is cold, key sk-live-9f3a")

    config = BackendConfig(preset=None,
                           request={"method": "POST", "url": "http://backend.internal/v1/go?key=sekret",
                                    "body": {"prompt": "{input.prompt}"}},
                           response={"mode": "sync", "result": {"media_b64": "$.data[*]"}},
                           retries=0)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    with caplog.at_level("WARNING"):
        with pytest.raises(BackendError) as exc:
            await driver.run(media_job(), {"prompt": "a cat"})

    wire = str(exc.value)
    assert str(status) in wire
    for secret in ("backend.internal", "sekret", "sk-live-9f3a", "qwen-x"):
        assert secret not in wire
    assert "backend.internal" in caplog.text and "sk-live-9f3a" in caplog.text


async def test_a_video_job_carries_the_length_it_was_priced_for():
    statuses = [{"status": "COMPLETED", "images": [{"url": "http://cdn/a.mp4"}]}]
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    job = media_job()
    job.modality = "video"
    result = await driver.run(job, {"prompt": "a cat", "width": 1280, "height": 720, "duration_secs": 3})
    assert result.duration_secs == 3


async def test_raw_poll_failed_status_raises():
    statuses = [{"status": "FAILED"}]
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    with pytest.raises(BackendError):
        await driver.run(media_job(), {"prompt": "x"})


async def test_raw_poll_timeout_raises():
    statuses = [{"status": "QUEUED"}]  # never completes
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=statuses)))
    # fake clock advances 60s per poll; timeout_s=100 -> gives up
    ticks = iter([0, 60, 120, 180])
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep, clock=lambda: next(ticks))
    with pytest.raises(BackendError, match="timed out|timeout"):
        await driver.run(media_job(), {"prompt": "x"})


def _chat_driver(handler, **limits):
    be = BackendConfig(preset="openai-chat", params={"base_url": "http://r/v1", "model": "m"}, **limits)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BackendDriver.from_config(client, be, sleep=no_sleep)


async def test_a_5xx_is_one_attempt_and_a_retryable_error():
    # The scheduler owns the retry loop; the driver makes one call and says
    # whether another could succeed.
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "warming up"})

    driver = _chat_driver(handler, retries=2)
    with pytest.raises(BackendError) as exc:
        await driver.run(text_job(), {"messages": []})
    assert calls["n"] == 1
    assert exc.value.retryable and exc.value.retry_after_s is None


async def test_a_429_is_retryable_and_carries_retry_after():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow down"})

    with pytest.raises(BackendError) as exc:
        await _chat_driver(handler).run(text_job(), {"messages": []})
    assert exc.value.retryable
    assert exc.value.retry_after_s == 7
    assert "429" in str(exc.value)


async def test_a_429_without_retry_after_leaves_the_delay_to_the_policy():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"})

    with pytest.raises(BackendError) as exc:
        await _chat_driver(handler).run(text_job(), {"messages": []})
    assert exc.value.retryable and exc.value.retry_after_s is None


async def test_any_other_4xx_is_not_retryable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(BackendError) as exc:
        await _chat_driver(handler).run(text_job(), {"messages": []})
    assert not exc.value.retryable


async def test_a_listed_status_is_retried_by_configuration():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="")

    with pytest.raises(BackendError) as exc:
        await _chat_driver(handler, retry_statuses=(404,)).run(text_job(), {"messages": []})
    assert exc.value.retryable
    with pytest.raises(BackendError) as exc:
        await _chat_driver(handler).run(text_job(), {"messages": []})
    assert not exc.value.retryable                       # unlisted, the 4xx rule stands


async def test_a_timeout_is_a_retryable_error():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no bytes", request=req)

    with pytest.raises(BackendError) as exc:
        await _chat_driver(handler).run(text_job(), {"messages": []})
    assert exc.value.retryable


async def test_a_retryable_poll_tick_failure_is_absorbed_and_the_poll_completes():
    # The work is submitted; a status tick that fails must not reach the
    # scheduler's retry, which would submit it again.
    state = {"submits": 0, "polls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/submit":
            state["submits"] += 1
            return httpx.Response(200, json={"status_url": "http://q/status/1"})
        state["polls"] += 1
        if state["polls"] == 1:
            return httpx.Response(503, text="status store restarting")
        return httpx.Response(200, json={"status": "COMPLETED", "images": [{"url": "http://cdn/a.png"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat"})
    assert result.media_urls == ["http://cdn/a.png"]
    assert state == {"submits": 1, "polls": 2}


async def test_a_non_retryable_poll_tick_failure_still_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/submit":
            return httpx.Response(200, json={"status_url": "http://q/status/1"})
        return httpx.Response(404, text="no such task")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    with pytest.raises(BackendError) as exc:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert not exc.value.retryable


async def test_poll_ticks_keep_the_callers_request_timeout():
    # The poll's own `timeout_s` is a wall-clock budget for the whole wait; each
    # tick still runs under the per-request ceiling the caller set.
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.extensions.get("timeout"))
        if req.url.path == "/submit":
            return httpx.Response(200, json={"status_url": "http://q/status/1"})
        return httpx.Response(200, json={"status": "COMPLETED", "images": [{"url": "http://cdn/a.png"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    await driver.run(media_job(), {"prompt": "a cat"}, timeout_s=50)
    assert [t["read"] for t in seen] == [50, 50]
    assert [t["connect"] for t in seen] == [10.0, 10.0]


# --- resumable poll: {handle}, max_polls, resume= ------------------------------


def handle_backend(*, statuses, log):
    """A submit/poll server addressed by the handle the submit returned.
    `log` collects (method, path) of every request."""
    state = {"polls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, req.url.path))
        if req.url.path == "/submit":
            return httpx.Response(200, json={"id": "task_9", "status": "QUEUED"})
        if req.url.path == "/status/task_9":
            i = min(state["polls"], len(statuses) - 1)
            state["polls"] += 1
            return httpx.Response(200, json=statuses[i])
        return httpx.Response(404, json={})

    return handler


def handle_config(**poll_extra):
    return BackendConfig(
        preset=None,
        request={"method": "POST", "url": "http://q/submit", "body": {"prompt": "{input.prompt}"}},
        response={
            "mode": "poll",
            "poll": {"handle": "$.id",
                     "request": {"method": "GET", "url": "http://q/status/{handle}"},
                     "status_field": "$.status",
                     "done_values": ["COMPLETED"], "failed_values": ["FAILED"],
                     **poll_extra},
            "result": {"media_urls": "$.images[*].url"},
        },
        retries=0,
    )


DONE = {"status": "COMPLETED", "images": [{"url": "http://cdn/a.png"}]}


async def test_the_handle_is_reported_before_the_poll_and_bound_in_its_templates():
    log, handles = [], []
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle_backend(statuses=[DONE], log=log)))
    driver = BackendDriver.from_config(client, handle_config(interval_s=0), sleep=no_sleep)
    assert driver.resumable
    result = await driver.run(media_job(), {"prompt": "a cat"}, on_handle=handles.append)
    assert handles == ["task_9"]
    assert log == [("POST", "/submit"), ("GET", "/status/task_9")]
    assert result.media_urls == ["http://cdn/a.png"]


async def test_resume_skips_the_submit_and_polls_the_stored_handle():
    log, handles = [], []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        handle_backend(statuses=[{"status": "RUNNING"}, DONE], log=log)))
    driver = BackendDriver.from_config(client, handle_config(interval_s=0), sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat"}, resume="task_9", on_handle=handles.append)
    assert handles == []                                   # nothing was submitted
    assert log == [("GET", "/status/task_9"), ("GET", "/status/task_9")]
    assert result.media_urls == ["http://cdn/a.png"]


async def test_a_submit_that_yields_no_handle_fails_the_job_without_a_retry():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "QUEUED"})     # no id

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, handle_config(interval_s=0), sleep=no_sleep)
    with pytest.raises(BackendError, match="handle") as exc:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert not exc.value.retryable


async def test_a_mapping_without_a_handle_is_not_resumable():
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=[DONE])))
    driver = BackendDriver.from_config(client, raw_media_config(), sleep=no_sleep)
    assert not driver.resumable


async def test_the_poll_interval_is_the_sla_window_over_max_polls():
    sleeps = []

    async def sleep(s):
        sleeps.append(s)

    def fresh_client():   # every run gets its own poll counter
        return httpx.AsyncClient(transport=httpx.MockTransport(
            handle_backend(statuses=[{"status": "RUNNING"}, {"status": "RUNNING"}, DONE], log=[])))

    driver = BackendDriver.from_config(fresh_client(), handle_config(max_polls=60), sleep=sleep)
    await driver.run(media_job(), {"prompt": "a cat"})          # media_job() is a 24h job
    assert sleeps == [86_400 / 60, 86_400 / 60]

    sleeps.clear()
    driver = BackendDriver.from_config(fresh_client(), handle_config(), sleep=sleep)  # default max_polls
    await driver.run(text_job(), {"prompt": "a cat"})           # a 1h job
    assert sleeps == [60, 60]


async def test_an_explicit_interval_overrides_the_derived_one():
    sleeps = []

    async def sleep(s):
        sleeps.append(s)

    client = httpx.AsyncClient(transport=httpx.MockTransport(
        handle_backend(statuses=[{"status": "RUNNING"}, DONE], log=[])))
    driver = BackendDriver.from_config(client, handle_config(interval_s=5, max_polls=60), sleep=sleep)
    await driver.run(media_job(), {"prompt": "a cat"})
    assert sleeps == [5]


async def test_a_poll_without_its_own_wall_waits_as_long_as_the_caller_allows():
    # No `timeout_s` on the poll: the scheduler's deadline wall bounds the wait,
    # so a clock that jumps hours between ticks does not end it here.
    ticks = iter([0, 5_000, 10_000, 15_000, 20_000, 25_000])
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        handle_backend(statuses=[{"status": "RUNNING"}, {"status": "RUNNING"}, DONE], log=[])))
    driver = BackendDriver.from_config(client, handle_config(interval_s=0), sleep=no_sleep,
                                       clock=lambda: next(ticks))
    result = await driver.run(media_job(), {"prompt": "a cat"})
    assert result.media_urls == ["http://cdn/a.png"]


# -- plan_media_units: units_out is a cap; clamp count/duration, settle actual ---

def _img_job(units_out):
    return EvmJob(job_id="job_i", model="black-forest-labs/flux-2-dev:fp8", modality="image", state="Claimed",
                  sla="24h", created_at=0, units_out=units_out)


def _vid_job(units_out):
    return EvmJob(job_id="job_v", model="wan-ai/wan-2-6:fp8", modality="video", state="Claimed",
                  sla="24h", created_at=0, units_out=units_out)


_HD = 1024 * 1024  # pixels in a 1024×1024 image


def test_plan_units_within_cap_passes_through():
    inp = {"prompt": "x", "width": 1024, "height": 1024, "num_images": 2}
    out, units = plan_media_units(_img_job(2 * _HD), inp)
    assert out["num_images"] == 2
    assert units == 2 * _HD


def test_plan_units_clamps_overask_down():
    # paid for 2 × 1024² pixels, asked for 4 → clamp to 2 images, settle actual 2 × 1024²
    inp = {"prompt": "x", "width": 1024, "height": 1024, "num_images": 4}
    out, units = plan_media_units(_img_job(2 * _HD), inp)
    assert out["num_images"] == 2
    assert units == 2 * _HD


def test_plan_units_fails_when_one_unit_exceeds_cap():
    # a single 1024² image is 1_048_576 px; a cap of half that covers none → fail
    with pytest.raises(BackendError):
        plan_media_units(_img_job(_HD // 2), {"prompt": "x", "width": 1024, "height": 1024, "num_images": 1})


def test_plan_units_defaults_dimensions_when_absent():
    # no width/height → defaults to 1024×1024, so one image fits a 1024² cap
    out, units = plan_media_units(_img_job(_HD), {"prompt": "x", "num_images": 1})
    assert out.get("num_images", 1) == 1
    assert units == _HD


def test_plan_units_clamps_video_duration():
    # per-frame pixels × duration; cap = 5 × 1024² → asked 10s clamps to 5s
    out, units = plan_media_units(_vid_job(5 * _HD), {"prompt": "x", "width": 1024, "height": 1024, "duration_secs": 10})
    assert out["duration_secs"] == 5
    assert units == 5 * _HD


def test_plan_units_applies_param_caps():
    """param_caps bounds compute drivers (steps/fps) that the pixel unit does not price."""
    inp = {"prompt": "x", "width": 1024, "height": 1024, "num_images": 1, "steps": 90}
    out, units = plan_media_units(_img_job(_HD), inp, caps={"steps": 40})
    assert out["steps"] == 40
    assert units == _HD                       # the billed unit is untouched


def test_plan_units_param_caps_leave_lower_values_alone():
    inp = {"prompt": "x", "width": 1024, "height": 1024, "num_images": 1, "steps": 20}
    out, _ = plan_media_units(_img_job(_HD), inp, caps={"steps": 40, "fps": 24})
    assert out["steps"] == 20
    assert "fps" not in out                   # a cap never injects a param


# -- input_shortfall: units_in is declared, billed, and until now unchecked -----
#
# The chain bills `rateIn·unitsIn` whatever the client wrote, and there is no
# delivered-input quantity to clamp it against — so a provider's only move is to
# decline. This is the predicate that decides when to.


def _text_job(units_in, *, rate_in=200_000, modality="text"):
    return EvmJob(job_id="job_t", model="deepseek-ai/deepseek-v3", modality=modality, state="Open", sla="1h",
                  created_at=0, rate_in=rate_in, rate_out=600_000,
                  units_in=units_in, units_out=128)


def test_a_declaration_that_covers_the_payload_is_not_short():
    assert input_shortfall(_text_job(10_000), 40_000, bytes_per_unit=16, modality="text") == 0


def test_the_shortfall_is_the_bytes_past_what_was_paid_for():
    """Pinned as arithmetic, not as `> 0`: the number reaches an operator's log."""
    assert input_shortfall(_text_job(1), 1_000_000, bytes_per_unit=16, modality="text") == (
        1_000_000 - ENVELOPE_SLACK_BYTES - 16
    )


def test_a_payload_inside_the_envelope_slack_is_never_short():
    """What the slack is for: the envelope frame and a `custom_id` of unknown width
    are both invisible from outside, so every measurement is forgiven that much.
    """
    assert input_shortfall(_text_job(0), ENVELOPE_SLACK_BYTES, bytes_per_unit=16, modality="text") == 0


@pytest.mark.parametrize("modality", ["image", "video"])
def test_a_pixel_metered_modality_is_never_judged_on_its_payload_bytes(modality):
    """`units_in` is not denominated in bytes on every modality, so the floor must
    not be applied on every modality.

    For text and embeddings the clients declare one unit per four bytes of input,
    which is what makes a byte measurement a floor at all. For image and video the
    unit is a pixel-second of *reference* — a number with no relation to how large
    the payload is, since a sharper reference of identical dimensions costs more
    bytes and buys nothing. Judging those by weight would decline honest media work
    for carrying its own input, so they are checked by re-deriving the reference
    instead, after decryption, where the real dimensions are knowable.

    Today no shipped media model quotes an input rate at all, so `rate_in` alone
    happens to exempt them. That is a coincidence about the catalog and not a
    property of the design — the moment a media model prices its input side, this
    is the gate that has to hold.
    """
    job = _text_job(0, modality=modality)
    assert input_shortfall(job, 1_000_000, bytes_per_unit=16, modality=modality) == 0


def test_a_bid_whose_window_meters_no_input_is_never_short():
    """Media quotes an output rate and no input one. With `rate_in` unset the input
    leg bills nothing whatever `units_in` says, so there is nothing to be short of —
    and judging it would refuse honest media work for a number nobody is charged for.
    """
    assert input_shortfall(_text_job(None, rate_in=None), 1_000_000, bytes_per_unit=16, modality="text") == 0
    assert input_shortfall(_text_job(1, rate_in=0), 1_000_000, bytes_per_unit=16, modality="text") == 0


def test_a_row_that_carries_no_units_in_is_not_judged():
    """Fail open. `units_in` is a signed order term the coordinator has no reason to
    drop, so its absence means something changed upstream — and the worst answer to
    that is a daemon that silently stops claiming everything.
    """
    assert input_shortfall(_text_job(None), 1_000_000, bytes_per_unit=16, modality="text") == 0


def test_a_zero_bytes_per_unit_turns_the_check_off():
    assert input_shortfall(_text_job(1), 10 * 1024 * 1024, bytes_per_unit=0, modality="text") == 0


def test_a_tighter_allowance_finds_what_a_looser_one_forgives():
    """The knob's direction, pinned: lower means stricter."""
    # 40 KB less the slack is 35_904 bytes to cover: 3_000 units buy 48_000 of
    # them at 16 and only 12_000 at 4.
    job, payload = _text_job(3_000), 40_000
    assert input_shortfall(job, payload, bytes_per_unit=4, modality="text") > 0
    assert input_shortfall(job, payload, bytes_per_unit=16, modality="text") == 0


# --- openai-chat preset: streaming -------------------------------------------


def _sse(*events) -> bytes:
    return "".join(
        f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events
    ).encode()


def _stream_driver(handler):
    be = BackendConfig(preset="openai-chat", stream=True, params={
        "base_url": "http://runtime/v1", "model": "runtime-model", "api_key": "sk-x",
    })
    return BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)


async def test_stream_reassembles_the_completion_and_keeps_the_usage_block():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(
            {"id": "c1", "object": "chat.completion.chunk", "created": 7, "model": "runtime-model",
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"id": "c1", "choices": [{"index": 0, "delta": {"reasoning_content": "let me see"}}]},
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Paris"}}]},
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": " it is."},
                                      "finish_reason": "stop"}]},
            {"id": "c1", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 7}},
            "[DONE]",
        ))

    # The client's own `stream: false` is dropped (NEVER_FORWARDED); the operator's switch wins.
    result = await _stream_driver(handler).run(
        text_job(), {"messages": [{"role": "user", "content": "hey"}], "stream": False})

    body = captured["body"]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert (result.kind, result.text, result.completion_tokens) == ("text", "Paris it is.", 7)
    choice = result.raw["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Paris it is.",
                                 "reasoning_content": "let me see"}
    assert choice["finish_reason"] == "stop"
    assert (result.raw["id"], result.raw["object"], result.raw["model"]) == (
        "c1", "chat.completion", "runtime-model")
    assert result.raw["usage"]["completion_tokens"] == 7


async def test_a_stream_that_ends_without_finishing_is_retryable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Par"}}]}))

    with pytest.raises(BackendError, match="closed the stream early") as info:
        await _stream_driver(handler).run(text_job(), {"messages": []})
    assert info.value.retryable


async def test_a_backend_that_fails_mid_stream_is_retryable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Par"}}]},
            {"error": {"message": "worker died", "code": 500}},
        ))

    with pytest.raises(BackendError, match="mid-stream") as info:
        await _stream_driver(handler).run(text_job(), {"messages": []})
    assert info.value.retryable


async def test_a_streamed_request_classifies_statuses_like_a_plain_one():
    def throttled(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, content=b'{"status":429}')

    with pytest.raises(BackendError, match="HTTP 429") as info:
        await _stream_driver(throttled).run(text_job(), {"messages": []})
    assert info.value.retryable and info.value.retry_after_s == 7

    def refused(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b'{"error":"bad"}')

    with pytest.raises(BackendError, match="HTTP 400") as info:
        await _stream_driver(refused).run(text_job(), {"messages": []})
    assert not info.value.retryable


async def test_stream_is_the_operators_switch_not_the_clients():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}],
                                         "usage": {"completion_tokens": 1}})

    be = BackendConfig(preset="openai-chat", params={"base_url": "http://runtime/v1", "model": "m"})
    driver = BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)
    await driver.run(text_job(), {"messages": [], "stream": True,
                                  "stream_options": {"include_usage": True}})
    assert "stream" not in captured["body"] and "stream_options" not in captured["body"]


async def test_a_stream_that_ends_without_its_usage_block_is_retryable():
    # `[DONE]` and a finish reason, but the block the settle needs never came:
    # a plain response without usage is abandoned by the scheduler's guard, a
    # streamed one asked for the block and is turned over instead.
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Paris"}, "finish_reason": "stop"}]},
            "[DONE]",
        ))

    with pytest.raises(BackendError, match="no usage block") as info:
        await _stream_driver(handler).run(text_job(), {"messages": []})
    assert info.value.retryable


# --- openai-responses preset (background submit → poll by id) ------------------


RESPONSE_DONE = {
    "id": "resp_1", "object": "response", "created_at": 1700000000, "model": "rt",
    "status": "completed",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think "}]},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Hello", "annotations": []},
                     {"type": "output_text", "text": " world", "annotations": []}]},
    ],
    "usage": {"input_tokens": 7, "output_tokens": 12, "total_tokens": 19},
}


def responses_backend(*, statuses, log, captured):
    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, req.url.path, req.headers.get("Authorization")))
        if req.method == "POST" and req.url.path == "/v1/responses":
            captured["body"] = json.loads(req.content)
            return httpx.Response(200, json={"id": "resp_1", "object": "response", "status": "queued"})
        if req.method == "GET" and req.url.path == "/v1/responses/resp_1":
            i = min(captured.setdefault("polls", 0), len(statuses) - 1)
            captured["polls"] += 1
            return httpx.Response(200, json=statuses[i])
        return httpx.Response(404, json={})

    return handler


def _responses_driver(handler, sleep=no_sleep, **params):
    be = BackendConfig(preset="openai-responses", params={
        "base_url": "http://runtime/v1", "model": "rt", "api_key": "sk-x", **params})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BackendDriver.from_config(client, be, sleep=sleep)


def test_expand_openai_responses_shape():
    be = BackendConfig(preset="openai-responses",
                       params={"base_url": "http://r/v1/", "model": "m", "api_key": "k", "max_polls": 10})
    mapping = expand_preset(be)
    assert mapping.request == {"method": "POST", "url": "http://r/v1/responses",
                               "headers": {"Content-Type": "application/json", "Authorization": "Bearer k"}}
    poll = mapping.response["poll"]
    assert mapping.response["mode"] == "poll"
    assert poll["handle"] == "$.id"
    assert poll["request"] == {"method": "GET", "url": "http://r/v1/responses/{handle}",
                               "headers": {"Content-Type": "application/json", "Authorization": "Bearer k"}}
    assert poll["done_values"] == ["completed"]
    assert set(poll["failed_values"]) == {"failed", "cancelled", "incomplete"}
    assert poll["max_polls"] == 10
    assert mapping.response["result"] == {"text": "$.choices[0].message.content",
                                          "completion_tokens": "$.usage.completion_tokens"}


async def test_openai_responses_submits_in_the_background_then_polls_by_id():
    log, captured, handles = [], {}, []
    driver = _responses_driver(
        responses_backend(statuses=[{"id": "resp_1", "status": "in_progress"}, RESPONSE_DONE],
                          log=log, captured=captured),
        service_tier="flex")
    assert driver.resumable
    result = await driver.run(text_job(), {"messages": [{"role": "user", "content": "hi"}],
                                           "temperature": 0.2, "n": 3},
                              on_handle=handles.append)

    body = captured["body"]
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert "messages" not in body and "max_tokens" not in body
    assert body["max_output_tokens"] == 128            # the job's units_out
    assert body["model"] == "rt"
    assert body["service_tier"] == "flex" and body["background"] is True
    assert body["temperature"] == 0.2 and "n" not in body   # the chat allowlist still applies
    assert handles == ["resp_1"]
    assert [(m, p) for m, p, _ in log] == [("POST", "/v1/responses"),
                                           ("GET", "/v1/responses/resp_1"),
                                           ("GET", "/v1/responses/resp_1")]
    assert {a for _, _, a in log} == {"Bearer sk-x"}     # the poll carries the same auth

    assert result.kind == "text"
    assert result.text == "Hello world"
    assert result.completion_tokens == 12
    # The sealed object is the chat-completion shape a client already reads.
    assert result.raw["object"] == "chat.completion"
    assert result.raw["choices"][0]["message"] == {"role": "assistant", "content": "Hello world",
                                                   "reasoning_content": "think "}
    assert result.raw["usage"] == {"prompt_tokens": 7, "completion_tokens": 12, "total_tokens": 19}
    assert result.raw["id"] == "resp_1" and result.raw["model"] == "rt"


async def test_openai_responses_omits_the_tier_when_none_is_configured():
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured))
    await driver.run(text_job(), {"input": "hi"})
    assert "service_tier" not in captured["body"]
    assert captured["body"]["input"] == [{"role": "user", "content": "hi"}]   # the messages sugar


async def test_openai_responses_extra_params_override_the_presets_own_keys():
    # `extra_params` is documented as overriding everything, and this preset sets
    # two keys of its own after the chat body is built — so they are re-applied
    # last or the escape hatch is silently ignored on exactly this preset.
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured),
                               service_tier="flex",
                               extra_params={"background": False, "service_tier": "priority"})
    await driver.run(text_job(), {"input": "hi"})
    assert captured["body"]["background"] is False
    assert captured["body"]["service_tier"] == "priority"


async def test_openai_responses_extra_params_are_respelled_for_the_endpoint():
    # `max_tokens` and `messages` are the chat dialect's names for two keys this
    # preset renames. Re-applied verbatim they would sit beside the renamed ones
    # and the endpoint would answer 400.
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured),
                               extra_params={"max_tokens": 7})
    await driver.run(text_job(), {"input": "hi"})
    assert captured["body"]["max_output_tokens"] == 7
    assert "max_tokens" not in captured["body"]


async def test_openai_responses_resumes_without_a_second_submit():
    log, captured = [], {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=log, captured=captured))
    result = await driver.run(text_job(), {"input": "hi"}, resume="resp_1")
    assert "body" not in captured
    assert [(m, p) for m, p, _ in log] == [("GET", "/v1/responses/resp_1")]
    assert result.text == "Hello world"


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete"])
async def test_openai_responses_failure_states_fail_the_job(status):
    driver = _responses_driver(responses_backend(
        statuses=[{"id": "resp_1", "status": status, "output": []}], log=[], captured={}))
    with pytest.raises(BackendError, match=status) as exc:
        await driver.run(text_job(), {"input": "hi"})
    assert not exc.value.retryable


def test_responses_to_chat_handles_a_bare_object():
    out = responses_to_chat({"id": "resp_2", "status": "completed", "output": [], "usage": {}})
    assert out["choices"][0]["message"] == {"role": "assistant", "content": ""}
    assert "usage" not in out         # no output count → extraction reports None → the job fails closed


async def test_openai_responses_refuses_a_payload_without_a_prompt():
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured))
    with pytest.raises(BackendError, match="messages or input") as exc:
        await driver.run(text_job(), {"temperature": 0.2})
    assert not exc.value.retryable
    assert "body" not in captured        # refused before the round trip


@pytest.mark.parametrize("final", [
    {"id": "resp_1", "status": "completed", "output": 5},
    {"id": "resp_1", "status": "completed", "output": [{"type": "message", "content": "text"}]},
])
async def test_openai_responses_fails_closed_on_an_unreadable_output(final):
    driver = _responses_driver(responses_backend(statuses=[final], log=[], captured={}))
    with pytest.raises(BackendError, match="unreadable") as exc:
        await driver.run(text_job(), {"input": "hi"})
    assert not exc.value.retryable


async def test_openai_responses_respells_the_chat_dialect():
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured),
                               extra_params={"reasoning": {"summary": "auto"}})
    await driver.run(text_job(), {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "low",
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True}},
        "tools": [{"type": "function", "function": {"name": "f", "description": "d",
                                                    "parameters": {"type": "object"}}},
                  {"type": "function", "name": "already_flat", "parameters": {}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
        "top_k": 5, "seed": 1, "reasoning_max_tokens": 50,
    })
    body = captured["body"]
    assert "reasoning_effort" not in body
    assert body["reasoning"] == {"summary": "auto", "effort": "low"}     # merged, not replaced
    assert "response_format" not in body
    assert body["text"] == {"format": {"type": "json_schema", "name": "answer",
                                       "schema": {"type": "object"}, "strict": True}}
    assert body["tools"] == [{"type": "function", "name": "f", "description": "d",
                              "parameters": {"type": "object"}},
                             {"type": "function", "name": "already_flat", "parameters": {}}]
    assert body["tool_choice"] == {"type": "function", "name": "f"}
    # The default allowlist is the Responses vocabulary: chat-only knobs never leave.
    assert "top_k" not in body and "seed" not in body
    # The budget has no field on this surface, and no name to be respelled to.
    assert "reasoning_max_tokens" not in body


async def test_openai_responses_json_object_format_passes_through():
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured))
    await driver.run(text_job(), {"input": "hi", "response_format": {"type": "json_object"}})
    assert captured["body"]["text"] == {"format": {"type": "json_object"}}


async def test_openai_responses_extra_params_are_respelled_and_win():
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured),
                               service_tier="flex",
                               extra_params={"reasoning_effort": "high", "max_tokens": 5,
                                             "background": False, "service_tier": "priority"})
    await driver.run(text_job(), {"input": "hi", "reasoning_effort": "low"})
    body = captured["body"]
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_output_tokens"] == 5 and "max_tokens" not in body
    assert body["background"] is False and body["service_tier"] == "priority"
    assert "reasoning_effort" not in body


def test_responses_to_chat_carries_tool_calls_and_reasoning_text():
    final = {
        "id": "resp_2", "object": "response", "created_at": 1, "model": "rt", "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs", "summary": [],
             "content": [{"type": "reasoning_text", "text": "I should call f."}]},
            {"type": "message", "id": "m", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "\n\n"}]},
            {"type": "function_call", "id": "fc", "call_id": "call_1", "name": "f",
             "arguments": "{\"x\":1}", "status": "completed"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 30, "total_tokens": 40,
                  "output_tokens_details": {"reasoning_tokens": 12}},
    }
    chat = responses_to_chat(final)
    choice = chat["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "\n\n"
    assert choice["message"]["reasoning_content"] == "I should call f."
    assert choice["message"]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1}"}}]
    assert chat["usage"] == {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40,
                             "completion_tokens_details": {"reasoning_tokens": 12}}


def test_responses_to_chat_prefers_the_summary_and_stringifies_object_arguments():
    final = {"id": "r", "status": "completed", "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "S"}],
         "content": [{"type": "reasoning_text", "text": "full trace"}]},
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": {"x": 1}},
    ], "usage": {"output_tokens": 1}}
    chat = responses_to_chat(final)
    msg = chat["choices"][0]["message"]
    assert msg["reasoning_content"] == "S"
    assert msg["tool_calls"][0]["function"]["arguments"] == "{\"x\": 1}"
    assert "completion_tokens_details" not in chat["usage"]


async def test_openai_responses_extra_params_in_the_responses_spelling_are_not_clobbered():
    # An operator who writes the endpoint's own spelling must not have the payload
    # land on top of it: the preset's `max_tokens` renamed over their
    # `max_output_tokens` would be the escape hatch overriding nothing.
    captured = {}
    driver = _responses_driver(responses_backend(statuses=[RESPONSE_DONE], log=[], captured=captured),
                               extra_params={"max_output_tokens": 7,
                                             "input": [{"role": "system", "content": "s"}],
                                             "reasoning": {"effort": "high"}})
    await driver.run(text_job(), {"messages": [{"role": "user", "content": "hi"}],
                                  "reasoning_effort": "low"})
    body = captured["body"]
    assert body["max_output_tokens"] == 7 and "max_tokens" not in body
    assert body["input"] == [{"role": "system", "content": "s"}]
    assert body["reasoning"] == {"effort": "high"}


def test_responses_to_chat_omits_a_tool_call_id_it_was_not_given():
    chat = responses_to_chat({"id": "r", "status": "completed", "usage": {"output_tokens": 1},
                              "output": [{"type": "function_call", "name": "f", "arguments": "{}"}]})
    assert chat["choices"][0]["message"]["tool_calls"] == [
        {"type": "function", "function": {"name": "f", "arguments": "{}"}}]


# --- openai-batch preset (one job = one batch of one line) ------------------

BATCH_JOB_ID = "0x" + "ab" * 32
BATCH_CUSTOM_ID = "ab" * 32          # the id without its prefix, inside the 64-char cap

CHAT_BODY_DONE = {"id": "gen-1", "object": "chat.completion", "model": "rt",
                  "choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant", "content": "hello"}}],
                  "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}}


def batch_job():
    return EvmJob(job_id=BATCH_JOB_ID, model="m:fp8", modality="text", state="Claimed",
                  sla="24h", created_at=0, units_out=128)


BATCH_OBJECT = {"id": "batch_1", "input_file_id": "file_in",
                "output_file_id": "file_out", "error_file_id": "file_err"}


def _content_page(pages, skip):
    """One page of a JSONL content file, `X-Incomplete` while more follow."""
    start = 0
    for i, page in enumerate(pages):
        if start == skip:
            headers = {"X-Incomplete": "true"} if i + 1 < len(pages) else {}
            return httpx.Response(200, headers=headers,
                                  text="".join(json.dumps(l) + "\n" for l in page))
        start += len(page)
    return httpx.Response(200, text="")


def batch_backend(*, statuses, output_lines=(), output_pages=None, error_lines=(),
                  output_status=200, delete_status=200, log, captured):
    """A Files + Batches backend: upload, create, status, the two content files
    and the input file's delete.

    `output_pages` serves the output file one page at a time — each page but the
    last answers `X-Incomplete: true` and the next is asked for with `?skip=`.
    `output_status` other than 200 is what the output file answers instead of
    its content; `delete_status` is what the input file's DELETE answers.
    """
    pages = output_pages if output_pages is not None else [list(output_lines)]

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, req.url.path))
        if req.method == "POST" and req.url.path == "/v1/files":
            captured["upload_content_type"] = req.headers.get("content-type", "")
            captured["upload"] = req.content.decode()
            return httpx.Response(201, json={"id": "file_in", "object": "file", "purpose": "batch"})
        if req.method == "POST" and req.url.path == "/v1/batches":
            captured["create"] = json.loads(req.content)
            return httpx.Response(201, json={**BATCH_OBJECT, "status": "validating"})
        if req.method == "GET" and req.url.path == "/v1/batches/batch_1":
            i = min(captured.setdefault("polls", 0), len(statuses) - 1)
            captured["polls"] += 1
            return httpx.Response(200, json={**BATCH_OBJECT, "status": statuses[i]})
        if req.method == "GET" and req.url.path == "/v1/files/file_out/content":
            skip = req.url.params.get("skip")
            captured.setdefault("content_skips", []).append(skip)
            if output_status != 200:
                return httpx.Response(output_status, json={"error": "no such file"})
            return _content_page(pages, int(skip or 0))
        if req.method == "GET" and req.url.path == "/v1/files/file_err/content":
            return httpx.Response(200, text="".join(json.dumps(l) + "\n" for l in error_lines))
        if req.method == "DELETE" and req.url.path == "/v1/files/file_in":
            return httpx.Response(delete_status, json={"id": "file_in", "deleted": True})
        return httpx.Response(404, json={})
    return handler


def _batch_driver(handler, **params):
    be = BackendConfig(preset="openai-batch", params={
        "base_url": "http://runtime/v1", "model": "rt", "api_key": "sk-x", **params})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BackendDriver.from_config(client, be, sleep=no_sleep)


def test_expand_openai_batch_shape():
    be = BackendConfig(preset="openai-batch",
                       params={"base_url": "http://r/v1/", "model": "m", "api_key": "k", "max_polls": 288})
    mapping = expand_preset(be)
    assert mapping.request["url"] == "http://r/v1/batches"
    poll = mapping.response["poll"]
    assert mapping.response["mode"] == "poll" and poll["handle"] == "$.id"
    assert poll["request"]["url"] == "http://r/v1/batches/{handle}"
    assert poll["request"]["headers"]["Authorization"] == "Bearer k"
    assert poll["done_values"] == ["completed"]
    assert set(poll["failed_values"]) == {"failed", "cancelled", "expired"}
    assert poll["max_polls"] == 288
    assert mapping.response["result"]["text"] == "$.choices[0].message.content"


async def test_openai_batch_uploads_one_line_creates_polls_and_collects():
    log, captured, handles = [], {}, []
    ok_line = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 200, "body": CHAT_BODY_DONE},
               "error": None}
    driver = _batch_driver(batch_backend(statuses=["validating", "in_progress", "completed"],
                                         output_lines=[ok_line], log=log, captured=captured),
                           completion_window="24h")
    assert driver.resumable
    result = await driver.run(batch_job(), {"input": "hi", "temperature": 0.1, "n": 2},
                              on_handle=handles.append)

    assert captured["upload_content_type"].startswith("multipart/form-data")
    assert 'name="purpose"\r\n\r\nbatch' in captured["upload"]
    line_json = next(l for l in captured["upload"].splitlines() if l.startswith("{"))
    line = json.loads(line_json)
    assert line["custom_id"] == BATCH_CUSTOM_ID
    assert line["method"] == "POST" and line["url"] == "/v1/chat/completions"
    assert line["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert line["body"]["model"] == "rt" and line["body"]["max_tokens"] == 128
    assert line["body"]["temperature"] == 0.1 and "n" not in line["body"]   # the chat schema, verbatim
    assert captured["create"] == {"input_file_id": "file_in", "endpoint": "/v1/chat/completions",
                                  "completion_window": "24h"}
    assert handles == ["batch_1"]
    assert log == [("POST", "/v1/files"), ("POST", "/v1/batches"),
                   ("GET", "/v1/batches/batch_1"), ("GET", "/v1/batches/batch_1"),
                   ("GET", "/v1/batches/batch_1"), ("GET", "/v1/files/file_out/content"),
                   ("DELETE", "/v1/files/file_in")]
    assert result.kind == "text" and result.text == "hello" and result.completion_tokens == 2
    assert result.raw == CHAT_BODY_DONE


async def test_openai_batch_custom_id_is_the_job_id_within_the_cap():
    from vorqd.backend import _batch_custom_id
    assert _batch_custom_id(batch_job()) == BATCH_CUSTOM_ID
    assert _batch_custom_id(EvmJob(job_id="plain-id", model="m", modality="text", state="Claimed",
                                   sla="1h", created_at=0)) == "plain-id"
    assert len(_batch_custom_id(EvmJob(job_id="x" * 90, model="m", modality="text", state="Claimed",
                                       sla="1h", created_at=0))) == 64


async def test_openai_batch_resumes_from_the_batch_id_without_uploading():
    log, captured = [], {}
    ok_line = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 200, "body": CHAT_BODY_DONE}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_lines=[ok_line],
                                         log=log, captured=captured))
    result = await driver.run(batch_job(), {"input": "hi"}, resume="batch_1")
    assert result.text == "hello"
    assert "create" not in captured          # nothing uploaded, no second batch created
    # Resumed, so nothing here uploaded the input file — the batch object names
    # it all the same, and a next boot is the only thing left to clean it up.
    assert log == [("GET", "/v1/batches/batch_1"), ("GET", "/v1/files/file_out/content"),
                   ("DELETE", "/v1/files/file_in")]


async def test_openai_batch_a_content_fetch_fault_is_absorbed_not_resubmitted():
    # The batch has run and been billed by the time its output file is read; a
    # retryable fault that reached the scheduler would upload and create a
    # second batch for work already done.
    log, captured = [], {}
    ok_line = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 200, "body": CHAT_BODY_DONE}}
    content = batch_backend(statuses=["completed"], output_lines=[ok_line],
                            log=log, captured=captured)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == "/v1/files/file_out/content" \
                and not captured.get("fetch_failed"):
            captured["fetch_failed"] = True
            log.append((req.method, req.url.path))
            return httpx.Response(502, text="bad gateway")
        return content(req)

    result = await _batch_driver(handler).run(batch_job(), {"input": "hi"})
    assert result.text == "hello"
    assert log.count(("POST", "/v1/files")) == 1
    assert log.count(("POST", "/v1/batches")) == 1
    assert log.count(("GET", "/v1/files/file_out/content")) == 2


async def test_openai_batch_refuses_a_promptless_payload_before_uploading():
    log = []

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, req.url.path))
        return httpx.Response(500, text="must not be reached")

    with pytest.raises(BackendError) as exc:
        await _batch_driver(handler).run(batch_job(), {"seed": 1})
    assert "no messages or input" in str(exc.value)
    assert log == []


async def test_openai_batch_a_failed_line_fails_the_job_without_retry():
    err_line = {"custom_id": BATCH_CUSTOM_ID, "response": None,
                "error": {"code": None, "message": "{\"type\":\"NonRetriableHttpStatus\"}"}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_lines=[], error_lines=[err_line],
                                         log=[], captured={}))
    with pytest.raises(BackendError, match="the batch line failed") as exc:
        await driver.run(batch_job(), {"input": "hi"})
    assert not exc.value.retryable
    assert "NonRetriableHttpStatus" not in str(exc.value)     # the upstream detail stays in the log


async def test_openai_batch_a_non_200_line_fails_the_job():
    bad = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 422, "body": {"error": "x"}}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_lines=[bad], log=[], captured={}))
    with pytest.raises(BackendError, match="the batch line failed") as exc:
        await driver.run(batch_job(), {"input": "hi"})
    assert not exc.value.retryable


async def test_openai_batch_no_line_anywhere_fails_the_job():
    driver = _batch_driver(batch_backend(statuses=["completed"], output_lines=[], log=[], captured={}))
    with pytest.raises(BackendError, match="no result") as exc:
        await driver.run(batch_job(), {"input": "hi"})
    assert not exc.value.retryable


async def test_openai_batch_a_failed_status_fails_the_job():
    captured = {}
    driver = _batch_driver(batch_backend(statuses=["expired"], output_lines=[], log=[],
                                         captured=captured))
    with pytest.raises(BackendError, match="expired"):
        await driver.run(batch_job(), {"input": "hi"})
    # The entry named no window: the preset's own default is what was created.
    assert captured["create"]["completion_window"] == "24h"


async def test_openai_batch_upload_refusal_is_not_retried_and_creates_nothing():
    log = []

    def handler(req):
        log.append((req.method, req.url.path))
        return httpx.Response(403, text="Line 1: model not available")
    driver = _batch_driver(handler)
    with pytest.raises(BackendError) as exc:
        await driver.run(batch_job(), {"input": "hi"})
    assert not exc.value.retryable
    assert log == [("POST", "/v1/files")]


async def test_openai_batch_an_unreadable_output_file_falls_through_to_the_error_file():
    # A `completed` batch whose output file answers 404 is not the job's verdict:
    # the error file is where the refusal lives, and it has not been read yet.
    log, captured = [], {}
    err_line = {"custom_id": BATCH_CUSTOM_ID, "response": None,
                "error": {"code": None, "message": "{\"type\":\"NonRetriableHttpStatus\"}"}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_status=404,
                                         error_lines=[err_line], log=log, captured=captured))
    with pytest.raises(BackendError, match="the batch line failed") as exc:
        await driver.run(batch_job(), {"input": "hi"})
    assert not exc.value.retryable
    assert "NonRetriableHttpStatus" not in str(exc.value)
    assert log.count(("GET", "/v1/files/file_out/content")) == 1     # refused once, not retried
    assert log.count(("GET", "/v1/files/file_err/content")) == 1


async def test_openai_batch_an_unreadable_output_file_and_an_empty_error_file_is_no_result():
    log, captured = [], {}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_status=404,
                                         log=log, captured=captured))
    with pytest.raises(BackendError, match="no result") as exc:
        await driver.run(batch_job(), {"input": "hi"})
    assert not exc.value.retryable
    assert log.count(("GET", "/v1/files/file_out/content")) == 1
    assert log.count(("GET", "/v1/files/file_err/content")) == 1


async def test_openai_batch_pages_a_partial_content_file():
    # A surface that answers a slice of the file says so in `X-Incomplete` and
    # takes `?skip=` for the rest; the job's line is only on the second page.
    log, captured = [], {}
    other = {"custom_id": "cd" * 32, "response": {"status_code": 200, "body": {}}}
    ok_line = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 200, "body": CHAT_BODY_DONE}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_pages=[[other], [ok_line]],
                                         log=log, captured=captured))
    result = await driver.run(batch_job(), {"input": "hi"})

    assert result.text == "hello"
    assert captured["content_skips"] == [None, "1"]                 # the second page skipped one row
    assert log.count(("GET", "/v1/files/file_out/content")) == 2
    assert ("GET", "/v1/files/file_err/content") not in log


async def test_openai_batch_stops_paging_when_a_page_carries_no_rows():
    # The no-progress guard: `X-Incomplete` forever with nothing behind it must
    # not spin. The empty page ends it, and the error file has the verdict.
    log, captured = [], {}
    other = {"custom_id": "cd" * 32, "response": {"status_code": 200, "body": {}}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_pages=[[other], [], []],
                                         log=log, captured=captured))
    with pytest.raises(BackendError, match="no result"):
        await driver.run(batch_job(), {"input": "hi"})
    assert log.count(("GET", "/v1/files/file_out/content")) == 2


async def test_openai_batch_deletes_the_input_file_once_the_result_is_collected():
    log, captured = [], {}
    ok_line = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 200, "body": CHAT_BODY_DONE}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_lines=[ok_line],
                                         log=log, captured=captured))
    result = await driver.run(batch_job(), {"input": "hi"})
    assert result.text == "hello"
    assert log[-1] == ("DELETE", "/v1/files/file_in")


async def test_openai_batch_deletes_the_input_file_after_a_failed_line_too():
    log, captured = [], {}
    err_line = {"custom_id": BATCH_CUSTOM_ID, "response": None, "error": {"message": "boom"}}
    driver = _batch_driver(batch_backend(statuses=["completed"], error_lines=[err_line],
                                         log=log, captured=captured))
    with pytest.raises(BackendError, match="the batch line failed"):
        await driver.run(batch_job(), {"input": "hi"})
    assert log[-1] == ("DELETE", "/v1/files/file_in")


async def test_openai_batch_a_refused_delete_does_not_change_the_result():
    log, captured = [], {}
    ok_line = {"custom_id": BATCH_CUSTOM_ID, "response": {"status_code": 200, "body": CHAT_BODY_DONE}}
    driver = _batch_driver(batch_backend(statuses=["completed"], output_lines=[ok_line],
                                         delete_status=500, log=log, captured=captured))
    result = await driver.run(batch_job(), {"input": "hi"})
    assert result.text == "hello"                                  # best-effort: the job is unaffected
    assert log[-1] == ("DELETE", "/v1/files/file_in")


async def test_openai_batch_an_unreadable_upload_answer_is_retryable_and_creates_nothing():
    # A gateway that answers 200 with an HTML error page: the upload cannot be
    # read, so nothing names a file id and no batch may be created for it.
    log = []

    def handler(req):
        log.append((req.method, req.url.path))
        return httpx.Response(200, text="<html>gateway</html>")
    with pytest.raises(BackendError, match="unreadable body") as exc:
        await _batch_driver(handler).run(batch_job(), {"input": "hi"})
    assert exc.value.retryable
    assert "<html>" not in str(exc.value)
    assert log == [("POST", "/v1/files")]


# -- plan_media_units: re-deriving the request, and checking its reference ------
#
# The client declares what its reference is and the input leg of the order is
# priced on that declaration. Nothing on the wire holds it to the truth, so this
# is where the truth is established — after decryption, from the reference's own
# header. Unlike `units_out` there is nothing to clamp: input cannot be delivered
# short, so an under-declared reference is refused rather than trimmed.

from tests.test_media_decode import jpeg, mp4, png            # noqa: E402
from vorqd.errors import MediaInputRefused                     # noqa: E402


def _ref(raw: bytes, media_type: str, width: int, height: int, **extra):
    return {"b64": base64.b64encode(raw).decode(), "media_type": media_type,
            "width": width, "height": height, **extra}


def _vid_job_in(units_in, units_out=10_000_000):
    return EvmJob(job_id="job_v", model="wan-ai/wan-2-6:fp8", modality="video", state="Claimed",
                  sla="24h", created_at=0, rate_in=10, rate_out=20_000,
                  units_in=units_in, units_out=units_out)


def test_a_tiered_request_is_rewritten_into_the_pixels_a_template_can_read():
    """An operator writes `width: "{input.width}"` against a request that names
    only a tier. The daemon puts the derived numbers into the request it runs —
    never into the sealed payload, which stays the caller's bytes — so the
    template resolves and both spellings work through one backend config.
    """
    out, units = plan_media_units(
        _vid_job_in(0), {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9", "duration": 5})
    assert (out["width"], out["height"], out["duration_secs"]) == (1280, 720, 5)
    assert units == 1280 * 720 * 5
    assert out["resolution"] == "720p"      # the caller's own spelling survives too


def test_the_clients_arithmetic_is_re_derived_and_not_trusted():
    """A client that computed its own pixels wrongly still gets a correctly priced
    job: the table is the authority, and the order's units_out only ever caps it.
    """
    out, units = plan_media_units(
        _vid_job_in(0), {"prompt": "x", "resolution": "1080p", "aspect_ratio": "16:9",
                         "duration": 2, "num_images": 9})
    assert units == 1920 * 1080 * 2


def test_a_reference_matching_its_declaration_runs():
    raw = png(1280, 720)
    out, _ = plan_media_units(
        _vid_job_in(1280 * 720),
        {"prompt": "x", "image": _ref(raw, "image/png", 1280, 720),
         "resolution": "720p", "duration": 5})
    assert out["image"]["width"] == 1280


def test_a_reference_smaller_than_declared_runs_because_the_client_overpaid():
    """Declared 1920x1080, delivered 1280x720: the client bought more input than
    it sent. There is no refund path for the input leg and nothing to correct, so
    the job runs.
    """
    raw = png(1280, 720)
    plan_media_units(
        _vid_job_in(1920 * 1080),
        {"prompt": "x", "image": _ref(raw, "image/png", 1920, 1080),
         "resolution": "720p", "duration": 5})


def test_a_reference_larger_than_declared_is_refused():
    """The attack this exists for: declare a thumbnail, send a photograph. The
    input leg is already priced at the declaration and cannot be re-billed, so the
    only move is to hand the job back.
    """
    raw = png(1920, 1080)
    with pytest.raises(MediaInputRefused, match="1920x1080"):
        plan_media_units(
            _vid_job_in(64 * 64),
            {"prompt": "x", "image": _ref(raw, "image/png", 64, 64),
             "resolution": "720p", "duration": 5})


def test_a_reference_against_a_zero_input_declaration_is_refused():
    raw = png(64, 64)
    with pytest.raises(MediaInputRefused):
        plan_media_units(
            _vid_job_in(0),
            {"prompt": "x", "image": _ref(raw, "image/png", 64, 64),
             "resolution": "720p", "duration": 5})


def test_a_clip_longer_than_declared_is_refused():
    """Dimensions agreeing is not enough — a reference's length is bought too."""
    raw = mp4(640, 480, seconds=9)
    with pytest.raises(MediaInputRefused):
        plan_media_units(
            _vid_job_in(640 * 480 * 2),
            {"prompt": "x", "video": _ref(raw, "video/mp4", 640, 480, duration_secs=2),
             "resolution": "480p", "duration": 5})


def test_a_reference_that_cannot_be_read_is_refused():
    with pytest.raises(MediaInputRefused, match="could not be read|not a PNG"):
        plan_media_units(
            _vid_job_in(64 * 64),
            {"prompt": "x", "image": _ref(b"not an image", "image/png", 64, 64),
             "resolution": "720p", "duration": 5})


def test_a_media_type_the_operator_does_not_accept_is_refused():
    """A capability gap, not a lie: the operator's backend takes PNG and this is a
    JPEG. Refusing is what `fail` is for — the client is refunded and re-posts.
    """
    raw = jpeg(64, 64)
    with pytest.raises(MediaInputRefused, match="image/jpeg"):
        plan_media_units(
            _vid_job_in(64 * 64),
            {"prompt": "x", "image": _ref(raw, "image/jpeg", 64, 64),
             "resolution": "720p", "duration": 5},
            accept=["image/png"])


def test_an_operator_who_names_no_accept_list_takes_what_the_daemon_can_read():
    raw = jpeg(64, 64)
    plan_media_units(
        _vid_job_in(64 * 64),
        {"prompt": "x", "image": _ref(raw, "image/jpeg", 64, 64),
         "resolution": "720p", "duration": 5})


def test_a_reference_past_the_shared_cap_is_refused_even_if_declared_honestly():
    """The caps are the network's, not the client's to opt out of by being honest
    about exceeding them."""
    raw = png(4000, 4000)
    with pytest.raises(MediaInputRefused, match="reference_pixels|too large"):
        plan_media_units(
            _vid_job_in(4000 * 4000),
            {"prompt": "x", "image": _ref(raw, "image/png", 4000, 4000),
             "resolution": "720p", "duration": 5})


def test_the_output_clamp_still_binds_on_a_tiered_request():
    """`units_out` is a cap for every spelling. Paid for two seconds of 720p,
    asked for five: clamp to two and settle two.
    """
    out, units = plan_media_units(
        _vid_job_in(0, units_out=1280 * 720 * 2),
        {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9", "duration": 5})
    assert out["duration_secs"] == 2
    assert units == 1280 * 720 * 2


def test_a_tier_whose_single_second_exceeds_the_cap_fails():
    with pytest.raises(BackendError):
        plan_media_units(
            _vid_job_in(0, units_out=1000),
            {"prompt": "x", "resolution": "1080p", "aspect_ratio": "16:9", "duration": 5})


def test_a_refusal_never_carries_the_references_bytes():
    """`_report_fail` puts this message on the network. It may carry dimensions,
    counts and media types; it must never carry the reference itself.
    """
    raw = png(1920, 1080)
    encoded = base64.b64encode(raw).decode()
    with pytest.raises(MediaInputRefused) as caught:
        plan_media_units(
            _vid_job_in(64 * 64),
            {"prompt": "x", "image": _ref(raw, "image/png", 64, 64),
             "resolution": "720p", "duration": 5})
    assert encoded not in str(caught.value)
    assert encoded[:32] not in str(caught.value)


# --- what review found --------------------------------------------------------


def test_a_reference_is_held_to_its_area_not_to_its_orientation():
    """A phone photograph is 640x480 in its header and 480x640 to everything that
    applies its orientation tag, the browser that measured it included. Same
    pixels, same bill — and an honest client must not be failed after the claim
    for which way up its camera was.
    """
    plan_media_units(
        _vid_job_in(480 * 640),
        {"prompt": "x", "image": _ref(jpeg(640, 480), "image/jpeg", 480, 640),
         "resolution": "720p", "duration": 5})


def test_a_reference_with_more_area_than_declared_is_still_refused():
    with pytest.raises(MediaInputRefused, match="640x480"):
        plan_media_units(
            _vid_job_in(480 * 480),
            {"prompt": "x", "image": _ref(jpeg(640, 480), "image/jpeg", 480, 480),
             "resolution": "720p", "duration": 5})


@pytest.mark.parametrize("field,value", [
    ("aspect_ratio", []),
    ("aspect_ratio", {"w": 16}),
])
def test_an_aspect_ratio_that_is_not_a_string_falls_back_rather_than_crashing(field, value):
    """Both SDKs refuse this before signing, so it arrives only from a hand-built
    order — and an exception that is not a BackendError leaves a claimed job with
    no `fail` on record, which is the provider's penalty for the client's typo.
    """
    out, _ = plan_media_units(
        _vid_job_in(0), {"prompt": "x", "resolution": "720p", field: value, "duration": 5})
    assert (out["width"], out["height"]) == (1280, 720)


def test_a_backend_that_renders_only_some_lengths_gets_one_of_them():
    """The prevailing interface takes its duration from a short list. Seven seconds
    paid for against a backend that renders five or ten is five: the greatest
    length the backend has that the order covers.
    """
    out, units = plan_media_units(
        _vid_job_in(0), {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9",
                         "duration": 7}, durations=[5, 10])
    assert out["duration_secs"] == 5
    assert units == 1280 * 720 * 5


def test_the_cap_is_applied_before_the_backends_lengths():
    """Asked for ten, paid for six: the cap makes it six and the list makes it five."""
    out, units = plan_media_units(
        _vid_job_in(0, units_out=1280 * 720 * 6),
        {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9", "duration": 10},
        durations=[5, 10])
    assert out["duration_secs"] == 5
    assert units == 1280 * 720 * 5


def test_a_length_shorter_than_anything_the_backend_renders_is_refused():
    with pytest.raises(BackendError, match="5"):
        plan_media_units(
            _vid_job_in(0), {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9",
                             "duration": 3}, durations=[5, 10])


@pytest.mark.parametrize("written,expected", [(9, 2), ("9", "2")])
def test_a_clamp_reaches_the_callers_own_spelling_of_the_length(written, expected):
    """An operator may template `{input.duration}` as readily as
    `{input.duration_secs}`. A clamp that reached only one of them would render
    the seconds nobody paid for through the other — in the type the caller wrote,
    because the interface takes this field as a string as often as a number.
    """
    out, _ = plan_media_units(
        _vid_job_in(0, units_out=1280 * 720 * 2),
        {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9", "duration": written})
    assert out["duration_secs"] == 2
    assert out["duration"] == expected


# --- prepare: requests a submit depends on --------------------------------------
#
# A backend that takes its reference as a URL, and offers an upload endpoint to
# mint one, needs a request made *before* the submit whose answer the submit can
# name. The reference still never leaves the party that was always going to read
# it: the upload goes to the same backend the submit does.


def _prepare_config():
    return BackendConfig(
        preset=None,
        request={
            "prepare": [
                {"name": "first", "when": "input.image",
                 "method": "POST", "url": "http://up/upload",
                 "body": {"data": "data:{input.image.media_type};base64,{input.image.b64}"},
                 "extract": {"url": "$.data.downloadUrl"}},
                {"name": "last", "when": "input.end_image",
                 "method": "POST", "url": "http://up/upload",
                 "body": {"data": "data:{input.end_image.media_type};base64,{input.end_image.b64}"},
                 "extract": {"url": "$.data.downloadUrl"}},
            ],
            "method": "POST", "url": "http://q/submit",
            "body": {"prompt": "{input.prompt}",
                     "first_frame_url": "{prepare.first.url?}",
                     "last_frame_url": "{prepare.last.url?}"},
        },
        response={"mode": "sync", "result": {"media_urls": "$.urls[*]"}},
        retries=0,
    )


def _prepare_backend(seen, *, upload=None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/upload":
            return httpx.Response(200, json=upload if upload is not None
                                  else {"data": {"downloadUrl": f"http://files/{len(seen)}.png"}})
        return httpx.Response(200, json={"urls": ["http://cdn/out.mp4"]})
    return handler


async def test_a_prepare_step_runs_first_and_the_submit_names_its_answer():
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(_prepare_backend(seen)))
    driver = BackendDriver.from_config(client, _prepare_config(), sleep=no_sleep)
    await driver.run(media_job(), {"prompt": "a cat",
                                   "image": {"b64": "QUJD", "media_type": "image/png"}})
    assert [path for path, _ in seen] == ["/upload", "/submit"]
    assert seen[0][1] == {"data": "data:image/png;base64,QUJD"}
    # The step that did not apply left no field behind — not an empty one.
    assert seen[1][1] == {"prompt": "a cat", "first_frame_url": "http://files/1.png"}


async def test_a_prepare_step_whose_input_is_absent_is_skipped():
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(_prepare_backend(seen)))
    driver = BackendDriver.from_config(client, _prepare_config(), sleep=no_sleep)
    await driver.run(media_job(), {"prompt": "a cat"})
    assert seen == [("/submit", {"prompt": "a cat"})]


async def test_a_prepare_step_that_answers_without_what_it_was_run_for_fails_the_job():
    """Otherwise the optional token downstream drops the field and the job runs as
    text-to-video — a different render from the one the client paid for."""
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        _prepare_backend(seen, upload={"success": False, "msg": "quota"})))
    driver = BackendDriver.from_config(client, _prepare_config(), sleep=no_sleep)
    with pytest.raises(BackendError, match="first"):
        await driver.run(media_job(), {"prompt": "a cat",
                                       "image": {"b64": "QUJD", "media_type": "image/png"}})
    assert [path for path, _ in seen] == ["/upload"]      # the submit never went out


async def test_a_request_field_the_job_does_not_carry_fails_the_job_not_the_daemon():
    """An unresolved required token used to leave as a KeyError, which the
    scheduler's crash handler meets with no `fail` on record — the claim then sits
    until the SLA reclaims it, at the provider's expense, over a client's omission.
    """
    config = raw_media_config()
    config.request["body"]["resolution"] = "{input.resolution}"
    client = httpx.AsyncClient(transport=httpx.MockTransport(queue_backend(statuses=[])))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    with pytest.raises(BackendError, match="input.resolution") as caught:
        await driver.run(media_job(), {"prompt": "a secret prompt"})
    assert not caught.value.retryable
    assert "secret" not in str(caught.value)


def test_the_ratio_that_was_priced_is_the_ratio_the_backend_is_asked_for():
    """`auto` is this network's word, resolved from the reference, and the frame it
    resolves to is what the order is billed on. A backend handed `auto` — or
    nothing — renders a shape of its own choosing, at somebody's expense.
    """
    portrait = _ref(png(480, 640), "image/png", 480, 640)
    out, units = plan_media_units(
        _vid_job_in(480 * 640),
        {"prompt": "x", "image": portrait, "resolution": "480p", "aspect_ratio": "auto",
         "duration": 4})
    assert out["aspect_ratio"] == "3:4"
    assert units == 552 * 736 * 4

    out, _ = plan_media_units(_vid_job_in(0), {"prompt": "x", "resolution": "480p", "duration": 4})
    assert out["aspect_ratio"] == "16:9"            # the fallback, said out loud


def test_a_request_in_raw_pixels_is_not_given_a_ratio_it_never_named():
    out, _ = plan_media_units(_vid_job_in(0), {"prompt": "x", "width": 1280, "height": 720,
                                               "duration_secs": 4})
    assert "aspect_ratio" not in out


# --- listed references, sound, and what one backend can take ---------------------


def _sound(n_bytes=64, media_type="audio/mpeg"):
    return {"b64": base64.b64encode(b"\x00" * n_bytes).decode(), "media_type": media_type}


def _tiered(**extra):
    return {"prompt": "x", "resolution": "720p", "aspect_ratio": "16:9", "duration": 5, **extra}


def test_listed_references_are_decoded_and_summed_like_singular_ones():
    refs = [_ref(png(640, 480), "image/png", 640, 480), _ref(jpeg(320, 240), "image/jpeg", 320, 240)]
    clip = _ref(mp4(640, 480, seconds=3), "video/mp4", 640, 480, duration_secs=3)
    paid = 640 * 480 + 320 * 240 + 640 * 480 * 3
    plan_media_units(_vid_job_in(paid), _tiered(reference_images=refs, reference_videos=[clip]))
    with pytest.raises(MediaInputRefused, match="pixel-seconds"):
        plan_media_units(_vid_job_in(paid - 1), _tiered(reference_images=refs, reference_videos=[clip]))


def test_a_listed_reference_that_outruns_its_declaration_is_named_by_its_place():
    refs = [_ref(png(64, 64), "image/png", 64, 64), _ref(png(1920, 1080), "image/png", 64, 64)]
    with pytest.raises(MediaInputRefused, match=r"reference_images\[1\]"):
        plan_media_units(_vid_job_in(10_000_000), _tiered(reference_images=refs))


def test_a_list_longer_than_the_network_allows_is_refused():
    refs = [_ref(png(64, 64), "image/png", 64, 64)] * 10
    with pytest.raises(MediaInputRefused, match="reference_images"):
        plan_media_units(_vid_job_in(10_000_000), _tiered(reference_images=refs))


def test_reference_sound_runs_without_being_measured_because_nothing_bills_it():
    plan_media_units(_vid_job_in(0), _tiered(reference_audios=[_sound()]))


def test_reference_sound_is_still_held_to_a_type_and_a_size():
    with pytest.raises(MediaInputRefused, match="audio/ogg"):
        plan_media_units(_vid_job_in(0), _tiered(reference_audios=[_sound(media_type="audio/ogg")]))
    with pytest.raises(MediaInputRefused, match="reference_audio_bytes"):
        plan_media_units(_vid_job_in(0), _tiered(reference_audios=[_sound(15 * 1024 * 1024 + 1)]))


def test_a_picture_passed_off_as_sound_does_not_dodge_the_meter():
    """Sound counts zero, so the key decides the kind — and the type must agree
    with the key, or a 4K still listed under `reference_audios` would ride free."""
    smuggled = {"b64": base64.b64encode(png(3840, 2160)).decode(), "media_type": "image/png"}
    with pytest.raises(MediaInputRefused, match="reference_audios"):
        plan_media_units(_vid_job_in(0), _tiered(reference_audios=[smuggled]))


@pytest.mark.parametrize("bounds,why", [
    ({"still": {"min_side": 300}}, "min_side"),
    ({"still": {"max_side": 200}}, "max_side"),
    ({"still": {"max_ratio": 1.2}}, "max_ratio"),
    ({"still": {"min_ratio": 1.5}}, "min_ratio"),
    ({"still": {"min_pixels": 100_000}}, "min_pixels"),
    ({"still": {"max_pixels": 1_000}}, "max_pixels"),
    ({"still": {"max_bytes": 10}}, "max_bytes"),
    ({"still": {"max_count": 0}}, "max_count"),
])
def test_a_reference_outside_what_this_backend_takes_is_handed_back_before_any_upload(bounds, why):
    """The network's caps are wide; a backend's are its own. A 256x192 frame is a
    fine reference to the network and a 422 to a backend that wants 300 a side —
    found here it costs nothing, found by the backend it costs an upload and a
    claim."""
    with pytest.raises(MediaInputRefused, match=why):
        plan_media_units(_vid_job_in(256 * 192),
                         _tiered(image=_ref(png(256, 192), "image/png", 256, 192)), bounds=bounds)


def test_clips_are_bounded_one_by_one_and_together():
    clip = _ref(mp4(640, 480, seconds=8), "video/mp4", 640, 480, duration_secs=8)
    paid = 640 * 480 * 8 * 2
    plan_media_units(_vid_job_in(paid), _tiered(reference_videos=[clip, clip]),
                     bounds={"clip": {"max_secs": 8, "max_total_secs": 16}})
    with pytest.raises(MediaInputRefused, match="max_secs"):
        plan_media_units(_vid_job_in(paid), _tiered(reference_videos=[clip]),
                         bounds={"clip": {"max_secs": 7}})
    with pytest.raises(MediaInputRefused, match="max_total_secs"):
        plan_media_units(_vid_job_in(paid), _tiered(reference_videos=[clip, clip]),
                         bounds={"clip": {"max_total_secs": 15}})


# --- what one backend renders, and what it lets the model decide -----------------


def test_a_tier_this_backend_does_not_render_is_handed_back():
    with pytest.raises(BackendError, match="480p, 720p"):
        plan_media_units(_vid_job_in(0), _tiered(resolution="1080p"), resolutions=["480p", "720p"])


def test_a_backend_that_renders_tiers_is_not_handed_raw_pixels():
    """It would render its own default while the job was billed at the caller's size."""
    with pytest.raises(BackendError, match="resolution"):
        plan_media_units(_vid_job_in(0), {"prompt": "x", "width": 1280, "height": 720,
                                          "duration_secs": 5}, resolutions=["480p", "720p"])


def test_a_length_left_to_the_model_is_sent_in_the_backends_own_spelling():
    cap = 854 * 480 * 15
    out, units = plan_media_units(
        _vid_job_in(0, units_out=cap),
        {"prompt": "x", "resolution": "480p", "aspect_ratio": "16:9", "duration": "auto"},
        durations=[4, 8, 15], auto_duration=-1)
    assert out["duration_secs"] == -1 and out["duration"] == -1
    assert units == cap                       # a ceiling; the delivered clip settles


def test_a_length_left_to_the_model_needs_a_cap_that_covers_its_longest_choice():
    """There is no clamping a model that has not chosen yet. If it may pick fifteen
    seconds the order has to cover fifteen, or the difference is this provider's."""
    with pytest.raises(BackendError, match="15"):
        plan_media_units(
            _vid_job_in(0, units_out=854 * 480 * 14),
            {"prompt": "x", "resolution": "480p", "aspect_ratio": "16:9", "duration": "auto"},
            durations=[4, 8, 15], auto_duration=-1)


def test_a_backend_that_cannot_leave_the_length_to_the_model_says_so():
    with pytest.raises(BackendError, match="auto"):
        plan_media_units(
            _vid_job_in(0), {"prompt": "x", "resolution": "480p", "duration": "auto"})


def test_a_shape_left_to_the_model_is_priced_at_the_tiers_largest_frame():
    ref = _ref(png(1000, 800), "image/png", 1000, 800)
    out, units = plan_media_units(
        _vid_job_in(800_000), _tiered(image=ref, aspect_ratio="adaptive"), adaptive_aspect="keep")
    assert out["aspect_ratio"] == "keep"
    assert units == 1472 * 632 * 5
    with pytest.raises(BackendError, match="adaptive"):
        plan_media_units(_vid_job_in(800_000), _tiered(image=ref, aspect_ratio="adaptive"))


# --- a backend that says how it went in the body, not in the status line ---------


def _in_body_config(**poll_extra):
    return BackendConfig(
        preset=None,
        request={"method": "POST", "url": "http://q/submit", "body": {"prompt": "{input.prompt}"}},
        response={
            "mode": "poll",
            "ok": {"field": "$.code", "values": [200], "retry": [429, 455], "message": "$.msg"},
            "poll": {"handle": "$.data.id",
                     "request": {"method": "GET", "url": "http://q/task?id={handle}"},
                     "status_field": "$.data.state", "done_values": ["success"],
                     "failed_values": ["fail"], "interval_s": 0, **poll_extra},
            "result": {"media_urls": "$.data.urls[*]"},
        },
        retries=0,
    )


def _answers(*bodies):
    queue = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=queue.pop(0))
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_a_refusal_carried_in_the_body_is_read_as_one():
    """HTTP 200 with `code: 402` is not a submit that lost its handle. It is the
    backend saying no, and the reason on record should be the one it gave."""
    driver = BackendDriver.from_config(
        _answers({"code": 402, "msg": "insufficient credits", "data": None}),
        _in_body_config(), sleep=no_sleep)
    with pytest.raises(BackendError, match="402") as caught:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert not caught.value.retryable
    assert "insufficient" not in str(caught.value)    # the message is logged, never sent


async def test_a_busy_answer_carried_in_the_body_is_retryable():
    driver = BackendDriver.from_config(
        _answers({"code": 429, "msg": "slow down", "data": None}), _in_body_config(), sleep=no_sleep)
    with pytest.raises(BackendError) as caught:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert caught.value.retryable


async def test_a_busy_poll_tick_is_polled_through_not_read_as_a_status():
    driver = BackendDriver.from_config(
        _answers({"code": 200, "data": {"id": "t1"}},
                 {"code": 429, "msg": "slow down", "data": None},
                 {"code": 200, "data": {"state": "success", "urls": ["http://cdn/a.mp4"]}}),
        _in_body_config(), sleep=no_sleep)
    result = await driver.run(media_job(), {"prompt": "a cat"})
    assert result.media_urls == ["http://cdn/a.mp4"]


async def test_an_answer_without_the_field_is_not_a_refusal():
    driver = BackendDriver.from_config(
        _answers({"data": {"id": "t1"}}, {"data": {"state": "success", "urls": ["http://cdn/a.mp4"]}}),
        _in_body_config(), sleep=no_sleep)
    assert (await driver.run(media_job(), {"prompt": "a cat"})).media_urls == ["http://cdn/a.mp4"]


async def test_a_failed_task_is_reported_with_the_backends_own_code():
    """'fail' alone reads the same for a content refusal and an outage. The code
    travels in the reason; the message, which may quote the prompt, only to the log."""
    driver = BackendDriver.from_config(
        _answers({"code": 200, "data": {"id": "t1"}},
                 {"code": 200, "data": {"state": "fail", "failCode": "E_POLICY",
                                        "failMsg": "prompt mentions a cat"}}),
        _in_body_config(failure_code="$.data.failCode", failure_message="$.data.failMsg"),
        sleep=no_sleep)
    with pytest.raises(BackendError, match="E_POLICY") as caught:
        await driver.run(media_job(), {"prompt": "a cat"})
    assert "mentions" not in str(caught.value)


# --- a prepare step per listed reference -----------------------------------------


async def test_a_prepare_step_may_run_once_per_listed_reference():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path == "/upload":
            return httpx.Response(200, json={"url": f"http://files/{len(seen)}"})
        return httpx.Response(200, json={"urls": ["http://cdn/out.mp4"]})

    config = BackendConfig(
        preset=None,
        request={
            "prepare": [{"name": "refs", "for_each": "input.reference_images",
                         "method": "POST", "url": "http://up/upload",
                         "body": {"data": "{item.b64}"}, "extract": {"url": "$.url"}}],
            "method": "POST", "url": "http://q/submit",
            "body": {"prompt": "{input.prompt}", "reference_urls": "{prepare.refs[*].url?}"},
        },
        response={"mode": "sync", "result": {"media_urls": "$.urls[*]"}},
        retries=0,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = BackendDriver.from_config(client, config, sleep=no_sleep)
    await driver.run(media_job(), {"prompt": "x", "reference_images": [{"b64": "QQ=="}, {"b64": "Qg=="}]})
    assert [b for _, b in seen] == [{"data": "QQ=="}, {"data": "Qg=="},
                                    {"prompt": "x", "reference_urls": ["http://files/1", "http://files/2"]}]

    seen.clear()
    await driver.run(media_job(), {"prompt": "x"})
    assert seen == [("/submit", {"prompt": "x"})]


async def test_preset_headers_render_job_fields_over_the_presets_own():
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["headers"] = req.headers
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    be = BackendConfig(preset="openai-chat", params={
        "base_url": "http://runtime/v1", "model": "m", "api_key": "sk-x",
        "headers": {"X-Request-Id": "{job.id}", "X-Request-Client": "{job.owner}"},
    })
    driver = BackendDriver.from_config(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), be, sleep=no_sleep)
    job = text_job()
    job.owner = "0xabc0000000000000000000000000000000000001"

    await driver.run(job, {"messages": [{"role": "user", "content": "hi"}]})

    assert captured["headers"]["x-request-id"] == "job_1"
    assert captured["headers"]["x-request-client"] == job.owner
    assert captured["headers"]["authorization"] == "Bearer sk-x"
