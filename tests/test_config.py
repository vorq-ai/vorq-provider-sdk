"""Config loader + §6 validation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vorqd.config import (MAX_INPUT_BYTES_PER_UNIT, PricingConfig, load_config,
                          resolve_env)
from vorqd.errors import ConfigError

EXAMPLES = Path(__file__).parents[1] / "docs" / "examples"


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "vorqd.yaml"
    p.write_text(text)
    return p


# --- example configs load into typed objects --------------------------------


def test_loads_preset_example(monkeypatch):
    monkeypatch.setenv("VORQ_BOX_KEY", "bb" * 32)
    cfg = load_config(EXAMPLES / "vllm.yaml")
    # The daemon's id is admin-issued and discovered at handshake, never configured:
    # a `provider.id` key in the YAML (the example still carries one) is ignored.
    assert not hasattr(cfg.provider, "id")
    assert cfg.provider.wallet_key is None  # omitted -> generated under the hood
    assert cfg.provider.box_key == "bb" * 32  # required, via env: indirection
    assert cfg.provider.capacity == 4
    # optional provider keys default
    assert cfg.provider.metrics_port == 9090
    assert cfg.provider.log_level == "info"
    assert cfg.provider.poll_interval_s == 5
    assert cfg.provider.safety_margin_s == 60

    model = cfg.models[0]
    assert model.model == "deepseek-ai/deepseek-v4-pro:fp8"
    # Rates are USD per 1M units of work, carried verbatim as the quoted decimal
    # strings they are written as: the token's decimals scale them later.
    assert model.slas["24h"].rate_in == "0.16"
    assert model.slas["24h"].rate_out == "0.55"
    assert model.slas["1h"].rate_out == "0.75"
    assert model.modality == "text"   # the catalog names none, so the operator must
    assert model.backend.preset == "openai-chat"
    assert model.backend.params["base_url"] == "http://localhost:8000/v1"
    assert model.backend.params_supported == [
        "temperature", "top_p", "max_tokens", "seed", "stop", "reasoning_effort",
    ]
    assert model.backend.health == {"path": "/health"}


def test_loads_raw_poll_example_with_embedded_env(monkeypatch):
    monkeypatch.setenv("QUEUE_BACKEND_KEY", "sekret")
    monkeypatch.setenv("VORQ_BOX_KEY", "bb" * 32)
    cfg = load_config(EXAMPLES / "queue-backend.yaml")
    model = cfg.models[0]
    assert model.model == "black-forest-labs/flux-2-dev:fp8"
    assert model.slas["24h"].rate_in is None  # media: rate_out only
    assert model.slas["24h"].rate_out == "0.012"
    assert model.modality == "image"
    be = model.backend
    assert be.preset is None
    assert be.request["method"] == "POST"
    # env: indirection resolves even inside an embedded string
    assert be.request["headers"]["Authorization"] == "Key sekret"
    assert be.response["mode"] == "poll"
    assert be.response["poll"]["status_url"] == "$.status_url"
    assert be.response["result"]["media_urls"] == "$.images[*].url"
    assert be.retries == 2


# --- the examples' commented option blocks ----------------------------------

#: The optional blocks both examples ship commented out. They are documentation
#: no ordinary load can reach, so the tests below uncomment them and load the
#: result: an operator who follows the comment must get a config that parses,
#: not one that fails the moment they try the feature.
_OPTIONAL_BLOCKS = ("pricing:", "load:", "bid_filter:", "retries:")


def uncomment_optional_blocks(text: str) -> str:
    """Strip the leading ``# `` from the examples' commented option blocks.

    A block runs from its commented key to the first line that is not a comment,
    which is exactly how the examples are written — no line numbers to keep in
    step with the files.
    """
    out, inside = [], False
    for line in text.splitlines():
        m = re.match(r"^(\s*)# ?(.*)$", line)
        body = m.group(2) if m else None
        if m and any(body.startswith(k) for k in _OPTIONAL_BLOCKS):
            inside = True
        elif inside and (m is None or not body.strip()):
            inside = False
        out.append(f"{m.group(1)}{body}" if inside and m else line)
    return "\n".join(out) + "\n"


def test_the_preset_example_ships_todays_behaviour(monkeypatch):
    monkeypatch.setenv("VORQ_BOX_KEY", "bb" * 32)
    cfg = load_config(EXAMPLES / "vllm.yaml")
    # Copying an example must not opt an operator into paying themselves less.
    assert cfg.provider.pricing == PricingConfig()
    assert cfg.provider.bid_filter == {}
    assert cfg.models[0].load is None


def test_the_preset_examples_commented_blocks_are_the_config_they_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("VORQ_BOX_KEY", "bb" * 32)
    src = (EXAMPLES / "vllm.yaml").read_text()
    cfg = load_config(write(tmp_path, uncomment_optional_blocks(src)))
    assert cfg.provider.pricing == PricingConfig(
        max_discount_pct=20, bid_tolerance_pct=10, low_load_pct=30
    )
    # No `bid_filter` block in the example any more: the coordinator's matcher
    # decides which bids a daemon is offered, and the knob is inert.
    assert cfg.provider.bid_filter == {}
    load = cfg.models[0].load
    assert (load.url, load.metric, load.scale) == (
        "http://localhost:8000/metrics", "vllm:kv_cache_usage_perc", 1.0
    )
    be = cfg.models[0].backend
    assert (be.retries, be.retry_backoff_s, be.timeout_s, be.concurrency) == (8, 30.0, 300.0, 4)
    assert be.rate_limit == {"1m": 40, "1h": 1000, "24h": 5000}
    # limit keys are read by the loader, never forwarded as request params
    assert not set(be.params) & {"retries", "timeout_s", "concurrency", "rate_limit"}


def test_the_raw_examples_commented_probe_is_the_probe_it_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("VORQ_BOX_KEY", "bb" * 32)
    monkeypatch.setenv("QUEUE_BACKEND_KEY", "k")
    src = (EXAMPLES / "queue-backend.yaml").read_text()
    cfg = load_config(write(tmp_path, uncomment_optional_blocks(src)))
    load = cfg.models[0].load
    # A percentage series, hence the divisor — the case the comment exists to show.
    assert (load.url, load.metric, load.scale) == (
        "http://localhost:9400/metrics", "DCGM_FI_DEV_GPU_UTIL", 100.0
    )
    # The probe alone changes no price: that still takes a `pricing:` block.
    assert cfg.provider.pricing == PricingConfig()


# --- env indirection --------------------------------------------------------


def test_missing_env_var_fails_fast_naming_it(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    cfg = write(tmp_path, MINIMAL.replace("dev-stub", "env:NOPE_KEY"))
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    assert "NOPE_KEY" in str(exc.value)


def test_resolve_env_embedded(monkeypatch):
    monkeypatch.setenv("K", "xyz")
    assert resolve_env("Key env:K") == "Key xyz"
    assert resolve_env({"a": ["env:K", "lit"]}) == {"a": ["xyz", "lit"]}


# --- §6 validation ----------------------------------------------------------

MINIMAL = """
provider:
  id: p
  wallet_key: dev-stub
  box_key: dev-box
  api_url: http://localhost:8402
  capacity: 1
models:
  - model: m:fp8
    slas:
      "1h": { rate_out: "0.1" }
    backend:
      preset: openai-chat
      base_url: http://localhost:8000/v1
      model: runtime-model
"""


def test_minimal_valid(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.models[0].backend.preset == "openai-chat"


def test_provider_id_key_is_silently_ignored(tmp_path):
    # MINIMAL carries `id: p`; loading must not error and must not surface it.
    cfg = load_config(write(tmp_path, MINIMAL))
    assert not hasattr(cfg.provider, "id")


def test_backend_needs_exactly_one_of_preset_or_request(tmp_path):
    both = MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model",
        "    backend:\n      preset: openai-chat\n      request: { method: POST, url: http://x }",
    )
    with pytest.raises(ConfigError, match="preset.*request|exactly one"):
        load_config(write(tmp_path, both))

    neither = MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model",
        "    backend:\n      base_url: http://localhost:8000/v1",
    )
    with pytest.raises(ConfigError, match="preset.*request|exactly one"):
        load_config(write(tmp_path, neither))


def test_result_exactly_one(tmp_path):
    cfg = """
provider: { id: p, wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }
models:
  - model: m:fp8
    slas: { "1h": { rate_out: "0.1" } }
    backend:
      request: { method: POST, url: http://x, body: {} }
      response:
        mode: sync
        result: { text: "$.a", media_urls: "$.b" }
"""
    with pytest.raises(ConfigError, match="text.*media|exactly one"):
        load_config(write(tmp_path, cfg))


def test_raw_backend_carries_param_caps(tmp_path):
    """Media jobs — the caps' main use — are served by raw mappings, so the caps
    must survive the raw branch, not just the preset one."""
    cfg = """
provider: { id: p, wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }
models:
  - model: m:fp8
    slas: { "24h": { rate_out: "0.01" } }
    backend:
      param_caps: { steps: 40, fps: 24 }
      request: { method: POST, url: http://x, body: {} }
      response:
        mode: sync
        result: { media_urls: "$.urls" }
"""
    loaded = load_config(write(tmp_path, cfg))
    assert loaded.models[0].backend.param_caps == {"steps": 40, "fps": 24}

    bad = cfg.replace("param_caps: { steps: 40, fps: 24 }", "param_caps: { steps: forty }")
    with pytest.raises(ConfigError, match="param_caps"):
        load_config(write(tmp_path, bad))


def test_raw_text_result_requires_completion_tokens(tmp_path):
    # A text job is billed by the reported output-token count, so a raw mapping
    # that maps 'text' without 'completion_tokens' has no billable quantity and
    # must fail at load.
    cfg = """
provider: { id: p, wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }
models:
  - model: m:fp8
    slas: { "1h": { rate_out: "0.1" } }
    backend:
      request: { method: POST, url: http://x, body: {} }
      response:
        mode: sync
        result: { text: "$.choices[0].message.content" }
"""
    with pytest.raises(ConfigError, match="completion_tokens"):
        load_config(write(tmp_path, cfg))


def test_raw_text_result_with_completion_tokens_validates(tmp_path):
    cfg = """
provider: { id: p, wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }
models:
  - model: m:fp8
    slas: { "1h": { rate_out: "0.1" } }
    backend:
      request: { method: POST, url: http://x, body: {} }
      response:
        mode: sync
        result: { text: "$.choices[0].message.content", completion_tokens: "$.usage.completion_tokens" }
"""
    cfg = load_config(write(tmp_path, cfg))
    response = cfg.models[0].backend.response
    assert response is not None
    assert response["result"]["completion_tokens"] == "$.usage.completion_tokens"


def test_preset_satisfies_completion_tokens_requirement(tmp_path):
    # The openai-chat preset maps completion_tokens implicitly, so a preset config
    # with no explicit response block still loads.
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.models[0].backend.preset == "openai-chat"


def test_poll_exactly_one(tmp_path):
    cfg = """
provider: { id: p, wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }
models:
  - model: m:fp8
    slas: { "1h": { rate_out: "0.1" } }
    backend:
      request: { method: POST, url: http://x, body: {} }
      response:
        mode: poll
        poll:
          status_url: "$.u"
          request: { method: POST, url: http://x }
          status_field: "$.s"
        result: { media_urls: "$.b" }
"""
    with pytest.raises(ConfigError, match="status_url.*request|exactly one"):
        load_config(write(tmp_path, cfg))


def test_sla_requires_rate_out(tmp_path):
    cfg = MINIMAL.replace('"1h": { rate_out: "0.1" }', '"1h": { rate_in: "0.1" }')
    with pytest.raises(ConfigError, match="rate_out"):
        load_config(write(tmp_path, cfg))


def _with_rate(field: str, value: str) -> str:
    rates = (f'"1h": {{ rate_in: {value}, rate_out: "0.1" }}' if field == "rate_in"
             else f'"1h": {{ rate_out: {value} }}')
    cfg = MINIMAL.replace('"1h": { rate_out: "0.1" }', rates)
    assert rates in cfg     # the fixture edit landed; the refusal is the code's
    return cfg


@pytest.mark.parametrize("value", ["0.16", "160000", "0", "true"])
@pytest.mark.parametrize("field", ["rate_in", "rate_out"])
def test_an_unquoted_rate_is_refused_at_startup(tmp_path, field, value):
    """A rate is a quoted USD string. A YAML float cannot say which decimal was
    meant, and a YAML integer is ambiguous with the old atomic spelling, so both
    are refused at load, naming the model, the window and the side."""
    with pytest.raises(ConfigError, match="quoted USD decimal string") as exc:
        load_config(write(tmp_path, _with_rate(field, value)))
    assert "m:fp8" in str(exc.value)      # which model
    assert "1h" in str(exc.value)         # which window
    assert field in str(exc.value)        # which side


@pytest.mark.parametrize("value", ['"-1"', '"1e3"', '".5"', '"5."', '"01"', '"1,5"', '""',
                                   '"0.' + "1" * 19 + '"'])
@pytest.mark.parametrize("field", ["rate_in", "rate_out"])
def test_a_rate_outside_the_usd_grammar_is_refused_at_startup(tmp_path, field, value):
    with pytest.raises(ConfigError, match=field):
        load_config(write(tmp_path, _with_rate(field, value)))


def test_a_quoted_usd_rate_loads_verbatim(tmp_path):
    """Grammar is settled at load; the token's decimals are not known until the
    chain context is read, so a rate finer than six digits still loads here and
    is refused when converted (see the scheduler's tests)."""
    cfg = _with_rate("rate_in", '"0.16"')
    rate = load_config(write(tmp_path, cfg)).models[0].slas["1h"]
    assert (rate.rate_in, rate.rate_out) == ("0.16", "0.1")
    fine = load_config(write(tmp_path, _with_rate("rate_out", '"0.0000001"')))
    assert fine.models[0].slas["1h"].rate_out == "0.0000001"


def test_provider_required_fields(tmp_path):
    cfg = MINIMAL.replace("  capacity: 1\n", "")
    with pytest.raises(ConfigError, match="capacity"):
        load_config(write(tmp_path, cfg))


def test_capacity_left_unset_is_the_sum_of_what_the_entries_may_hold(tmp_path):
    """An entry's share is its `rate_limit` window at its shortest SLA, else
    its `concurrency`; the daemon requests the sum at boot."""
    cfg = MINIMAL.replace("  capacity: 1\n", "").replace(
        "      model: runtime-model\n",
        "      model: runtime-model\n      concurrency: 2\n      rate_limit: {\"1m\": 40}\n"
        "  - model: day:fp8\n    slas:\n      \"24h\": { rate_out: \"0.1\" }\n"
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n"
        "      model: runtime-day\n      concurrency: 4\n"
        "      rate_limit: {\"1m\": 40, \"24h\": 5000}\n",
    )
    assert load_config(write(tmp_path, cfg)).provider.capacity == 2 + 5000


def test_capacity_cannot_be_derived_past_an_unbounded_entry(tmp_path):
    cfg = MINIMAL.replace("  capacity: 1\n", "").replace(
        "      model: runtime-model\n", "      model: runtime-model\n      rate_limit: {\"1m\": 40}\n",
    )
    with pytest.raises(ConfigError, match="'m:fp8'.*cannot be derived"):
        load_config(write(tmp_path, cfg))


def test_a_day_budget_the_minute_window_cannot_start_is_refused(tmp_path):
    """The window at the SLA is what the entry holds; every shorter window has
    to be able to start that many inside the SLA, or the tail is failed back."""
    cfg = MINIMAL.replace("      model: runtime-model\n",
                          "      model: runtime-model\n      rate_limit: {\"1m\": 1, \"1h\": 61}\n")
    with pytest.raises(ConfigError, match=r"rate_limit\['1h'\] of 61 .* rate_limit\['1m'\] of 1 starts at most 60"):
        load_config(write(tmp_path, cfg))
    fine = cfg.replace('"1h": 61', '"1h": 60')
    assert load_config(write(tmp_path, fine)).models[0].backend.rate_limit == {"1m": 1, "1h": 60}


def test_capacity_from_the_environment_reads_as_a_number(tmp_path, monkeypatch):
    monkeypatch.setenv("VORQ_CAP", "7")
    cfg = MINIMAL.replace("  capacity: 1\n", "  capacity: env:VORQ_CAP\n")
    assert load_config(write(tmp_path, cfg)).provider.capacity == 7


def test_a_configured_capacity_is_kept_as_written(tmp_path):
    cfg = MINIMAL.replace("      model: runtime-model\n",
                          "      model: runtime-model\n      rate_limit: {\"1h\": 500}\n")
    assert load_config(write(tmp_path, cfg)).provider.capacity == 1
    with pytest.raises(ConfigError, match="capacity"):
        load_config(write(tmp_path, MINIMAL.replace("  capacity: 1\n", "  capacity: 0\n")))


def test_box_key_is_required(tmp_path):
    cfg = MINIMAL.replace("  box_key: dev-box\n", "")
    with pytest.raises(ConfigError, match="box_key"):
        load_config(write(tmp_path, cfg))


def test_param_map_parsed_and_validated(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      param_map: {reasoning: reasoning_effort}\n",
    )
    cfg = load_config(write(tmp_path, good))
    assert cfg.models[0].backend.param_map == {"reasoning": "reasoning_effort"}

    bad = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      param_map: [not, a, mapping]\n",
    )
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))


def test_param_map_object_form_parsed(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n"
        "      param_map:\n"
        "        reasoning_effort:\n"
        "          to: chat_template_kwargs\n"
        "          values: {none: {enable_thinking: false}}\n"
        "          unmapped: drop\n",
    )
    cfg = load_config(write(tmp_path, good))
    spec = cfg.models[0].backend.param_map["reasoning_effort"]
    assert spec["to"] == "chat_template_kwargs"
    assert spec["unmapped"] == "drop"

    for bad_fragment in (
        "        reasoning_effort: {unmapped: explode}\n",     # bad unmapped value
        "        reasoning_effort: {sideways: x}\n",           # unknown key
        "        reasoning_effort: 7\n",                       # not str or object
    ):
        bad = MINIMAL.replace(
            "      preset: openai-chat\n",
            "      preset: openai-chat\n      param_map:\n" + bad_fragment,
        )
        with pytest.raises(ConfigError, match="param_map"):
            load_config(write(tmp_path, bad))


def test_reasoning_dialect_resolves_into_param_map(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n"
        "      reasoning: { efforts: [none, low, high], budget: reasoning_budget }\n",
    )
    cfg = load_config(write(tmp_path, good))
    pm = cfg.models[0].backend.param_map
    assert pm["reasoning_effort"]["values"]["medium"] == "low"   # derived collapse
    assert pm["reasoning_max_tokens"] == "reasoning_budget"

    unknown = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      reasoning: effort_quantum\n",
    )
    with pytest.raises(ConfigError, match="effort_quantum"):
        load_config(write(tmp_path, unknown))

    off_ladder = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      reasoning: { efforts: [none, minimal, high] }\n",
    )
    with pytest.raises(ConfigError, match="minimal"):
        load_config(write(tmp_path, off_ladder))


def test_reasoning_default_effort_is_carried_beside_the_map(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n"
        "      reasoning: { efforts: [none, low, high], budget: reasoning_budget, default_effort: medium }\n",
    )
    cfg = load_config(write(tmp_path, good))
    be = cfg.models[0].backend
    assert be.default_effort == "medium"
    assert be.param_map["reasoning_max_tokens"] == "reasoning_budget"   # map resolved without it
    assert "default_effort" not in be.param_map

    bad = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n"
        "      reasoning: { efforts: [none, high], default_effort: mediumish }\n",
    )
    with pytest.raises(ConfigError, match="mediumish"):
        load_config(write(tmp_path, bad))


def test_reasoning_default_effort_on_a_template_preset(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n"
        "      reasoning: { preset: thinking_bool, default_effort: none }\n",
    )
    cfg = load_config(write(tmp_path, good))
    be = cfg.models[0].backend
    assert be.default_effort == "none"
    assert be.param_map["reasoning_effort"]["to"] == "chat_template_kwargs"


def test_params_supported_loads_and_is_validated(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      params_supported: [temperature, top_p, seed]\n",
    )
    cfg = load_config(write(tmp_path, good))
    assert cfg.models[0].backend.params_supported == ["temperature", "top_p", "seed"]

    bad = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      params_supported: {temperature: true}\n",
    )
    with pytest.raises(ConfigError, match="params_supported"):
        load_config(write(tmp_path, bad))


def test_preset_headers_load_and_are_validated(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      headers: {X-Request-Id: \"{job.id}\"}\n",
    )
    cfg = load_config(write(tmp_path, good))
    assert cfg.models[0].backend.params["headers"] == {"X-Request-Id": "{job.id}"}

    bad = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      headers: [X-Request-Id]\n",
    )
    with pytest.raises(ConfigError, match="headers"):
        load_config(write(tmp_path, bad))


def test_reasoning_inline_param_map_overrides(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n"
        "      reasoning: { efforts: [none, high, max] }\n"
        "      param_map: {reasoning_max_tokens: thinking_budget}\n",
    )
    cfg = load_config(write(tmp_path, good))
    pm = cfg.models[0].backend.param_map
    assert pm["reasoning_max_tokens"] == "thinking_budget"       # inline wins
    assert pm["reasoning_effort"]["values"]["max"] == "max"      # rest of the dialect kept


def test_param_caps_parsed_and_validated(tmp_path):
    good = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      param_caps: {steps: 40, fps: 24}\n",
    )
    cfg = load_config(write(tmp_path, good))
    assert cfg.models[0].backend.param_caps == {"steps": 40, "fps": 24}

    bad = MINIMAL.replace(
        "      preset: openai-chat\n",
        "      preset: openai-chat\n      param_caps: {steps: forty}\n",
    )
    with pytest.raises(ConfigError, match="param_caps"):
        load_config(write(tmp_path, bad))


# --- confidential model entries ---------------------------------------------

CONFIDENTIAL_YAML = """
provider:
  api_url: http://localhost:8402
  capacity: 2
models:
  - model: some/e2ee-model:fp8
    confidential: true
    slas: {"1h": {rate_in: "0.03", rate_out: "0.09"}}
    backend:
      preset: openai-chat
      base_url: https://inference.example.com/v1
      model: some/e2ee-model:fp8
      api_key: token
  - model: regular/model:fp8
    slas: {"1h": {rate_out: "0.09"}}
    backend:
      preset: openai-chat
      base_url: http://127.0.0.1:9999/v1
      model: runtime-id
"""


def test_confidential_entry_flags_the_model_and_keeps_the_backend_ordinary(tmp_path):
    """`confidential: true` changes what the daemon IS — an attested identity
    with a per-boot payload key — and nothing about how the model is served: the
    backend block is the same preset or raw mapping any model takes."""
    cfg = load_config(write(tmp_path, CONFIDENTIAL_YAML))
    assert cfg.confidential is True
    conf, regular = cfg.models
    assert conf.confidential and not regular.confidential
    assert conf.backend.preset == "openai-chat"
    assert conf.backend.params["base_url"] == "https://inference.example.com/v1"
    assert conf.backend.params["api_key"] == "token"
    assert cfg.provider.box_key is None    # ephemeral in confidential mode


def test_confidential_flag_alone_decides(tmp_path):
    """The `e2ee-` name prefix is a market naming convention with no code
    semantics: `confidential: true` is the only switch. A plain name flagged
    confidential is confidential; a prefixed name left unflagged is regular."""
    plain_named = CONFIDENTIAL_YAML.replace("some/e2ee-model:fp8", "some/model:fp8")
    conf = load_config(write(tmp_path, plain_named)).models[0]
    assert conf.model == "some/model:fp8"
    assert conf.confidential is True

    prefixed_unflagged = CONFIDENTIAL_YAML.replace("regular/model:fp8", "regular/e2ee-model:fp8")
    regular = load_config(write(tmp_path, prefixed_unflagged)).models[1]
    assert regular.model == "regular/e2ee-model:fp8"
    assert regular.confidential is False


def test_box_key_forbidden_in_confidential_mode_required_otherwise(tmp_path):
    with_box = CONFIDENTIAL_YAML.replace("capacity: 2", 'capacity: 2\n  box_key: "aa" ')
    with pytest.raises(ConfigError, match="ephemeral"):
        load_config(write(tmp_path, with_box))
    # A config with no confidential model still requires the operator box key.
    regular_only = CONFIDENTIAL_YAML.replace("    confidential: true\n", "")
    with pytest.raises(ConfigError, match="box_key"):
        load_config(write(tmp_path, regular_only))


def test_modality_is_optional_and_validated(tmp_path):
    from vorqd.config import ConfigError, load_config

    def cfg(line: str) -> str:
        return f"""
provider:
  wallet_key: "0x{'4a' * 32}"
  box_key: "{'aa' * 32}"
  api_url: http://x
  capacity: 1
models:
  - model: org/m:fp8
{line}
    slas:
      "1h": {{rate_out: "0.6"}}
    backend:
      preset: openai-chat
      base_url: http://r/v1
      model: rt
"""

    p = tmp_path / "v.yaml"
    p.write_text(cfg(""))
    assert load_config(p).models[0].modality is None   # the catalog is the normal source

    p.write_text(cfg('    modality: image'))
    assert load_config(p).models[0].modality == "image"

    # An input-metered modality: priced on prompt tokens, settling at completionTok 0.
    p.write_text(cfg('    modality: embedding'))
    assert load_config(p).models[0].modality == "embedding"

    p.write_text(cfg('    modality: audio'))
    with pytest.raises(ConfigError, match="modality"):
        load_config(p)


# --- dynamic pricing ---------------------------------------------------------

_PRICING_BASE = """
provider:
  box_key: "aa"
  api_url: http://x
  capacity: 2
{pricing}
models:
  - model: m:fp8
    modality: text
    slas: { "1h": { rate_out: "0.6" } }
    backend: { preset: openai-chat, base_url: http://b/v1, model: r }
"""


def _pricing_config(tmp_path, block: str = ""):
    # `.replace`, not `.format`: the YAML carries flow mappings of its own.
    return load_config(write(tmp_path, _PRICING_BASE.replace("{pricing}", block)))


def test_pricing_defaults_are_inert(tmp_path):
    """No `pricing:` block must accept exactly the configured rates."""
    pricing = _pricing_config(tmp_path).provider.pricing
    assert pricing.max_discount_pct == 0
    assert pricing.bid_tolerance_pct == 0
    assert pricing.low_load_pct == 30


def test_pricing_block_is_parsed(tmp_path):
    cfg = _pricing_config(tmp_path, """  pricing:
    max_discount_pct: 20
    bid_tolerance_pct: 10
    low_load_pct: 40
""")
    pricing = cfg.provider.pricing
    assert (pricing.max_discount_pct, pricing.bid_tolerance_pct) == (20, 10)
    assert pricing.low_load_pct == 40


def test_a_discount_over_ninety_percent_is_refused(tmp_path):
    """Past 90 the smallest rate rounds away and the floor stops being one."""
    with pytest.raises(ConfigError, match="max_discount_pct"):
        _pricing_config(tmp_path, "  pricing: { max_discount_pct: 91 }\n")
    with pytest.raises(ConfigError, match="bid_tolerance_pct"):
        _pricing_config(tmp_path, "  pricing: { bid_tolerance_pct: 91 }\n")


def test_pricing_bounds_and_typos_are_named(tmp_path):
    with pytest.raises(ConfigError, match="low_load_pct"):
        _pricing_config(tmp_path, "  pricing: { low_load_pct: 101 }\n")
    with pytest.raises(ConfigError, match="max_discout_pct"):
        _pricing_config(tmp_path, "  pricing: { max_discout_pct: 10 }\n")


def test_the_resting_bid_knob_is_gone_and_a_config_naming_it_is_refused(tmp_path):
    """Bid age is a coordinator-side query filter now, not daemon state.

    A knob that quietly stopped doing anything would be worse than one that was
    never there: the operator would keep configuring a subsidy the daemon no
    longer grants. So the loader names it as an unknown key.
    """
    assert not hasattr(PricingConfig(), "rest_tolerance_after_s")
    with pytest.raises(ConfigError, match="rest_tolerance_after_s"):
        _pricing_config(tmp_path, "  pricing: { rest_tolerance_after_s: 3600 }\n")


def test_the_bid_filter_block_is_parsed(tmp_path):
    """Still accepted, so an operator's older YAML keeps loading — the scheduler
    logs it as inert, since the coordinator's matcher decides which bids this
    daemon is offered."""
    cfg = _pricing_config(tmp_path, "  bid_filter: { min_age_s: 3600 }\n")
    assert cfg.provider.bid_filter == {"min_age_s": 3600}


def test_no_bid_filter_block_asks_for_no_narrowing(tmp_path):
    assert _pricing_config(tmp_path).provider.bid_filter == {}


def test_a_bid_filter_typo_is_named_rather_than_forwarded(tmp_path):
    """A validated passthrough, not a raw one.

    A key the daemon does not know is a typo, and a node that will simply ignore
    it turns that typo into a filter the operator believes is on and is not.
    """
    with pytest.raises(ConfigError, match="min_age_secs"):
        _pricing_config(tmp_path, "  bid_filter: { min_age_secs: 3600 }\n")


def test_the_bid_filter_cannot_override_what_the_scheduler_decides(tmp_path):
    """`min_rate_out`, `min_rate_in` and `limit` are decisions, not settings.

    They carry the private floor and the daemon's free capacity. An operator who
    could set them by hand could widen the poll past what the daemon will claim
    or, worse, narrow it under the floor it just computed.
    """
    for key in ("min_rate_out", "min_rate_in", "limit"):
        with pytest.raises(ConfigError, match=key):
            _pricing_config(tmp_path, "  bid_filter: { %s: 1 }\n" % key)


def test_a_negative_bid_age_is_refused(tmp_path):
    """It would name a block ahead of the chain's head: a filter for bids posted
    in the future, which is every bid or none depending on the node."""
    with pytest.raises(ConfigError, match="min_age_s"):
        _pricing_config(tmp_path, "  bid_filter: { min_age_s: -1 }\n")


def test_load_probe_is_parsed(tmp_path):
    cfg = load_config(write(tmp_path, """
provider: { box_key: "aa", api_url: http://x, capacity: 2 }
models:
  - model: m:fp8
    modality: text
    slas: { "1h": { rate_out: "0.6" } }
    backend: { preset: openai-chat, base_url: http://b/v1, model: r }
    load:
      url: http://b:8000/metrics
      metric: "vllm:kv_cache_usage_perc"
      scale: 1.0
"""))
    probe = cfg.models[0].load
    assert probe.url == "http://b:8000/metrics"
    assert probe.metric == "vllm:kv_cache_usage_perc"
    assert probe.scale == 1.0


OCCUPANCY_LOAD = """    load: { source: occupancy }
"""


def _with_load(yaml_text: str, block: str) -> str:
    # `load` is a model-level key: it goes beside `backend:`, not inside it.
    return yaml_text.replace("    backend:\n", block + "    backend:\n", 1)


def test_an_occupancy_load_source_is_parsed(tmp_path):
    cfg = load_config(write(tmp_path, _with_load(_with_limits(MINIMAL), OCCUPANCY_LOAD)))
    load = cfg.models[0].load
    assert (load.source, load.url, load.metric) == ("occupancy", None, None)


def test_a_probe_load_source_is_the_default(tmp_path):
    cfg = load_config(write(tmp_path, _with_load(
        MINIMAL, '    load: { url: "http://b:8000/metrics", metric: x }\n')))
    assert cfg.models[0].load.source == "probe"


@pytest.mark.parametrize("block, needle", [
    ('    load: { source: occupancy, url: "http://b/metrics" }\n', "takes no 'url'"),
    ("    load: { source: memory }\n", "'source' must be"),
])
def test_a_bad_load_source_is_refused(tmp_path, block, needle):
    with pytest.raises(ConfigError, match=needle):
        load_config(write(tmp_path, _with_load(_with_limits(MINIMAL), block)))


def test_an_occupancy_source_needs_a_bounded_entry(tmp_path):
    with pytest.raises(ConfigError, match="needs a 'concurrency' or a 'rate_limit'"):
        load_config(write(tmp_path, _with_load(MINIMAL, OCCUPANCY_LOAD)))


def test_no_load_block_leaves_the_model_statically_priced(tmp_path):
    assert _pricing_config(tmp_path).models[0].load is None


def test_a_load_block_nested_inside_backend_is_refused(tmp_path):
    """The one misplacement that would otherwise be silent.

    `backend:` forwards keys it does not recognise to the runtime as request
    params — that is how a preset carries a nonstandard knob — so a `load:` one
    level too deep is swallowed rather than rejected. The daemon then boots,
    publishes, claims at its configured rates and never discounts, and every
    gauge, log line and healthcheck looks exactly as it does when the feature is
    switched off. Which it is.
    """
    with pytest.raises(ConfigError, match="model-level key"):
        load_config(write(tmp_path, """
provider: { box_key: "aa", api_url: http://x, capacity: 2 }
models:
  - model: m:fp8
    modality: text
    slas: { "1h": { rate_out: "0.6" } }
    backend:
      preset: openai-chat
      base_url: http://b/v1
      model: r
      load: { url: "http://b:8000/metrics", metric: "vllm:kv_cache_usage_perc" }
"""))


def test_a_relative_probe_url_is_refused(tmp_path):
    """The metrics endpoint need not share a host or a port with the inference
    API, so there is nothing to join a path onto."""
    with pytest.raises(ConfigError, match="absolute URL"):
        load_config(write(tmp_path, """
provider: { box_key: "aa", api_url: http://x, capacity: 2 }
models:
  - model: m:fp8
    modality: text
    slas: { "1h": { rate_out: "0.6" } }
    backend: { preset: openai-chat, base_url: http://b/v1, model: r }
    load: { url: /metrics, metric: x }
"""))


# --- backend limits ------------------------------------------------------------

LIMITS_BLOCK = """      retries: 6
      retry_statuses: [404, 409]
      retry_backoff_s: 15
      retry_backoff_max_s: 600
      timeout_s: 310
      concurrency: 2
      rate_limit: {"1m": 40, "1h": 900, "24h": 5000}
      trip_after: 2
      trip_cooldown_s: 120
"""


def _with_limits(yaml_text: str, block: str = LIMITS_BLOCK) -> str:
    return yaml_text.replace("      model: runtime-model\n", "      model: runtime-model\n" + block)


def test_backend_limits_load_on_a_preset_entry(tmp_path):
    be = load_config(write(tmp_path, _with_limits(MINIMAL))).models[0].backend
    assert (be.retries, be.retry_backoff_s, be.retry_backoff_max_s) == (6, 15.0, 600.0)
    assert be.retry_statuses == (404, 409)
    assert (be.timeout_s, be.concurrency) == (310.0, 2)
    assert be.rate_limit == {"1m": 40, "1h": 900, "24h": 5000}
    assert (be.trip_after, be.trip_cooldown_s) == (2, 120.0)
    assert not set(be.params) & {"retries", "retry_statuses", "retry_backoff_s",
                                 "retry_backoff_max_s", "timeout_s", "concurrency", "rate_limit",
                                 "trip_after", "trip_cooldown_s"}


def test_backend_limits_default_to_no_limit_and_no_retry(tmp_path):
    be = load_config(write(tmp_path, MINIMAL)).models[0].backend
    assert (be.retries, be.retry_backoff_s, be.retry_backoff_max_s) == (0, 30.0, 900.0)
    assert be.timeout_s is None and be.concurrency is None and be.rate_limit == {}
    assert (be.trip_after, be.trip_cooldown_s) == (3, 60.0)


def test_backend_limits_load_on_a_raw_entry(tmp_path):
    raw = MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model\n",
        "    backend:\n      request: { method: POST, url: http://x }\n"
        "      response: { mode: sync, result: { text: \"$.text\", completion_tokens: \"$.n\" } }\n"
        "      retries: 3\n      concurrency: 1\n      rate_limit: {\"1h\": 10}\n",
    )
    be = load_config(write(tmp_path, raw)).models[0].backend
    assert (be.retries, be.concurrency, be.rate_limit) == (3, 1, {"1h": 10})


def test_backend_limits_load_on_a_confidential_entry(tmp_path):
    text = CONFIDENTIAL_YAML.replace(
        "      api_key: token\n",
        "      api_key: token\n      retries: 2\n      timeout_s: 120\n"
        "      concurrency: 3\n      rate_limit: {\"1m\": 5}\n",
    )
    be = load_config(write(tmp_path, text)).models[0].backend
    assert (be.retries, be.timeout_s, be.concurrency, be.rate_limit) == (2, 120.0, 3, {"1m": 5})


@pytest.mark.parametrize("block, needle", [
    ("      retries: -1\n", "retries"),
    ("      retries: two\n", "retries"),
    ("      retry_statuses: [503]\n", "retry_statuses"),
    ("      retry_statuses: 404\n", "retry_statuses"),
    ("      concurrency: 0\n", "concurrency"),
    ("      concurrency: 2\n      queue: 3\n", "queue"),   # gone: the SLA window holds
    ("      timeout_s: 0\n", "timeout_s"),
    ("      retry_backoff_s: 100\n      retry_backoff_max_s: 50\n", "retry_backoff_max_s"),
    ("      rate_limit: {\"1x\": 5}\n", "rate_limit"),
    ("      rate_limit: {\"1m\": 0}\n", "rate_limit"),
    ("      rate_limit: [40]\n", "rate_limit"),
    ("      trip_after: -1\n", "trip_after"),
    ("      trip_after: two\n", "trip_after"),
    ("      trip_cooldown_s: 0\n", "trip_cooldown_s"),
])
def test_a_bad_backend_limit_is_refused_naming_the_key_and_the_model(tmp_path, block, needle):
    with pytest.raises(ConfigError, match=needle) as exc:
        load_config(write(tmp_path, _with_limits(MINIMAL, block)))
    assert "m:fp8" in str(exc.value)


# --- backend.stream ------------------------------------------------------------


def test_stream_loads_on_a_chat_preset_and_never_reaches_the_body(tmp_path):
    be = load_config(write(tmp_path, _with_limits(MINIMAL, "      stream: true\n"))).models[0].backend
    assert be.stream is True
    assert "stream" not in be.params


def test_stream_defaults_off(tmp_path):
    assert load_config(write(tmp_path, MINIMAL)).models[0].backend.stream is False


def test_stream_is_refused_off_the_chat_preset(tmp_path):
    embeddings = _with_limits(MINIMAL, "      stream: true\n").replace(
        "preset: openai-chat", "preset: openai-embeddings")
    with pytest.raises(ConfigError, match="openai-chat preset only"):
        load_config(write(tmp_path, embeddings))
    with pytest.raises(ConfigError, match="must be true or false"):
        load_config(write(tmp_path, _with_limits(MINIMAL, "      stream: yes please\n")))


# --- provider.state_db ---------------------------------------------------------


def test_state_db_defaults_to_a_file_beside_the_daemon(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.provider.state_db == "vorqd-state.sqlite"


def test_state_db_is_read_from_the_provider_block(tmp_path):
    cfg = MINIMAL.replace("  capacity: 1\n", "  capacity: 1\n  state_db: /var/lib/vorqd/state.sqlite\n")
    assert cfg != MINIMAL
    assert load_config(write(tmp_path, cfg)).provider.state_db == "/var/lib/vorqd/state.sqlite"


def test_a_nulled_state_db_falls_back_to_the_default_file(tmp_path):
    cfg = MINIMAL.replace("  capacity: 1\n", "  capacity: 1\n  state_db: null\n")
    assert cfg != MINIMAL
    assert load_config(write(tmp_path, cfg)).provider.state_db == "vorqd-state.sqlite"


# --- poll.handle / poll.max_polls and the openai-responses preset --------------


def _raw_poll_yaml(poll_block: str) -> str:
    """`poll_block` is one or more lines, each indented by ten spaces."""
    return f"""
provider: {{ wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }}
models:
  - model: m:fp8
    modality: image
    slas: {{ "24h": {{ rate_out: "0.000001" }} }}
    backend:
      request: {{ method: POST, url: http://q/submit }}
      response:
        mode: poll
        poll:
{poll_block}
          status_field: "$.status"
          done_values: [done]
        result: {{ media_urls: "$.images[*].url" }}
"""


def test_poll_handle_and_max_polls_load(tmp_path):
    cfg = load_config(write(tmp_path, _raw_poll_yaml(
        '          handle: "$.id"\n'
        '          request: { method: GET, url: "http://q/status/{handle}" }\n'
        '          max_polls: 12')))
    poll = cfg.models[0].backend.response["poll"]
    assert poll["handle"] == "$.id" and poll["max_polls"] == 12


def test_poll_handle_requires_a_poll_request_not_a_status_url(tmp_path):
    with pytest.raises(ConfigError, match="handle.*request"):
        load_config(write(tmp_path, _raw_poll_yaml(
            '          handle: "$.id"\n'
            '          status_url: "$.status_url"')))


def test_a_resumable_poll_request_cannot_template_the_submit_response(tmp_path):
    with pytest.raises(ConfigError, match="submit"):
        load_config(write(tmp_path, _raw_poll_yaml(
            '          handle: "$.id"\n'
            '          request: { method: POST, url: http://q/status, body: { id: "{submit.id}" } }')))


def test_a_spaced_submit_reference_is_refused_in_a_resumable_poll_request(tmp_path):
    # The template engine strips whitespace inside the braces, so `{ submit.id }`
    # is the same reference — a literal substring match would let it through and
    # the resumed poll would render an empty id.
    with pytest.raises(ConfigError, match="submit"):
        load_config(write(tmp_path, _raw_poll_yaml(
            '          handle: "$.id"\n'
            '          request: { method: POST, url: http://q/status, body: { id: "{ submit.id }" } }')))


def test_a_poll_request_that_uses_the_handle_must_declare_one(tmp_path):
    # Nothing records a handle, so `{handle}` renders empty and the poll goes to
    # a URL that names no job at all.
    with pytest.raises(ConfigError, match="handle"):
        load_config(write(tmp_path, _raw_poll_yaml(
            '          request: { method: GET, url: "http://q/status/{handle}" }')))


def test_poll_handle_must_be_a_jsonpath(tmp_path):
    with pytest.raises(ConfigError, match="handle"):
        load_config(write(tmp_path, _raw_poll_yaml(
            '          handle: id\n'
            '          request: { method: GET, url: "http://q/status/{handle}" }')))


@pytest.mark.parametrize("value", ["0", "-1", "true", "2.5", "sixty"])
def test_poll_max_polls_must_be_a_positive_integer(tmp_path, value):
    with pytest.raises(ConfigError, match="max_polls"):
        load_config(write(tmp_path, _raw_poll_yaml(
            '          request: { method: GET, url: "http://q/status/1" }\n'
            f'          max_polls: {value}')))


def _responses_yaml(extra: str = "") -> str:
    """`extra` is one or more backend lines, each indented by six spaces."""
    return f"""
provider: {{ wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }}
models:
  - model: m:fp8
    slas: {{ "1h": {{ rate_out: "0.000001" }} }}
    backend:
      preset: openai-responses
      base_url: http://r/v1
      model: rt
{extra}
"""


def test_openai_responses_preset_loads_with_a_tier(tmp_path):
    cfg = load_config(write(tmp_path, _responses_yaml("      service_tier: flex")))
    be = cfg.models[0].backend
    assert be.preset == "openai-responses"
    assert be.params["service_tier"] == "flex"


def test_openai_responses_service_tier_is_flex_or_priority(tmp_path):
    with pytest.raises(ConfigError, match="service_tier"):
        load_config(write(tmp_path, _responses_yaml("      service_tier: turbo")))


def test_openai_responses_refuses_stream(tmp_path):
    with pytest.raises(ConfigError, match="stream"):
        load_config(write(tmp_path, _responses_yaml("      stream: true")))


def _batch_yaml(extra: str = "") -> str:
    """`extra` is one or more backend lines, each indented by six spaces."""
    return f"""
provider: {{ wallet_key: dev, box_key: dev-box, api_url: http://x, capacity: 1 }}
models:
  - model: m:fp8
    slas: {{ "24h": {{ rate_in: "0.000001", rate_out: "0.000002" }} }}
    backend:
      preset: openai-batch
      base_url: http://r/v1
      model: rt
{extra}
"""


def test_openai_batch_preset_validates_its_keys(tmp_path):
    cfg = load_config(write(tmp_path, _batch_yaml()))
    assert cfg.models[0].backend.preset == "openai-batch"
    for extra, needle in (
        ("      completion_window: 24\n", "completion_window"),
        ("      endpoint: chat/completions\n", "endpoint"),
        ("      max_polls: 0\n", "max_polls"),
        ("      stream: true\n", "stream"),
    ):
        with pytest.raises(ConfigError, match=needle):
            load_config(write(tmp_path, _batch_yaml(extra)))


# --- one backend per SLA window ------------------------------------------------

_SLA_BACKENDS = MINIMAL + """
models:
  - model: m:fp8
    slas:
      "1h":  {rate_in: "0.000001", rate_out: "0.000002"}
      "24h": {rate_in: "0.000001", rate_out: "0.000001"}
    backend:
      preset: openai-responses
      base_url: http://r/v1
      api_key: env:VORQ_TEST_KEY
      model: rt
      service_tier: flex
      params_supported: [temperature]
    sla_backends:
      "24h":
        preset: openai-batch
        service_tier: null
        params_supported: [temperature, top_k]
"""


def test_sla_backends_patch_over_the_model_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("VORQ_TEST_KEY", "k")
    cfg = load_config(write(tmp_path, _SLA_BACKENDS))
    model = cfg.models[0]
    assert model.backend.preset == "openai-responses"
    over = model.sla_backends["24h"]
    assert over.preset == "openai-batch"
    assert over.params["base_url"] == "http://r/v1" and over.params["model"] == "rt"   # inherited
    assert over.params["api_key"] == "k"
    assert "service_tier" not in over.params                                          # nulled away
    assert over.params["params_supported"] == ["temperature", "top_k"]                # patched
    assert model.backend.params["params_supported"] == ["temperature"]                # untouched


@pytest.mark.parametrize("patch, needle", [
    ('    sla_backends:\n      "12h": {preset: openai-batch}\n', "12h"),
    ('    sla_backends:\n      "24h": {preset: openai-batch, retries: 2}\n', "retries"),
    ('    sla_backends: [openai-batch]\n', "sla_backends"),
])
def test_sla_backends_refuses_unknown_windows_limits_and_shapes(tmp_path, patch, needle):
    text = MINIMAL + """
models:
  - model: m:fp8
    slas: {"24h": {rate_in: "0.000001", rate_out: "0.000001"}}
    backend: {preset: openai-chat, base_url: http://r/v1, model: rt}
""" + patch
    with pytest.raises(ConfigError, match=needle):
        load_config(write(tmp_path, text))


def test_backend_for_answers_the_window_its_own_backend_or_the_default(tmp_path):
    """What a job is served through is chosen by its window, and everything that
    reads the backend for a running job — the media caps included — goes through
    this one lookup rather than reaching for `backend` directly."""
    text = MINIMAL + """
models:
  - model: m:fp8
    slas:
      "1h":  {rate_out: "0.000001"}
      "24h": {rate_out: "0.000001"}
    backend:
      preset: openai-chat
      base_url: http://r/v1
      model: rt
      param_caps: {steps: 40}
    sla_backends:
      "24h":
        param_caps: {steps: 20}
"""
    model = load_config(write(tmp_path, text)).models[0]
    assert model.backend_for("24h").param_caps == {"steps": 20}
    assert model.backend_for("1h") is model.backend           # no override for this window
    assert model.backend_for(None) is model.backend


def test_a_patched_preset_drops_an_inherited_raw_mapping(tmp_path):
    text = MINIMAL + """
models:
  - model: m:fp8
    slas: {"24h": {rate_out: "0.000001"}}
    backend:
      request: { method: POST, url: http://x, body: {} }
      response:
        mode: sync
        result: { text: "$.a", completion_tokens: "$.n" }
    sla_backends:
      "24h":
        preset: openai-batch
        base_url: http://r/v1
        model: rt
"""
    over = load_config(write(tmp_path, text)).models[0].sla_backends["24h"]
    assert over.preset == "openai-batch"
    assert over.request is None and over.response is None


def test_a_patched_raw_mapping_drops_an_inherited_preset(tmp_path):
    text = MINIMAL + """
models:
  - model: m:fp8
    slas: {"24h": {rate_out: "0.000001"}}
    backend: {preset: openai-chat, base_url: http://r/v1, model: rt}
    sla_backends:
      "24h":
        request: { method: POST, url: http://x, body: {} }
        response:
          mode: sync
          result: { text: "$.a", completion_tokens: "$.n" }
"""
    over = load_config(write(tmp_path, text)).models[0].sla_backends["24h"]
    assert over.preset is None
    assert over.request == {"method": "POST", "url": "http://x", "body": {}}
    assert over.response["result"] == {"text": "$.a", "completion_tokens": "$.n"}


# -- max_input_bytes_per_unit: the floor under a client-declared units_in -------


def test_input_bytes_per_unit_defaults_to_the_module_constant(tmp_path):
    # Against the constant, never against the literal: the default is a policy
    # decision recorded in one place, and a test repeating the number would let
    # the two drift apart while staying green.
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.provider.max_input_bytes_per_unit == MAX_INPUT_BYTES_PER_UNIT


def test_input_bytes_per_unit_is_read_from_the_provider_block(tmp_path):
    cfg = MINIMAL.replace("  capacity: 1\n", "  capacity: 1\n  max_input_bytes_per_unit: 8\n")
    assert load_config(write(tmp_path, cfg)).provider.max_input_bytes_per_unit == 8


def test_zero_bytes_per_unit_is_valid_and_turns_the_guard_off(tmp_path):
    cfg = MINIMAL.replace("  capacity: 1\n", "  capacity: 1\n  max_input_bytes_per_unit: 0\n")
    assert load_config(write(tmp_path, cfg)).provider.max_input_bytes_per_unit == 0


@pytest.mark.parametrize("value", ["-1", "4.5", "true", "lots"])
def test_a_bytes_per_unit_that_is_not_a_count_is_refused_naming_the_key(tmp_path, value):
    cfg = MINIMAL.replace("  capacity: 1\n",
                          f"  capacity: 1\n  max_input_bytes_per_unit: {value}\n")
    with pytest.raises(ConfigError, match="max_input_bytes_per_unit"):
        load_config(write(tmp_path, cfg))


def test_input_bytes_per_unit_from_the_environment_reads_as_a_number(tmp_path, monkeypatch):
    monkeypatch.setenv("VORQ_BPU", "8")
    cfg = MINIMAL.replace("  capacity: 1\n",
                          "  capacity: 1\n  max_input_bytes_per_unit: env:VORQ_BPU\n")
    assert load_config(write(tmp_path, cfg)).provider.max_input_bytes_per_unit == 8


# -- backend.reference.accept: which reference types this backend takes ---------


def test_a_backend_names_no_reference_types_by_default(tmp_path):
    """Silence means "whatever the daemon can read". An operator opts into a
    narrower list; nobody opts into a wider one, because wider means a decoder
    that does not exist."""
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.models[0].backend.reference_accept is None


def test_a_backend_may_narrow_the_reference_types_it_takes(tmp_path):
    cfg = MINIMAL.replace(
        "      model: runtime-model\n",
        "      model: runtime-model\n      reference: { accept: [image/png, image/jpeg] }\n")
    assert load_config(write(tmp_path, cfg)).models[0].backend.reference_accept == \
        ["image/png", "image/jpeg"]


def test_a_reference_type_the_daemon_cannot_read_is_refused_at_load(tmp_path):
    """Caught at boot, not on the first job that carries one: an accept list
    naming a type with no reader would take every such job to a claim and then
    hand it straight back."""
    cfg = MINIMAL.replace(
        "      model: runtime-model\n",
        "      model: runtime-model\n      reference: { accept: [image/gif] }\n")
    with pytest.raises(ConfigError, match="image/gif"):
        load_config(write(tmp_path, cfg))


def test_an_empty_accept_list_is_refused_rather_than_read_as_no_opinion(tmp_path):
    """`accept: []` is a backend that takes no reference at all, which is what
    omitting the whole block already says less confusingly. Read as "no opinion"
    it would silently mean the opposite of what it looks like."""
    cfg = MINIMAL.replace(
        "      model: runtime-model\n",
        "      model: runtime-model\n      reference: { accept: [] }\n")
    with pytest.raises(ConfigError, match="reference"):
        load_config(write(tmp_path, cfg))


def test_an_unknown_key_in_the_reference_block_is_refused(tmp_path):
    cfg = MINIMAL.replace(
        "      model: runtime-model\n",
        "      model: runtime-model\n      reference: { accpet: [image/png] }\n")
    with pytest.raises(ConfigError, match="reference"):
        load_config(write(tmp_path, cfg))


def test_a_raw_mapping_backend_carries_reference_accept_too(tmp_path):
    """Reference-conditioned media is served by raw mappings — the same reason
    `param_caps` had to survive this branch."""
    raw = MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model",
        "    backend:\n"
        "      request: { method: POST, url: 'http://x', body: {} }\n"
        "      response: { mode: sync, result: { media_b64: '$.images[*].b64' } }\n"
        "      reference: { accept: [image/png] }")
    assert load_config(write(tmp_path, raw)).models[0].backend.reference_accept == ["image/png"]


# -- backend.durations: the clip lengths a backend renders at all ---------------


def test_a_backend_renders_any_length_by_default(tmp_path):
    assert load_config(write(tmp_path, MINIMAL)).models[0].backend.durations is None


def test_a_backend_may_list_the_clip_lengths_it_renders(tmp_path):
    cfg = MINIMAL.replace(
        "      model: runtime-model\n",
        "      model: runtime-model\n      durations: [5, 10]\n")
    assert load_config(write(tmp_path, cfg)).models[0].backend.durations == [5, 10]


@pytest.mark.parametrize("written", ["[]", "[0, 5]", "[5, '10']", "[true]", "[2.5]", "5"])
def test_a_list_of_lengths_that_is_not_positive_whole_seconds_is_refused(tmp_path, written):
    """An empty list is a backend that renders nothing; a string or a fraction is a
    length the clamp cannot compare. Either would fail every job after its claim."""
    cfg = MINIMAL.replace(
        "      model: runtime-model\n",
        f"      model: runtime-model\n      durations: {written}\n")
    with pytest.raises(ConfigError, match="durations"):
        load_config(write(tmp_path, cfg))


def test_a_raw_mapping_backend_carries_its_durations_too(tmp_path):
    raw = MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model",
        "    backend:\n"
        "      request: { method: POST, url: 'http://x', body: {} }\n"
        "      response: { mode: sync, result: { media_b64: '$.images[*].b64' } }\n"
        "      durations: [5, 10]")
    assert load_config(write(tmp_path, raw)).models[0].backend.durations == [5, 10]


# -- request.prepare: requests a raw submit depends on ---------------------------

_RAW = ("    backend:\n"
        "      request:\n"
        "        method: POST\n        url: 'http://x'\n        body: {}\n"
        "        prepare: PREPARE\n"
        "      response: { mode: sync, result: { media_urls: '$.urls[*]' } }")


def _with_prepare(written: str) -> str:
    return MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model",
        _RAW.replace("PREPARE", written))


def test_a_raw_request_may_name_the_requests_it_depends_on(tmp_path):
    cfg = load_config(write(tmp_path, _with_prepare(
        "[{ name: first, when: input.image, url: 'http://up', extract: { url: '$.data.url' } }]")))
    assert cfg.models[0].backend.request["prepare"][0]["name"] == "first"


@pytest.mark.parametrize("written", [
    "{ name: first }",                                                         # not a list
    "[{ url: 'http://up', extract: { url: '$.u' } }]",                         # no name
    "[{ name: 'a-b', url: 'http://up', extract: { url: '$.u' } }]",            # not addressable
    "[{ name: first, extract: { url: '$.u' } }]",                              # no url
    "[{ name: first, url: 'http://up' }]",                                     # nothing to extract
    "[{ name: first, url: 'http://up', extract: { url: 'data.u' } }]",         # not a JSONPath
    "[{ name: a, url: 'http://up', extract: { u: '$.u' } }, { name: a, url: 'http://up', extract: { u: '$.u' } }]",
    "[{ name: first, url: 'http://up', extract: { url: '$.u' }, wehn: input.image }]",
])
def test_a_prepare_step_that_could_not_work_is_refused_at_load(tmp_path, written):
    with pytest.raises(ConfigError, match="prepare"):
        load_config(write(tmp_path, _with_prepare(written)))


# -- the rest of a media backend's policy -----------------------------------------

_MEDIA_RAW = ("    backend:\n"
              "      request: { method: POST, url: 'http://x', body: {} }\n"
              "      response:\n"
              "        mode: sync\n"
              "        result: { media_urls: '$.urls[*]' }\n"
              "EXTRA")


def _media_backend(extra: str) -> str:
    return MINIMAL.replace(
        "    backend:\n      preset: openai-chat\n      base_url: http://localhost:8000/v1\n      model: runtime-model",
        _MEDIA_RAW.replace("EXTRA", extra).rstrip("\n"))


def test_a_media_backends_policy_is_one_set_of_arguments_for_the_planner(tmp_path):
    cfg = load_config(write(tmp_path, _media_backend(
        "      reference:\n"
        "        accept: [image/png, video/mp4, audio/mpeg]\n"
        "        still: { min_side: 300, max_ratio: 2.5 }\n"
        "        clip: { max_secs: 15, max_total_secs: 15, max_count: 3 }\n"
        "        audio: { max_bytes: 1000 }\n"
        "      resolutions: [480p, 720p]\n"
        "      durations: [4, 5]\n"
        "      auto_duration: -1\n"
        "      adaptive_aspect: adaptive\n")))
    assert cfg.models[0].backend.media_policy == {
        "accept": ["image/png", "video/mp4", "audio/mpeg"],
        "bounds": {"still": {"min_side": 300, "max_ratio": 2.5},
                   "clip": {"max_secs": 15, "max_total_secs": 15, "max_count": 3},
                   "audio": {"max_bytes": 1000}},
        "resolutions": ["480p", "720p"], "durations": [4, 5],
        "auto_duration": -1, "adaptive_aspect": "adaptive",
    }


def test_a_backend_with_no_media_policy_passes_none_of_it(tmp_path):
    assert load_config(write(tmp_path, MINIMAL)).models[0].backend.media_policy == {
        "accept": None, "bounds": {}, "resolutions": None, "durations": None,
        "auto_duration": None, "adaptive_aspect": None}


@pytest.mark.parametrize("extra,names", [
    ("      reference: { accept: [image/png], still: { min_sied: 300 } }\n", "min_sied"),
    ("      reference: { accept: [image/png], still: { min_side: -1 } }\n", "min_side"),
    ("      reference: { accept: [image/png], picture: { min_side: 1 } }\n", "picture"),
    ("      resolutions: [480p, 8k]\n", "resolutions"),
    ("      resolutions: []\n", "resolutions"),
    ("      auto_duration: [1]\n", "auto_duration"),
    ("      adaptive_aspect: { keep: true }\n", "adaptive_aspect"),
])
def test_a_media_policy_that_could_not_work_is_refused_at_load(tmp_path, extra, names):
    with pytest.raises(ConfigError, match=names):
        load_config(write(tmp_path, _media_backend(extra)))


def test_an_in_body_status_block_is_checked_at_load(tmp_path):
    good = _media_backend("").replace(
        "        mode: sync\n",
        "        mode: sync\n        ok: { field: '$.code', values: [200], retry: [429], message: '$.msg' }\n")
    assert load_config(write(tmp_path, good)).models[0].backend.response["ok"]["values"] == [200]
    for bad in ("{ field: code, values: [200] }", "{ field: '$.code' }",
                "{ field: '$.code', values: [200], retyr: [429] }"):
        with pytest.raises(ConfigError, match="response.ok"):
            load_config(write(tmp_path, good.replace(
                "{ field: '$.code', values: [200], retry: [429], message: '$.msg' }", bad)))


def test_a_prepare_step_may_loop_but_not_loop_and_branch(tmp_path):
    step = "[{ name: refs, for_each: input.reference_images, url: 'http://up', extract: { url: '$.u' }WHEN }]"
    load_config(write(tmp_path, _with_prepare(step.replace("WHEN", ""))))
    with pytest.raises(ConfigError, match="for_each"):
        load_config(write(tmp_path, _with_prepare(step.replace("WHEN", ", when: input.x"))))
