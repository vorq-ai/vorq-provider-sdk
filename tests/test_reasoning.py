"""Reasoning dialects: the canonical ladder collapsed onto a backend's own enum."""

from __future__ import annotations

import pytest

from vorqd.reasoning import (
    LADDER,
    REASONING_PRESETS,
    budget_for_effort,
    effort_map,
    extract_default_effort,
    resolve_reasoning,
)

CANONICAL = ("none", *LADDER)


def _values(efforts, budget=None) -> dict:
    return effort_map(efforts, budget)["reasoning_effort"]["values"]


def test_full_ladder_is_forwarded_unchanged():
    v = _values(list(CANONICAL))
    for level in CANONICAL:
        assert v[level] == level


def test_maps_to_the_nearest_level_not_above_the_one_asked_for():
    """The rule that replaces the old hand-written tables: never move a client up
    into a level it did not ask for and would pay more for."""
    v = _values(["none", "high", "max"])
    assert v["none"] == "none"
    assert v["high"] == "high"
    assert v["xhigh"] == "high"      # nearest accepted level below xhigh, not max
    assert v["max"] == "max"


def test_below_the_floor_rises_to_the_floor():
    """No accepted level is cheap enough — fall back to the cheapest that exists
    rather than fail a job the model could still serve."""
    v = _values(["none", "high", "max"])
    assert v["low"] == v["medium"] == "high"


def test_mid_ladder_collapses_downward():
    v = _values(["none", "low", "high"])
    assert v["none"] == "none"
    assert v["low"] == v["medium"] == "low"
    assert v["high"] == v["xhigh"] == v["max"] == "high"

    v = _values(["none", "medium", "high"])
    assert v["low"] == v["medium"] == "medium"       # low is below the floor
    assert v["high"] == v["xhigh"] == v["max"] == "high"


def test_dialect_without_an_off_value_floors_none():
    """`none` cannot be honored, so it rises to the cheapest thinking level —
    the client keeps paying for reasoning it asked to disable, which is why such
    a model must be quoted against a reasoning-sized units_out."""
    v = _values(["low", "medium", "high"])
    assert v["none"] == "low"
    assert v["medium"] == "medium"
    assert v["high"] == v["xhigh"] == v["max"] == "high"


def test_declaring_a_budget_renames_the_cap():
    assert effort_map(["none", "low", "high"], "reasoning_budget")["reasoning_max_tokens"] == "reasoning_budget"


def test_omitting_a_budget_drops_the_cap():
    """An unpublished field risks a 400, and a rejected request is a job the
    provider fails at its own cost — so the cap is dropped, not invented."""
    spec = effort_map(["none", "low", "high"])["reasoning_max_tokens"]
    assert spec["unmapped"] == "drop" and spec["values"] == {}


def test_no_published_control_drops_everything():
    m = effort_map([])
    for key in ("reasoning_effort", "reasoning_max_tokens"):
        assert m[key]["unmapped"] == "drop"
        assert m[key]["values"] == {}


def test_every_dialect_covers_the_whole_canonical_ladder():
    """Any canonical value a client can send must map (or deliberately drop) —
    a hole would leak the raw value into a backend that cannot parse it."""
    dialects = [["none", "high", "max"], ["none", "low", "high"], ["none", "medium", "high"],
                ["low", "medium", "high"], list(CANONICAL)]
    for efforts in dialects:
        v = _values(efforts)
        for level in CANONICAL:
            assert level in v, f"{efforts} does not map {level!r}"
            assert v[level] in efforts, f"{efforts} maps {level!r} outside the accepted enum"


def test_only_none_accepted_drops_every_thinking_level():
    """`efforts: [none]` can honor the off switch and nothing else: every ladder
    level drops (the backend default stands) rather than being folded into an
    off it did not ask for."""
    v = _values(["none"])
    assert v == {"none": "none"}
    m = effort_map(["none"])
    assert m["reasoning_effort"]["unmapped"] == "drop"


def test_effort_map_rejects_a_non_canonical_level():
    with pytest.raises(ValueError, match="minimal"):
        effort_map(["none", "minimal", "high"])


# --- template-switch dialects ------------------------------------------------


def test_thinking_bool_collapses_the_ladder_to_a_switch():
    e = REASONING_PRESETS["thinking_bool"]["reasoning_effort"]
    assert e["to"] == "chat_template_kwargs"
    assert e["values"]["none"] == {"enable_thinking": False}
    for level in LADDER:
        assert e["values"][level] == {"enable_thinking": True}


def test_thinking_bool_low_effort_keeps_three_states():
    e = REASONING_PRESETS["thinking_bool_low_effort"]["reasoning_effort"]
    assert e["values"]["none"] == {"enable_thinking": False}
    for level in ("low", "medium"):
        assert e["values"][level] == {"enable_thinking": True, "low_effort": True}
    for level in ("high", "xhigh", "max"):
        # low_effort is stated, not omitted: an unset flag would inherit the
        # template default and make `high` indistinguishable from `low`.
        assert e["values"][level] == {"enable_thinking": True, "low_effort": False}


def test_template_dialects_drop_the_cap():
    for name in REASONING_PRESETS:
        spec = REASONING_PRESETS[name]["reasoning_max_tokens"]
        assert spec["unmapped"] == "drop" and spec["values"] == {}


# --- derived thinking budget --------------------------------------------------


def test_budget_is_the_efforts_share_of_the_cap():
    assert budget_for_effort("low", 10_000) == 2_000
    assert budget_for_effort("medium", 10_000) == 5_000
    assert budget_for_effort("high", 10_000) == 8_000
    assert budget_for_effort("xhigh", 10_000) == 9_500
    assert budget_for_effort("max", 10_000) == 9_500


def test_budget_floors_then_stays_strictly_below_the_cap():
    assert budget_for_effort("low", 4096) == 1024    # 20% of 4096 is below the useful floor
    assert budget_for_effort("max", 1024) == 1023    # never equals the cap: the answer needs room
    assert budget_for_effort("high", 1) is None      # no room below the cap at all


def test_budget_ceiling_bounds_huge_caps():
    assert budget_for_effort("max", 1_000_000) == 128_000


def test_no_budget_for_off_unknown_or_missing_cap():
    assert budget_for_effort("none", 10_000) is None
    assert budget_for_effort("minimal", 10_000) is None
    assert budget_for_effort("high", None) is None


# --- default_effort -----------------------------------------------------------


def test_extract_default_effort_splits_without_mutating():
    spec = {"efforts": ["none", "high"], "budget": "reasoning_budget", "default_effort": "medium"}
    rest, default = extract_default_effort(spec)
    assert default == "medium"
    assert rest == {"efforts": ["none", "high"], "budget": "reasoning_budget"}
    assert "default_effort" in spec    # YAML anchors share one spec object; it is never mutated


def test_extract_default_effort_accepts_any_canonical_value():
    for level in CANONICAL:
        assert extract_default_effort({"efforts": list(CANONICAL), "default_effort": level})[1] == level


def test_extract_default_effort_rejects_a_non_canonical_value():
    with pytest.raises(ValueError, match="minimal"):
        extract_default_effort({"efforts": ["none"], "default_effort": "minimal"})


def test_extract_default_effort_rejects_a_control_free_dialect():
    with pytest.raises(ValueError, match="efforts"):
        extract_default_effort({"efforts": [], "default_effort": "none"})


def test_extract_passes_specs_without_a_default_through():
    assert extract_default_effort("thinking_bool") == ("thinking_bool", None)
    assert extract_default_effort(None) == (None, None)
    spec = {"efforts": ["none", "high"]}
    assert extract_default_effort(spec) == (spec, None)


def test_extract_default_effort_carries_through_a_template_preset():
    rest, default = extract_default_effort({"preset": "thinking_bool", "default_effort": "medium"})
    assert default == "medium"
    assert resolve_reasoning(rest, None) == REASONING_PRESETS["thinking_bool"]


# --- resolve -----------------------------------------------------------------


def test_resolve_dict_preset_form_expands_the_named_preset():
    assert resolve_reasoning({"preset": "thinking_bool"}, None) == REASONING_PRESETS["thinking_bool"]


def test_resolve_rejects_preset_combined_with_efforts():
    with pytest.raises(ValueError, match="preset"):
        resolve_reasoning({"preset": "thinking_bool", "efforts": ["none"]}, None)


def test_resolve_builds_from_the_mapping_form():
    resolved = resolve_reasoning({"efforts": ["none", "low", "high"], "budget": "reasoning_budget"}, None)
    assert resolved == effort_map(["none", "low", "high"], "reasoning_budget")


def test_resolve_expands_a_named_template_preset():
    assert resolve_reasoning("thinking_bool", None) == REASONING_PRESETS["thinking_bool"]


def test_resolve_merges_inline_map_over_the_dialect():
    merged = resolve_reasoning({"efforts": ["none", "high", "max"]}, {"reasoning_max_tokens": "thinking_budget"})
    assert merged["reasoning_max_tokens"] == "thinking_budget"          # inline wins
    assert merged["reasoning_effort"]["values"]["low"] == "high"        # rest of the dialect kept


def test_resolve_none_spec_returns_inline():
    inline = {"foo": "bar"}
    assert resolve_reasoning(None, inline) == inline
    assert resolve_reasoning(None, None) is None


@pytest.mark.parametrize("spec, match", [
    ("effort_quantum", "effort_quantum"),
    ({"effort": ["none"]}, "effort"),
    ({"budget": "reasoning_budget"}, "efforts"),
    ({"efforts": ["none"], "budget": 4}, "budget"),
    (["none", "high"], "preset name or a mapping"),
])
def test_resolve_rejects_a_malformed_spec(spec, match):
    with pytest.raises(ValueError, match=match):
        resolve_reasoning(spec, None)
