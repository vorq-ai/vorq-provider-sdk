"""Reasoning dialects: the canonical effort ladder mapped onto a backend's own.

The network's canonical reasoning vocabulary is ``reasoning_effort`` —
``none | low | medium | high | xhigh | max`` (``none`` is the explicit off
switch) — and ``reasoning_max_tokens``, the thinking-token budget. Backends
speak narrower dialects: a shorter effort enum, a separately named budget key,
or a chat-template boolean.

Most dialects differ only in *which levels the enum accepts*, so a model
declares that list rather than naming a shape::

    reasoning: { efforts: [none, low, high], budget: reasoning_budget }

transcribed straight from the backend's published schema. The collapse map is
derived, by one rule: **map to the nearest accepted level not above the one
asked for; below the backend's floor, rise to the floor.** A client is never
silently moved up into a more expensive level than it requested, and a level the
dialect cannot represent falls back to a working one rather than failing the job.

`budget` names the backend's thinking-cap field. Omit it when the schema
publishes none — `reasoning_max_tokens` is then dropped rather than sent under
an invented name, since an unpublished field risks a 400 and a rejected request
is a job the provider fails at its own cost. Where a budget field exists, an
effort sent without an explicit budget derives one as that effort's share of
the charged cap (:data:`EFFORT_BUDGET_RATIOS`), bounded strictly below
``units_out`` so the final answer always has room.

A spec may also declare ``default_effort``, the canonical effort the daemon
injects when a payload carries no reasoning control at all — the model's
declared default, replacing the backend's own. A template preset carries one
through the dict form: ``{ preset: thinking_bool, default_effort: medium }``.

Two dialects are not points in that enum space — thinking is a chat-template
switch, and the target is a nested object — so they keep names:
``thinking_bool`` and ``thinking_bool_low_effort``.

Whatever the form, it resolves to a ``param_map`` fragment (see
mapping-reference §4); an inline ``param_map`` merges over it key-by-key.
"""

from __future__ import annotations

OFF = "none"
# The canonical ladder, in ascending intensity. `none` sits outside it as the
# off switch — it is not "less thinking", it is no thinking.
LADDER = ("low", "medium", "high", "xhigh", "max")
CANONICAL_EFFORTS = (OFF, *LADDER)

# Each effort's share of the charged output cap when a thinking budget must be
# derived from it (the backend publishes a budget field but the client sent only
# an effort). Thinking tokens are generated inside `units_out`, so the ratio is
# taken of the cap; it tops out below 1 so the final answer always has room.
EFFORT_BUDGET_RATIOS = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
    "xhigh": 0.95,
    "max": 0.95,
}
_BUDGET_FLOOR = 1024     # below this a thinking budget is not worth sending
_BUDGET_CEILING = 128_000  # gains saturate long before this; nothing above is ever derived


def budget_for_effort(effort: str, units_out) -> int | None:
    """Derive a thinking-token budget from an effort level and the charged cap.

    ``max(min(units_out × ratio, 128000), 1024)``, then bounded strictly below
    ``units_out`` — the budget spends the cap from the inside, and an answer
    must fit in what remains. Returns ``None`` when nothing should be sent:
    ``none`` (off is expressed through the effort mapping, not a zero budget),
    an unknown level, or a cap too small to leave room below it.
    """
    ratio = EFFORT_BUDGET_RATIOS.get(effort)
    if ratio is None or not isinstance(units_out, int):
        return None
    budget = max(min(int(units_out * ratio), _BUDGET_CEILING), _BUDGET_FLOOR)
    budget = min(budget, units_out - 1)
    return budget if budget > 0 else None

# Drop a canonical key outright: no value maps, and nothing passes through.
_DROP = {"values": {}, "unmapped": "drop"}

# Dialects where thinking is a chat-template switch rather than an enum level.
REASONING_PRESETS: dict[str, dict] = {
    # A single boolean: the ladder collapses to on/off.
    "thinking_bool": {
        "reasoning_effort": {
            "to": "chat_template_kwargs",
            "values": {
                OFF: {"enable_thinking": False},
                **{level: {"enable_thinking": True} for level in LADDER},
            },
            "unmapped": "drop",
        },
        "reasoning_max_tokens": _DROP,
    },
    # The same switch plus a documented concise-thinking flag, so the ladder
    # keeps three states: off, brief, full. `low_effort` is stated at every
    # thinking level — leaving it unset would inherit the template default and
    # make `high` indistinguishable from `low`.
    "thinking_bool_low_effort": {
        "reasoning_effort": {
            "to": "chat_template_kwargs",
            "values": {
                OFF: {"enable_thinking": False},
                "low": {"enable_thinking": True, "low_effort": True},
                "medium": {"enable_thinking": True, "low_effort": True},
                "high": {"enable_thinking": True, "low_effort": False},
                "xhigh": {"enable_thinking": True, "low_effort": False},
                "max": {"enable_thinking": True, "low_effort": False},
            },
            "unmapped": "drop",
        },
        "reasoning_max_tokens": _DROP,
    },
}


def effort_map(efforts, budget: str | None = None) -> dict:
    """Build the ``param_map`` fragment for a backend accepting ``efforts``.

    ``efforts`` is the enum the backend's schema publishes, as canonical values
    (order does not matter); ``[]`` means it publishes no reasoning control and
    both canonical params are dropped. ``budget`` names its thinking-cap field,
    or is ``None`` when it has none.
    """
    accepted = _check_efforts(efforts)
    levels = [level for level in LADDER if level in accepted]     # ascending, canonical order

    values: dict[str, str] = {}
    if OFF in accepted:
        values[OFF] = OFF
    elif levels:
        # No disable value: `none` cannot be honored, so it rises to the floor.
        # The client keeps paying for thinking — quote such a model against a
        # reasoning-sized units_out.
        values[OFF] = levels[0]
    for i, level in enumerate(LADDER):
        below = [x for x in levels if LADDER.index(x) <= i]
        if below:
            values[level] = below[-1]      # nearest accepted level not above `level`
        elif levels:
            values[level] = levels[0]      # below the backend's floor: rise to it

    return {
        "reasoning_effort": {"values": values, "unmapped": "drop"} if values else _DROP,
        "reasoning_max_tokens": budget if budget else _DROP,
    }


def extract_default_effort(spec) -> tuple[object, str | None]:
    """Split ``default_effort`` out of a ``reasoning:`` spec.

    Returns ``(spec_without_it, default_effort | None)`` without mutating the
    input — YAML anchors share one spec object across models. The default is
    the canonical effort the daemon injects when a sealed payload carries no
    reasoning control at all, replacing the backend's own (unpublished, often
    expensive) default with a declared one. Any canonical value is a valid
    default. Raises :class:`ValueError` on a non-canonical value or when the
    dialect has no control to apply it to (``efforts: []``).
    """
    if not isinstance(spec, dict) or "default_effort" not in spec:
        return spec, None
    default = spec["default_effort"]
    if default not in CANONICAL_EFFORTS:
        raise ValueError(
            f"reasoning: default_effort {default!r} must be a canonical effort "
            f"({', '.join(CANONICAL_EFFORTS)})"
        )
    rest = {k: v for k, v in spec.items() if k != "default_effort"}
    if rest.get("efforts") == []:
        raise ValueError("reasoning: default_effort needs a reasoning control to apply to ('efforts' is empty)")
    return rest, default


def resolve_reasoning(spec, inline_map: dict | None) -> dict | None:
    """Expand a ``reasoning:`` config value and merge an inline ``param_map`` over it.

    ``spec`` is a mapping (``{efforts, budget}``), the name of a template-switch
    preset, or ``None``. The inline map wins key-by-key, so an operator overrides
    one entry without restating the rest. Raises :class:`ValueError` describing
    the problem when the spec is malformed or names an unknown preset.
    """
    if spec is None:
        return inline_map
    if isinstance(spec, str):
        if spec not in REASONING_PRESETS:
            raise ValueError(
                f"unknown reasoning preset {spec!r} (one of: {', '.join(sorted(REASONING_PRESETS))}), "
                f"or a mapping with 'efforts' and an optional 'budget'"
            )
        base = REASONING_PRESETS[spec]
    elif isinstance(spec, dict) and "preset" in spec:
        # The dict wrapper exists so a template preset can carry a
        # `default_effort` (stripped by `extract_default_effort` before this).
        extra = set(spec) - {"preset"}
        if extra:
            raise ValueError(f"reasoning: 'preset' cannot be combined with {', '.join(sorted(extra))}")
        return resolve_reasoning(spec["preset"], inline_map)
    elif isinstance(spec, dict):
        unknown = set(spec) - {"efforts", "budget"}
        if unknown:
            raise ValueError(f"reasoning: unknown key(s) {', '.join(sorted(unknown))}")
        if "efforts" not in spec:
            raise ValueError("reasoning: a mapping form requires 'efforts'")
        budget = spec.get("budget")
        if budget is not None and not isinstance(budget, str):
            raise ValueError("reasoning: 'budget' must be the backend's cap field name")
        base = effort_map(spec["efforts"], budget)
    else:
        raise ValueError("reasoning: must be a preset name or a mapping with 'efforts'")
    # Copy so no caller ever holds REASONING_PRESETS itself — a mutation there
    # would silently rewrite every later config using the preset.
    return {**base, **inline_map} if inline_map else dict(base)


def _check_efforts(efforts) -> set[str]:
    if not isinstance(efforts, list) or not all(isinstance(e, str) for e in efforts):
        raise ValueError("reasoning: 'efforts' must be a list of canonical effort values")
    unknown = [e for e in efforts if e not in CANONICAL_EFFORTS]
    if unknown:
        raise ValueError(
            f"reasoning: {', '.join(unknown)} is not a canonical effort "
            f"({', '.join(CANONICAL_EFFORTS)}) — declare what the backend accepts in canonical terms"
        )
    return set(efforts)
