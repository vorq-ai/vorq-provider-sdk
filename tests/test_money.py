"""The USD ↔ atomic conversion, held to the shared ``money-v1`` vectors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vorqd.money import format_usd, parse_usd

VECTORS = json.loads((Path(__file__).parent / "vectors" / "money-v1.json").read_text())


@pytest.mark.parametrize("case", VECTORS["canonical"], ids=lambda c: f'{c["usd"]}@{c["decimals"]}')
def test_canonical_round_trips(case):
    assert parse_usd(case["usd"], case["decimals"]) == int(case["atomic"])
    assert format_usd(int(case["atomic"]), case["decimals"]) == case["usd"]


@pytest.mark.parametrize("case", VECTORS["parse_only"], ids=lambda c: c["usd"])
def test_parse_accepts_trailing_zeros_and_format_drops_them(case):
    atomic = parse_usd(case["usd"], case["decimals"])
    assert atomic == int(case["atomic"])
    assert format_usd(atomic, case["decimals"]) == case["formats_as"]


@pytest.mark.parametrize("case", VECTORS["refused"], ids=lambda c: c["why"] + ":" + repr(c["usd"]))
def test_refused(case):
    with pytest.raises(ValueError):
        parse_usd(case["usd"], case["decimals"])


@pytest.mark.parametrize("value", [None, 1, 0.5])
def test_only_a_string_is_money(value):
    with pytest.raises(ValueError):
        parse_usd(value, 6)


@pytest.mark.parametrize("value", [-1, True, 1.0])
def test_only_a_non_negative_int_formats(value):
    with pytest.raises(ValueError):
        format_usd(value, 6)
