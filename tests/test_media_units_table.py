"""The media billing table as the daemon reads it.

The table is hand-authored data distributed from the meta-repo by `make media`
and held byte-identical across four repos by `make media-check`. These tests are
the half a single repo can see: that this copy parses, that the numbers it yields
are the ones the daemon used to hard-code, and that the caps it declares actually
bound the convention inside the uint32 an order signs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vorqd import media_units as mu


def test_the_table_ships_beside_the_module_that_reads_it():
    """Packaged data, not a repo-relative path: a wheel that installed the module
    without the table would fail at import on a provider's machine, not here.
    """
    assert (Path(mu.__file__).parent / "media-units-v1.json").is_file()


def test_the_defaults_are_the_numbers_the_daemon_used_to_spell_out():
    # Pinned against the literals deliberately: this is the test that proves the
    # move to shipped data changed no behaviour.
    assert mu.DEFAULT_DIM == 1024
    assert mu.DEFAULT_DURATION_S == 5
    assert mu.DEFAULT_UNITS_OUT == 4096


def test_every_aspect_ratio_names_every_resolution():
    assert set(mu.FRAMES) == set(mu.AUTO_ORDER)
    for aspect, row in mu.FRAMES.items():
        assert set(row) == set(mu.RESOLUTIONS), aspect


@pytest.mark.parametrize("tier", ["480p", "720p", "1080p", "4k"])
def test_every_shape_of_a_tier_is_the_same_pixel_budget(tier):
    """A tier is an area, not a height: the models this interface describes render
    every shape of one tier at about the same number of pixels, and a per-second
    upstream charges the same for all of them. One `rate_out` can only price them
    alike if the table does too — a height-anchored square would buy barely half
    the pixels of the 16:9 frame for the same upstream second.
    """
    anchor = mu.FRAMES["16:9"][tier][0] * mu.FRAMES["16:9"][tier][1]
    for aspect, row in mu.FRAMES.items():
        width, height = row[tier]
        assert abs(width * height / anchor - 1) < 0.015, (aspect, tier)


def test_the_anchor_frame_is_the_one_the_tier_is_named_for():
    assert [mu.FRAMES["16:9"][t] for t in mu.RESOLUTIONS] == [
        (854, 480), (1280, 720), (1920, 1080), (3840, 2160)]


@pytest.mark.parametrize("landscape,portrait", [("16:9", "9:16"), ("4:3", "3:4")])
def test_a_portrait_row_is_the_transpose_it_claims_to_be(landscape, portrait):
    for tier in mu.RESOLUTIONS:
        w, h = mu.FRAMES[landscape][tier]
        assert mu.FRAMES[portrait][tier] == (h, w)


def test_the_caps_keep_the_convention_inside_the_uint32_an_order_signs():
    """`units_in` is a uint32 on the wire and `OrderTerms` refuses an overflow
    rather than wrapping — but refusing is a client-side error the caller has to
    hit to learn about. These caps are what make it unreachable instead: the
    worst case a request can legally declare still fits, with room to spare.
    """
    uint32_max = 2**32 - 1
    clips = 1 + mu.REFERENCE_LIST_KEYS["reference_videos"]
    stills = mu.MAX_REFERENCE_ASSETS - clips
    worst = (clips * mu.MAX_REFERENCE_PIXELS * mu.MAX_REFERENCE_DURATION_S
             + stills * mu.MAX_REFERENCE_PIXELS)
    assert stills > 0
    # With real headroom, not by a byte — a later tier or one more listed clip
    # must not be one edit away from overflowing.
    assert worst * 2 < uint32_max


def test_the_widest_output_a_tier_can_buy_also_fits():
    widest = max(w * h for row in mu.FRAMES.values() for (w, h) in row.values())
    assert widest * mu.MAX_REFERENCE_DURATION_S < 2**32 - 1


def test_the_copy_is_the_master_verbatim_including_its_prose():
    """`make media` copies the file whole. The daemon reads four keys out of it,
    so a copy that silently lost the rest would pass every test above while
    ceasing to be the shared artifact — and `make media-check` diffs the whole
    file, so the two halves of that gate must agree on what "the file" is.
    """
    raw = json.loads((Path(mu.__file__).parent / "media-units-v1.json").read_text())
    assert set(raw) >= {"format", "purpose", "why_a_file", "rules", "defaults",
                        "caps", "resolutions", "auto_order", "frames", "cases"}


# --- the shared cases, as this daemon derives them ------------------------------
#
# Both client suites run these against what they *sign*. This runs them against
# what the daemon *prices*, which is the half that matters when they disagree: a
# client that signs seven seconds for a request this reads as five has escrowed
# for work that will not be done, and one that signs fewer input units than this
# counts has its job handed back after the claim.

import json                                                     # noqa: E402
from pathlib import Path                                        # noqa: E402

import pytest                                                   # noqa: E402

from vorqd import media                                         # noqa: E402

_CASES = json.loads(
    (Path(__file__).parent.parent / "vorqd" / "media-units-v1.json").read_text())["cases"]


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
def test_the_daemon_prices_a_shared_case_as_the_clients_sign_it(case):
    request = case["input"]
    kind = media.shape(request)
    if kind == "text":
        pytest.skip("token-metered: the daemon meters completion tokens, not a derivation")
    assert media.declared_reference_units(request) == case["units_in"], case["why"]
    if "units_out_override" in case:
        return                              # the caller named its own ceiling
    width, height = media.frame_dims(request)
    per_frame = width * height
    count = (media.duration_secs(request) if kind == "video"
             else int(request.get("num_images", 1) or 1))
    assert per_frame * count == case["units_out"], case["why"]
