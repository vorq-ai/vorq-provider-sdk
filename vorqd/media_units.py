"""The media billing table, as shipped data rather than as constants.

``media-units-v1.json`` sits beside this module and is a verbatim copy of the
meta-repo master that both client SDKs also carry. It holds the enum→pixels frame
table, the reference caps, and the three defaults — frame dimension, video
duration, output ceiling — that used to be hand-written here, in both SDKs and in
``scheduler.py`` at once.

Reading them instead of writing them is the same ruling as the body ceiling: a
constant repeated across repos is a constant that drifts, and the last round found
a threshold spelled out in four places where editing one left every suite green.
``make media-check`` regenerates every copy and diffs, so a file that has drifted
is a failing gate and not a surprise on a live chain.

The load happens once at import and is deliberately not defensive: a missing or
malformed table is a broken installation, and a daemon that started anyway would
price media against numbers nobody chose.
"""

from __future__ import annotations

import json
from pathlib import Path

_TABLE = json.loads((Path(__file__).parent / "media-units-v1.json").read_text())

#: The format tag the master carries, so a copy from a future revision is a loud
#: failure at import rather than a quiet disagreement about what a field means.
FORMAT = _TABLE["format"]
if FORMAT != "vorq-media-units-v1":
    raise RuntimeError(f"media-units table is {FORMAT!r}, this build reads 'vorq-media-units-v1'")

#: ``{aspect_ratio: {resolution: (width, height)}}`` in pixels. Read, never computed —
#: there is no rounding rule that lands on 854 in three languages.
FRAMES: dict[str, dict[str, tuple[int, int]]] = {
    aspect: {tier: (int(wh[0]), int(wh[1])) for tier, wh in row.items()}
    for aspect, row in _TABLE["frames"].items()
}

#: The aspect ratios ``auto`` chooses between, in the order that breaks ties.
AUTO_ORDER: tuple[str, ...] = tuple(_TABLE["auto_order"])

#: What ``auto`` chooses when there is no reference to measure. A text-to-video
#: request is a legitimate shape, so this has to be a number and never an error.
AUTO_FALLBACK: str = _TABLE["auto_fallback"]

#: The reference assets a request may carry, in the order they are counted.
REFERENCE_KEYS: tuple[str, ...] = tuple(_TABLE["reference_keys"])

#: The three spellings of an output-token ceiling, in precedence order. A request
#: naming any of them is token-metered on **both** sides.
OUTPUT_CEILING_KEYS: tuple[str, ...] = tuple(_TABLE["output_ceiling_keys"])

#: The resolution tiers a request may name.
RESOLUTIONS: tuple[str, ...] = tuple(_TABLE["resolutions"])

#: Each output frame's dimension when the request names neither pixels nor a tier.
DEFAULT_DIM: int = int(_TABLE["defaults"]["dim"])

#: A video request that names no length runs this many seconds; the same value
#: prices the job and labels the frame it returns.
DEFAULT_DURATION_S: int = int(_TABLE["defaults"]["duration_secs"])

#: The output ceiling a request with nothing to size it by is quoted at.
DEFAULT_UNITS_OUT: int = int(_TABLE["defaults"]["units_out"])

#: The most pixels one reference asset may declare.
MAX_REFERENCE_PIXELS: int = int(_TABLE["caps"]["reference_pixels"])

#: How many reference assets one request may carry (pixel-bearing ones, singular and listed together).
MAX_REFERENCE_ASSETS: int = int(_TABLE["caps"]["reference_assets"])

#: The longest reference clip, in seconds.
MAX_REFERENCE_DURATION_S: int = int(_TABLE["caps"]["reference_duration_s"])

#: The most reference audio one element may carry, in decoded bytes. Sound counts
#: zero units — nothing meters it — so this and the list length are its only bound.
MAX_REFERENCE_AUDIO_BYTES: int = int(_TABLE["caps"]["reference_audio_bytes"])

#: The list-valued reference keys, each with the most elements it may hold.
REFERENCE_LIST_KEYS: dict[str, int] = {k: int(v) for k, v in _TABLE["reference_list_keys"].items()}

#: Keys whose assets are clips (a ``duration_secs`` is required) and keys whose
#: assets are sound (no pixels at all). Everything else is a still.
CLIP_KEYS: frozenset[str] = frozenset(_TABLE["clip_keys"])
AUDIO_KEYS: frozenset[str] = frozenset(_TABLE["audio_keys"])

#: The ``aspect_ratio`` that asks the model to keep the reference's own shape. No
#: row names that shape, so it is priced at the tier's largest frame — a cap.
ADAPTIVE_ASPECT: str = _TABLE["adaptive_aspect"]

#: The ``duration`` that asks the model to choose the length, and the seconds it
#: is priced at — a cap the delivered clip settles under.
AUTO_DURATION: str = _TABLE["auto_duration"]
AUTO_DURATION_S: int = int(_TABLE["defaults"]["auto_duration_secs"])
