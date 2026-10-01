"""What a media request asks for, and what its reference actually is.

Two halves, and the split is the point.

The **derivation** mirrors the client SDKs exactly, because it is the same table
read from the same file: a request names a resolution tier and a whole number of
seconds, and `media-units-v1.json` turns that into the pixels the chain bills.
The client signs those numbers; this re-derives them rather than trusting the
order, so a client that got the arithmetic wrong is a job priced correctly anyway.

The **decoders** are the half no client has. A reference asset declares its own
dimensions and the input leg of the order is priced on that declaration, with
nothing on the wire holding it to the truth. Here the bytes have been decrypted,
so the truth is available: read the header the reference really carries and refuse
a job whose reference outruns the units bought for it.

Header arithmetic only. This daemon has no image or video dependency and must not
grow one — these bytes are attacker-supplied, they arrive inside a container
already claimed and paid for, and a decoder that renders is a decoder that can be
made to allocate. Nothing here allocates proportional to the content, and every
malformed input leaves as a :class:`BackendError` rather than an ``IndexError``
that would take the sweep down instead of the job.
"""

from __future__ import annotations

import math
import re
import struct

from . import media_units as mu
from .errors import BackendError

#: Media types this daemon can measure. An operator may narrow it per backend
#: (``backend.reference.accept``); nothing widens it, because widening means a
#: reader that does not exist.
READABLE_TYPES = ("image/png", "image/jpeg", "image/webp", "video/mp4")

#: Reference sound. Carried, never measured: it has no pixels, the convention
#: counts it zero, and so there is nothing a decoder would protect.
AUDIO_TYPES = ("audio/mpeg", "audio/wav")

#: What a backend may be configured to accept — everything above.
ACCEPTABLE_TYPES = READABLE_TYPES + AUDIO_TYPES


# --- the derivation, shared with both client SDKs -----------------------------


def assets(model_input: dict) -> list[tuple[str, str, object]]:
    """Every reference the request carries, as ``(label, kind, asset)`` in counting
    order: the singular keys, then each list key's elements — the clients' rule.

    ``kind`` is ``"still"``, ``"clip"`` or ``"audio"`` and comes from the *key*,
    never from the asset's own ``media_type``: the key is what the caller meant and
    what the order was priced as, and the type is a claim checked against the bytes.
    """
    found: list[tuple[str, str, object]] = []
    for key in mu.REFERENCE_KEYS:
        if isinstance(model_input.get(key), dict):
            found.append((key, _kind(key), model_input[key]))
    for key in mu.REFERENCE_LIST_KEYS:
        if isinstance(model_input.get(key), list):
            found.extend((f"{key}[{i}]", _kind(key), item)
                         for i, item in enumerate(model_input[key]))
    return found


def _kind(key: str) -> str:
    return "clip" if key in mu.CLIP_KEYS else "audio" if key in mu.AUDIO_KEYS else "still"


def references(model_input: dict) -> list[dict]:
    """The pixel-bearing references, in counting order — what ``auto`` measures."""
    return [asset for _, kind, asset in assets(model_input)
            if kind != "audio" and isinstance(asset, dict)]


def shape(model_input: dict) -> str:
    """``"text"``, ``"image"`` or ``"video"`` — the request's own shape.

    The same single decision both clients make, from the same rule in the same
    table. It is not the model's modality and does not replace it: the catalog
    says what a model *is*, this says what a request *asked for*, and the clamp
    below is where the two meet.
    """
    if any(key in model_input for key in mu.OUTPUT_CEILING_KEYS):
        return "text"
    if "duration" in model_input or "duration_secs" in model_input:
        return "video"
    if (any(key in model_input for key in ("num_images", "width", "resolution"))
            or assets(model_input)):
        return "image"
    return "text"


def resolve_aspect_ratio(declared, assets: list[dict]) -> str:
    """The frame table row this request is shaped by — the clients' rule exactly.

    A ratio the table does not name falls back rather than raising. The client
    already refused it before signing; reaching this with one means the order was
    built by something else, and a provider that cannot price it should decline
    the units rather than crash on the lookup.
    """
    if declared == mu.ADAPTIVE_ASPECT:
        return mu.ADAPTIVE_ASPECT
    if declared is not None and declared != "auto":
        # `isinstance` first: a list is unhashable, and the lookup would raise a
        # TypeError where this promises a fallback.
        return declared if isinstance(declared, str) and declared in mu.FRAMES \
            else mu.AUTO_FALLBACK
    if not assets:
        return mu.AUTO_FALLBACK
    first = assets[0]
    width, height = first.get("width"), first.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        return mu.AUTO_FALLBACK
    target = math.log(width / height)
    return min(
        mu.AUTO_ORDER,
        key=lambda a: (abs(target - math.log(mu.FRAMES[a]["1080p"][0]
                                             / mu.FRAMES[a]["1080p"][1])),
                       mu.AUTO_ORDER.index(a)),
    )


def priced_aspect(model_input: dict) -> str | None:
    """The frame-table row a tiered request is priced on, or ``None`` for one that
    named raw pixels (or nothing) and so is not shaped by a row at all."""
    if "width" in model_input or "height" in model_input:
        return None
    tier = model_input.get("resolution")
    if not isinstance(tier, str) or tier not in mu.RESOLUTIONS:
        return None
    return resolve_aspect_ratio(model_input.get("aspect_ratio"), references(model_input))


def frame_dims(model_input: dict) -> tuple[int, int]:
    """One output frame's pixels, explicit ones winning over a tier."""
    if "width" in model_input or "height" in model_input:
        return (int(model_input.get("width") or mu.DEFAULT_DIM),
                int(model_input.get("height") or mu.DEFAULT_DIM))
    aspect = priced_aspect(model_input)
    if aspect == mu.ADAPTIVE_ASPECT:
        # The model keeps the reference's own shape, which no row names. The
        # tier's largest frame is what the order was priced at — a cap; what is
        # delivered is read off the clip and settles under it.
        return max((row[model_input["resolution"]] for row in mu.FRAMES.values()),
                   key=lambda wh: wh[0] * wh[1])
    if aspect is not None:
        return mu.FRAMES[aspect][model_input["resolution"]]
    return (mu.DEFAULT_DIM, mu.DEFAULT_DIM)


#: A duration written as a string: ASCII digits and nothing else — the clients' rule.
_WHOLE_SECONDS = re.compile(r"[0-9]{1,9}")


def duration_secs(model_input: dict) -> int:
    """How many seconds of output the request buys — ``duration_secs`` then ``duration``.

    The clients' rule exactly, except where they refuse: a spelling that is not a
    duration prices the default here, since raising would crash a claimed job over
    an order no SDK could have built.
    """
    raw = model_input.get("duration_secs")
    if raw is None:                          # null is absent, as everywhere else
        raw = model_input.get("duration")
    if raw == mu.AUTO_DURATION:
        return mu.AUTO_DURATION_S
    if isinstance(raw, str) and _WHOLE_SECONDS.fullmatch(raw):
        seconds = int(raw)
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
        seconds = int(raw)
    else:
        return mu.DEFAULT_DURATION_S
    return seconds if seconds > 0 else mu.DEFAULT_DURATION_S


def auto_duration(model_input: dict) -> bool:
    """Whether the request leaves the clip's length to the model."""
    raw = model_input.get("duration_secs")
    if raw is None:
        raw = model_input.get("duration")
    return raw == mu.AUTO_DURATION


def declared_reference_units(model_input: dict) -> int:
    """``units_in`` as the client's own numbers make it — what the order was priced on."""
    total = 0
    for asset in references(model_input):
        width, height = asset.get("width"), asset.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            continue
        seconds = asset.get("duration_secs")
        seconds = seconds if isinstance(seconds, int) and seconds > 0 else 1
        total += width * height * seconds
    return total


# --- the decoders -------------------------------------------------------------


def _need(raw: bytes, offset: int, length: int, what: str) -> bytes:
    """A slice, or a refusal — never a short read silently padded by Python."""
    if offset < 0 or offset + length > len(raw):
        raise BackendError(f"the reference ends before its {what}")
    return raw[offset:offset + length]


def _png(raw: bytes) -> tuple[int, int]:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise BackendError("the reference is not a PNG")
    # IHDR is required to be the first chunk, so the dimensions sit at a fixed
    # offset: 8 signature + 4 length + 4 type.
    if _need(raw, 12, 4, "IHDR header") != b"IHDR":
        raise BackendError("the reference's first PNG chunk is not IHDR")
    width, height = struct.unpack(">II", _need(raw, 16, 8, "IHDR dimensions"))
    return width, height


def _jpeg(raw: bytes) -> tuple[int, int]:
    if not raw.startswith(b"\xff\xd8"):
        raise BackendError("the reference is not a JPEG")
    # Baseline, extended and progressive are three markers for one thing; the
    # arithmetic ones (C4, C8, CC) are tables and carry no dimensions.
    frame_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    offset = 2
    while offset < len(raw):
        if _need(raw, offset, 1, "next marker")[0] != 0xFF:
            raise BackendError("the reference's JPEG structure is not markers")
        # Any number of `FF` may pad the front of a marker. Taking one for a
        # marker kind reads the next two bytes as a length and jumps wherever the
        # file's author chose — onto a smaller frame header hidden in a comment.
        while _need(raw, offset + 1, 1, "marker kind")[0] == 0xFF:
            offset += 1
        marker = raw[offset + 1]
        offset += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue                      # standalone markers carry no length
        if marker == 0xD9:
            break                         # end of image, and no frame was found
        (length,) = struct.unpack(">H", _need(raw, offset, 2, "segment length"))
        if length < 2:
            raise BackendError("the reference declares a JPEG segment shorter than its own length")
        if marker in frame_markers:
            height, width = struct.unpack(">HH", _need(raw, offset + 3, 4, "frame dimensions"))
            return width, height
        offset += length
    raise BackendError("the reference carries no JPEG start-of-frame")


def _webp(raw: bytes) -> tuple[int, int]:
    if not raw.startswith(b"RIFF") or _need(raw, 8, 4, "WEBP tag") != b"WEBP":
        raise BackendError("the reference is not a WebP")
    kind = _need(raw, 12, 4, "WebP chunk header")
    if kind == b"VP8 ":
        # 3 bytes of frame tag, a 3-byte start code, then two 14-bit dimensions.
        body = _need(raw, 20, 10, "VP8 frame header")
        if body[3:6] != b"\x9d\x01\x2a":
            raise BackendError("the reference's VP8 start code is wrong")
        width, height = struct.unpack("<HH", body[6:10])
        return width & 0x3FFF, height & 0x3FFF
    if kind == b"VP8L":
        body = _need(raw, 20, 5, "VP8L header")
        if body[0] != 0x2F:
            raise BackendError("the reference's VP8L signature is wrong")
        (packed,) = struct.unpack("<I", body[1:5])
        return (packed & 0x3FFF) + 1, ((packed >> 14) & 0x3FFF) + 1
    if kind == b"VP8X":
        body = _need(raw, 20, 10, "VP8X canvas")
        return (int.from_bytes(body[4:7], "little") + 1,
                int.from_bytes(body[7:10], "little") + 1)
    raise BackendError("the reference's WebP chunk is not one this daemon reads")


#: More boxes than any clip this daemon prices could carry — a minute of
#: per-frame fragments is a few thousand. Eight bytes make a box and a reference
#: may run to megabytes, so without a ceiling the walk is a loop of attacker-chosen
#: length on the event loop every other job shares.
_MAX_MP4_BOXES = 65_536

#: What a clip may run past a whole second and still be that many seconds: a
#: couple of frames at 24 fps.
_OVERSHOOT_MS = 100

#: Boxes that only hold other boxes, on the paths to the ones read below.
_MP4_CONTAINERS = (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"mvex")


def _mp4(raw: bytes) -> tuple[int, int, int]:
    """``(width, height, seconds)`` — the largest frame and the longest clock.

    Every header in the file is the author's to write, and the order is priced on
    what this returns, so wherever two boxes could disagree the larger one is
    believed. ``tkhd`` is a display hint a decoder ignores and ``stsd`` is the
    coded size it decodes; ``mvhd`` is the movie's length, ``mdhd`` each track's
    own, and ``mvex/mehd`` a fragmented file's — which legitimately writes zero in
    the other two.

    ``mvhd`` and ``mdhd`` come in a 32-bit and a 64-bit flavour and real files use
    both, so the version byte is read rather than an offset assumed.
    """
    frames: list[tuple[int, int]] = []
    clocks: list[float] = []
    movie_timescale = 0
    fragment_ticks: int | None = None
    fragmented = False
    handler: bytes | None = None
    boxes = 0

    def walk(start: int, end: int, depth: int) -> None:
        nonlocal movie_timescale, fragment_ticks, fragmented, handler, boxes
        if depth > 8:                     # containers nest, but not like this
            raise BackendError("the reference's MP4 boxes nest too deeply")
        offset = start
        while offset + 8 <= end:
            boxes += 1
            if boxes > _MAX_MP4_BOXES:
                raise BackendError(
                    f"the reference carries more than {_MAX_MP4_BOXES} MP4 boxes")
            (size,) = struct.unpack(">I", raw[offset:offset + 4])
            kind = raw[offset + 4:offset + 8]
            body = offset + 8
            if size == 1:                 # 64-bit size in the eight bytes that follow
                (size,) = struct.unpack(">Q", _need(raw, body, 8, "64-bit box size"))
                body += 8
            elif size == 0:               # "to end of file"
                size = end - offset
            # A box must advance the walk. Without this a size of 8 with a 64-bit
            # header, or any size under its own header, spins forever on
            # attacker-chosen bytes.
            if size < (body - offset) or offset + size > end:
                raise BackendError("the reference declares an MP4 box that does not fit")
            if kind in (b"mvhd", b"mdhd"):
                version = _need(raw, body, 1, f"{kind.decode()} version")[0]
                if version == 1:
                    timescale, ticks = struct.unpack(
                        ">IQ", _need(raw, body + 4 + 16, 12, "box timing"))
                    unknown = 0xFFFFFFFFFFFFFFFF
                else:
                    timescale, ticks = struct.unpack(
                        ">II", _need(raw, body + 4 + 8, 8, "box timing"))
                    unknown = 0xFFFFFFFF
                if kind == b"mvhd":
                    movie_timescale = timescale
                if timescale and ticks != unknown:
                    clocks.append(ticks / timescale)
            elif kind == b"mehd":
                version = _need(raw, body, 1, "mehd version")[0]
                fragment_ticks = int.from_bytes(
                    _need(raw, body + 4, 8 if version == 1 else 4, "mehd duration"), "big")
            elif kind == b"moof":
                fragmented = True
            elif kind == b"tkhd":
                version = _need(raw, body, 1, "tkhd version")[0]
                # version, flags, times, track id, reserved, duration — then a
                # fixed 52-byte run of layer/volume/matrix before the dimensions.
                head = 4 + (32 if version == 1 else 20) + 52
                w, h = struct.unpack(">II", _need(raw, body + head, 8, "tkhd dimensions"))
                # 16.16 fixed point; an audio track's display size is zero.
                if w >> 16 and h >> 16:
                    frames.append((w >> 16, h >> 16))
            elif kind == b"hdlr":
                handler = _need(raw, body + 8, 4, "handler type")
            elif kind == b"stsd" and handler in (None, b"vide"):
                # The first sample entry: its own 8-byte header, 8 bytes every
                # entry shares, 16 a visual one reserves, then the coded size. A
                # track that has not said what it is is read as video — the
                # mistake that refuses, never the one that under-prices.
                entry = body + 8
                w, h = struct.unpack(">HH", _need(raw, entry + 32, 4, "coded frame size"))
                if w and h:
                    frames.append((w, h))
            elif kind in _MP4_CONTAINERS:
                if kind == b"trak":
                    handler = None
                walk(body, offset + size, depth + 1)
            offset += size

    if len(raw) < 8 or _need(raw, 4, 4, "MP4 box type") not in (b"ftyp", b"moov"):
        raise BackendError("the reference is not an MP4")
    walk(0, len(raw), 0)
    if fragment_ticks is not None and movie_timescale:
        clocks.append(fragment_ticks / movie_timescale)
    elif fragmented:
        # The length of a fragmented file with no `mehd` is the sum of every
        # fragment's samples. That is a decode, not a header read — and the movie
        # header of such a file covers only what precedes the first fragment.
        raise BackendError("the reference is fragment-encoded and states no overall length")
    if not frames:
        raise BackendError("the reference's MP4 names no track dimensions")
    duration = max(clocks, default=0.0)
    if duration <= 0:
        raise BackendError("the reference's MP4 states no duration")
    width, height = max(frames, key=lambda f: f[0] * f[1])
    # Rounded up, and floored at one: a 1.9-second clip must not be billable as
    # one second, and a 400 ms one cannot be cheaper than a photograph. Less the
    # overshoot first — an encoder asked for four seconds writes 97 frames at 24,
    # which is 4.042, and rounding that up would refuse every honest clip. Exact
    # in milliseconds, so the boundary does not move with float rounding.
    millis = round(duration * 1000) - _OVERSHOOT_MS
    return width, height, max(1, -(-millis // 1000))


def sniff(raw: bytes) -> str | None:
    """The readable media type these bytes open as, by their own magic — or ``None``.

    For output, where the only type on hand is whatever a CDN chose to serve the
    file as. Never for a reference: there the declared type is a claim to check.
    """
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[4:8] in (b"ftyp", b"moov"):
        return "video/mp4"
    return None


def delivered(raw: bytes) -> tuple[str, int, int, int | None] | None:
    """``(media_type, width, height, seconds)`` as a rendered output states them,
    or ``None`` when its header cannot be read — the caller then falls back on what
    was priced, which is all it ever had before."""
    kind = sniff(raw)
    if kind is None:
        return None
    try:
        return (kind, *decode_dimensions(raw, kind))
    except BackendError:
        return None


def decode_dimensions(raw: bytes, media_type: str) -> tuple[int, int, int | None]:
    """``(width, height, seconds)`` as the reference's own bytes state them.

    ``seconds`` is ``None`` for a still. Every failure is a :class:`BackendError`
    naming what could not be read — these bytes are attacker-supplied and the
    caller's job is to fail one job, not to carry an exception it never expected.
    """
    kind = (media_type or "").strip().lower()
    if kind not in READABLE_TYPES:
        raise BackendError(f"this daemon cannot read a reference of type {media_type!r}")
    try:
        if kind == "video/mp4":
            width, height, seconds = _mp4(raw)
        else:
            width, height = {"image/png": _png, "image/jpeg": _jpeg, "image/webp": _webp}[kind](raw)
            seconds = None
    except BackendError:
        raise
    except Exception as exc:  # noqa: BLE001 — every malformed input is one refusal
        raise BackendError(f"the reference could not be read as {kind}: {exc}") from exc
    if width < 1 or height < 1:
        raise BackendError(f"the reference reports a {width}x{height} frame")
    return width, height, seconds
