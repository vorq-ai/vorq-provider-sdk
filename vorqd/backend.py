"""The backend seam: execute one job against the operator's inference runtime.

Driven entirely by the YAML mapping. A preset (``openai-chat``,
``openai-embeddings``, ``openai-responses`` and ``openai-batch`` ship) expands to
the same request/response shape a raw mapping declares, so one execution path
serves both. Text jobs yield a text string plus ``completion_tokens``; media
jobs yield the output URLs or decoded base64 blobs the scheduler seals into the
result, along with the dimensions the job was priced against.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from . import media, media_units
from ._templates import extract, parse_path, render, resolves
from .config import BackendConfig
from .errors import BackendError, MediaInputRefused
from .limits import parse_retry_after, window_seconds
from .reasoning import budget_for_effort

log = logging.getLogger("vorqd")

# Each output frame's dimensions default to this when the request omits width/height.
# Media is metered in raw pixels — image units_out = num_images × width × height, video
# units_out = width × height × duration_secs — so the provider prices, clamps, and settles
# from the plaintext dimensions without decoding the produced pixels.
#
# Re-exported from the shipped table rather than written here: the client SDKs
# size their side of the same order from the same two numbers, and the party that
# signs the units and the party that prices them cannot afford to disagree. See
# `media_units.py`, and `make media-check` for what keeps the copies honest.
DEFAULT_DIM = media_units.DEFAULT_DIM

# A video request that names no length runs this many seconds; the same value
# prices the job and labels the frame it returns.
DEFAULT_DURATION_S = media_units.DEFAULT_DURATION_S

# The parameter vocabulary the openai-chat preset forwards: the standard
# chat-completions request fields that compose with VORQ's billing contract,
# plus the network's canonical reasoning controls (mapped into the backend's
# dialect by `param_map` / a `reasoning:` preset — see reasoning.py).
# This set is the default for a backend that publishes no schema of its own.
# `params_supported` REPLACES it with the exact fields one served model accepts
# — including any nonstandard knob of its own; `param_map` renames or
# value-maps a forwarded key to the backend's spelling.
OPENAI_CHAT_PARAMS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
    "max_tokens", "min_tokens", "seed", "stop",
    "reasoning_effort", "reasoning_max_tokens",
    "frequency_penalty", "presence_penalty", "logit_bias",
    "logprobs", "top_logprobs",
    "response_format", "tools", "tool_choice", "parallel_tool_calls",
})

# The Responses-surface vocabulary: what `openai-responses` forwards when the
# entry declares no `params_supported`. The chat-only sampling knobs
# (`top_k`, `min_p`, `seed`, logprobs, ...) are absent because the surface has
# no such fields — and a gateway that drops an unknown field silently would
# turn each into a control the client believes it set. `reasoning_max_tokens`
# is absent for the same reason: the surface spells the thinking budget as an
# effort, and a backend that does publish a budget field gets it the explicit
# way — named in `params_supported`, spelled by `reasoning:` / `param_map`.
OPENAI_RESPONSES_PARAMS = frozenset({
    "temperature", "top_p", "max_tokens", "stop",
    "frequency_penalty", "presence_penalty", "reasoning_effort",
    "response_format", "tools", "tool_choice", "parallel_tool_calls",
})


def _flat_tool(tool):
    """A chat-shaped function tool (`{type, function: {...}}`) as the Responses
    shape (`{type, name, description, parameters, ...}`); anything else verbatim."""
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        return {**{k: v for k, v in tool.items() if k != "function"}, **tool["function"]}
    return tool


def _responses_dialect(body: dict) -> dict:
    """Respell, in place, every chat-completion key whose Responses spelling differs.

    `messages` → `input`; `max_tokens` → `max_output_tokens`; `reasoning_effort`
    → `reasoning.effort` (merged into an existing `reasoning` object);
    `response_format` → `text.format`, with a `json_schema` wrapper flattened;
    function tools and a function `tool_choice` lose their `function` nesting.
    Keys already in the Responses spelling pass through untouched.
    """
    if "messages" in body:
        body["input"] = body.pop("messages")
    if "max_tokens" in body:
        body["max_output_tokens"] = body.pop("max_tokens")
    if "reasoning_effort" in body:
        effort = body.pop("reasoning_effort")
        current = body.get("reasoning")
        body["reasoning"] = {**(current if isinstance(current, dict) else {}), "effort": effort}
    if "response_format" in body:
        fmt = body.pop("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema" \
                and isinstance(fmt.get("json_schema"), dict):
            fmt = {"type": "json_schema", **fmt["json_schema"]}
        current = body.get("text")
        body["text"] = {**(current if isinstance(current, dict) else {}), "format": fmt}
    if isinstance(body.get("tools"), list):
        body["tools"] = [_flat_tool(t) for t in body["tools"]]
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        body["tool_choice"] = {**{k: v for k, v in choice.items() if k != "function"},
                               **choice["function"]}
    return body


# Valid OpenAI params that never reach a backend, whatever the config says.
#
# Billing: the preset bills on a single-choice, non-streamed, usage-bearing
# response at the priced upstream tier — `best_of` bills the backend for every
# generated sequence, `service_tier` upgrades the upstream price at an unchanged
# VORQ rate, `stream` on its own removes the usage block the settled count comes
# from (the operator's `backend.stream` asks for the block back on the final
# chunk and reassembles the completion — see `_request_stream`).
#
# Confidentiality: `user` and `metadata` are caller-chosen identifiers. The
# network seals a job's payload so a provider learns nothing about who is behind
# it until it claims; forwarding a stable end-user id to a third-party backend
# would undo that, linking one client's jobs across providers. The catalog
# forbids both, so a client never legitimately sends one — this is the backstop.
NEVER_FORWARDED = frozenset({
    "n", "best_of", "stream", "stream_options", "service_tier", "user", "metadata",
})

# The two sides an asymmetric retrieval model embeds into. A caller may name one
# only where the operator opened the field (`input_type_overridable`), and only
# from this set: anything else is dropped in favour of the operator's own side,
# because a sealed payload is untrusted and a rejected request is a job the
# provider already claimed and now fails at its own cost.
EMBEDDING_INPUT_TYPES = frozenset({"query", "passage"})


def _set_path(body: dict, path: str, value) -> None:
    """Write `value` at a (possibly dotted) key path, merging object values into
    an existing dict target so several params can compose one backend object
    (e.g. reasoning effort and budget both writing into chat_template_kwargs)."""
    target, _, sub = path.partition(".")
    if sub:
        body.setdefault(target, {})[sub] = value
    elif isinstance(value, dict) and isinstance(body.get(target), dict):
        body[target].update(value)
    elif isinstance(value, dict):
        body[target] = dict(value)  # copy: a preset's mapped object must never be mutated in place
    else:
        body[target] = value


def _apply_param_map(body: dict, key: str, value, spec) -> bool:
    """Place one forwarded param into `body` under its ``param_map`` entry.

    ``spec`` is the entry for `key`: absent → forward verbatim; a string → a
    rename (dotted for nesting); an object → ``{to, values, unmapped}`` where
    `values` maps the client's exact value to the backend's (a `None` mapping
    drops it) and `unmapped` says what an unlisted value does (`pass`, the
    default, forwards it; `drop` omits it). Returns False when the param was
    dropped so the caller can log it.
    """
    if spec is None or isinstance(spec, str):
        _set_path(body, spec or key, value)
        return True
    values = spec.get("values") or {}
    if isinstance(value, str) and value in values:
        mapped = values[value]
        if mapped is None:
            return False
        _set_path(body, spec.get("to") or key, mapped)
        return True
    if spec.get("unmapped", "pass") == "drop":
        return False
    _set_path(body, spec.get("to") or key, value)
    return True


def _frame_dims(input: dict) -> tuple[int, int]:
    """The output frame's pixel dimensions, from the request that priced it.

    Delegated so there is one reading of a request's shape in this daemon and it
    is the one the client SDKs share. By the time the frames are labelled, the
    request has been through `plan_media_units` and carries explicit pixels — but
    a second copy of the fallback rules here is a second thing to drift.
    """
    return media.frame_dims(input)


def _frame_pixels(input: dict) -> int:
    w, h = _frame_dims(input)
    return w * h


def _duration_secs(input: dict) -> int:
    return media.duration_secs(input)


def _as_int(value) -> int | None:
    """A JSON scalar as an int, or ``None`` when it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value) -> str | None:
    """A JSON scalar as a non-empty string, or ``None`` when it is not one.

    A wildcard JSONPath (``$.images[*].content_type``) evaluates to a *list*, and
    the frame the client opens carries one media type as a string. Anything that
    is not that string collapses here, so a mapping written with a wildcard falls
    back to the served type rather than sealing a list into the result.
    """
    return value.strip() if isinstance(value, str) and value.strip() else None


def _sla_s(window) -> float | None:
    """The job's SLA window in seconds, or ``None`` when it cannot be read."""
    try:
        return float(window_seconds(str(window)))
    except ValueError:
        return None


#: The longest `custom_id` the batch surface accepts.
_BATCH_CUSTOM_ID_MAX = 64
#: Wall on the best-effort input-file delete after a batch is collected.
_BATCH_DISCARD_TIMEOUT_S = 15.0


def _batch_custom_id(job) -> str:
    """The job id as the line's `custom_id`: the hex without its `0x`, capped at
    64 characters. Derived, never stored, so a resumed job recomputes the same one."""
    jid = str(getattr(job, "job_id", "") or "")
    if jid.startswith("0x"):
        jid = jid[2:]
    return jid[:_BATCH_CUSTOM_ID_MAX]


def _media_items(value) -> list:
    """A media path's value as a list of items.

    A wildcard path yields a list already; a scalar path naming a single output
    yields the item itself, which is one output rather than a sequence of its
    characters. Nothing extracted is an empty render, which the caller answers.
    """
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _media_urls(value) -> list[str]:
    """The mapping's ``media_urls`` value as fetchable absolute URLs.

    A JSONPath resolves against whatever the backend's JSON happens to hold, so
    the shape is checked here, at the one place the list is built — a value that
    is not a URL fails the job as a backend error, which refunds the client now,
    rather than surfacing as an unhandled error deep in the fetch loop with the
    escrow left to expire.

    The offending value is never quoted into the message: it becomes the reason
    on a failure report the network records, and a media URL can carry a signed
    token and names the operator's own runtime.
    """
    urls = []
    for item in _media_items(value):
        if not isinstance(item, str):
            raise BackendError(
                f"response.result.media_urls resolved to a {type(item).__name__}, not a URL"
            )
        parts = urlsplit(item)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise BackendError("response.result.media_urls resolved to a value that is not "
                               "an absolute http(s) URL")
        urls.append(item)
    return urls


def _media_blobs(value) -> list[bytes]:
    """The mapping's ``media_b64`` value as decoded frames, with the same guard."""
    blobs = []
    for item in _media_items(value):
        if not isinstance(item, str):
            raise BackendError(
                f"response.result.media_b64 resolved to a {type(item).__name__}, not base64"
            )
        try:
            blob = base64.b64decode(item)
        except ValueError as exc:   # binascii.Error is one of these
            raise BackendError("response.result.media_b64 resolved to a value that is "
                               "not valid base64") from exc
        if not blob:
            # b64decode drops characters outside the alphabet, so a value made
            # entirely of them decodes to nothing at all. An empty frame is not a
            # rendered output — fail the job rather than seal a zero-byte image.
            raise BackendError("response.result.media_b64 resolved to a value that "
                               "decodes to no bytes")
        blobs.append(blob)
    return blobs


def plan_media_units(job, input: dict, caps: dict | None = None, *,
                     accept=None, durations=None, bounds=None, resolutions=None,
                     auto_duration=None, adaptive_aspect=None) -> tuple[dict, int | None]:
    """Clamp a media request's cost driver to the charged ``units_out`` cap and return
    ``(request_to_run, planned_units)``.

    ``planned_units`` is what the clamped request would produce if the backend renders
    it in full — a ceiling, not a bill. The settled count is taken from the frames the
    backend actually returned (``Scheduler._build_result``), so a short render is
    charged short.

    ``units_out`` is a ceiling for every modality: the provider renders at most what the
    client paid for and settles no more than that (billed ≤ cap). The unit is exact
    output pixels — ``num_images × width × height`` for image, ``width × height ×
    duration_secs`` for video — so a per-image, per-megapixel, or per-second backend is
    all priced by a per-pixel ``rate_out``. A small over-ask is clamped down — ``num_images``
    for image, ``duration_secs`` for video, each frame kept at the requested resolution —
    rather than failing. It raises :class:`BackendError` only when even one minimum unit (a
    single image at the requested size, or one second at that resolution) exceeds the cap.
    Text jobs are metered by ``completion_tokens`` and return ``None``.

    ``caps`` (the backend's ``param_caps``) bounds compute drivers the pixel unit
    does not price — ``steps``, ``fps`` — clamping each present param down to the
    operator's ceiling. The billed unit is untouched: these params cost the
    provider compute, not the client money.

    ``durations`` (the backend's ``durations``) is the list of clip lengths the
    backend renders at all, for one that takes its length from a short list rather
    than any whole number. It is applied *after* the cap, and downward only: the
    greatest listed length the order covers. None at all is a job this backend
    cannot serve.

    ``resolutions`` is the tiers the backend renders. A request for another — or
    one in raw pixels, which such a backend would render at its own default while
    the job was billed at the caller's size — is handed back.

    ``auto_duration`` and ``adaptive_aspect`` are the backend's own spellings for
    "the model chooses the length" and "the model keeps the reference's shape".
    A backend that names neither cannot serve those requests. Both are priced at a
    cap — the longest clip, the tier's largest frame — because nothing here can
    clamp a choice that has not been made yet; what is delivered is what settles.
    """
    for key, ceiling in (caps or {}).items():
        if key in input and isinstance(input[key], (int, float)) and input[key] > ceiling:
            input = {**input, key: ceiling}

    # The input side first, and before any clamp: an under-declared reference is
    # refused outright, so there is no point sizing an output for a job that is
    # about to be handed back.
    check_references(job, input, accept=accept, bounds=bounds)

    # Then the request's own shape, re-derived from the shared table rather than
    # read off whatever the client happened to write. The derived pixels and
    # seconds are merged into the request this daemon RUNS — never into the sealed
    # payload, which stays the caller's bytes — so an operator's
    # `width: "{input.width}"` resolves against a request that named only a tier,
    # and one backend config serves both spellings.
    if resolutions is not None and input.get("resolution") not in resolutions:
        raise BackendError(
            f"this backend renders the resolution tiers {', '.join(resolutions)}, and the "
            f"request names {'none' if input.get('resolution') is None else 'another'}")
    aspect = media.priced_aspect(input)
    if aspect == media_units.ADAPTIVE_ASPECT:
        if adaptive_aspect is None:
            raise BackendError("this backend does not render an adaptive aspect_ratio")
    width, height = media.frame_dims(input)
    input = {**input, "width": width, "height": height}
    if aspect is not None:
        # The row the frame came from, in the table's own spelling: `auto` is this
        # network's word, and a backend handed it — or nothing — renders a shape
        # of its own choosing rather than the one the order is billed on.
        input["aspect_ratio"] = adaptive_aspect if aspect == media_units.ADAPTIVE_ASPECT else aspect
    px = width * height
    cap = getattr(job, "units_out", None)

    if job.modality == "image":
        n_req = int(input.get("num_images", 1) or 1)
        if cap is None:
            return input, n_req * px
        n = min(n_req, int(cap // px))
        if n < 1:
            raise BackendError(f"charged units_out {cap} does not cover one {px}-pixel image")
        return ({**input, "num_images": n} if n != n_req else input), n * px
    if job.modality == "video" and media.auto_duration(input):
        if auto_duration is None:
            raise BackendError("this backend does not choose a clip's length (duration 'auto')")
        longest = max(durations) if durations else media_units.AUTO_DURATION_S
        if cap is not None and cap < px * longest:
            raise BackendError(
                f"charged units_out {cap} does not cover the {longest}s this backend may "
                f"choose at {px} pixels/frame")
        input = {**input, "duration_secs": auto_duration}
        if "duration" in input:
            input["duration"] = auto_duration
        return input, px * longest
    if job.modality == "video":
        secs = media.duration_secs(input)
        if cap is not None:
            secs = min(secs, int(cap // px))
            if secs < 1:
                raise BackendError(
                    f"charged units_out {cap} does not cover one second at {px} pixels/frame")
        if durations:
            served = [int(d) for d in durations if int(d) <= secs]
            if not served:
                raise BackendError(
                    f"this backend renders {', '.join(str(d) for d in sorted(durations))} "
                    f"second clips and the order covers {secs}")
            secs = max(served)
        input = {**input, "duration_secs": secs}
        if "duration" in input:
            # The caller's own spelling follows the clamp, in the type it was
            # written in: a template reading `{input.duration}` must not render
            # the seconds the clamp just took away.
            input["duration"] = str(secs) if isinstance(input["duration"], str) else secs
        return input, px * secs
    return input, None


#: The bounds a backend may put on the references it takes, per kind of asset.
REFERENCE_BOUNDS = ("min_side", "max_side", "min_ratio", "max_ratio", "min_pixels", "max_pixels",
                    "min_secs", "max_secs", "max_total_secs", "max_bytes", "max_count")


def check_references(job, input: dict, *, accept=None, bounds: dict | None = None) -> None:
    """Hold each reference to what the order paid for it, or refuse the job.

    The client declares a reference's dimensions and the chain bills
    ``rate_in × units_in`` against that declaration, unconditionally. Here the
    bytes are decrypted, so the declaration is checkable — and this is the only
    place it ever is.

    **There is no clamp on this side.** ``units_out`` can be trimmed because a
    provider may deliver less; input cannot, because it has already been sent and
    the input leg has already been priced. So every divergence resolves to exactly
    one of two answers: the client bought more than it sent, which runs, or it
    bought less, which is handed back.

    Handed back is cheap on purpose — inside the chain's 300-second grace window a
    ``fail`` refunds the client in full and costs the provider no reputation, and
    :class:`MediaInputRefused` is non-retryable and not the backend's fault, so
    the circuit breaker never sees it.

    ``accept`` is the operator's own list of media types for this backend. It
    narrows :data:`vorqd.media.ACCEPTABLE_TYPES` and never widens it. ``bounds``
    is ``{kind: {bound: value}}`` for the kinds ``still``, ``clip`` and ``audio``
    (:data:`REFERENCE_BOUNDS`): the network's caps are wide and a backend's are its
    own, and a reference outside them is found here for nothing rather than by the
    backend after an upload.

    Sound is carried but never measured. It has no pixels and counts zero, so the
    *key* decides that an asset is sound and the type must agree with the key —
    otherwise a picture listed as sound would ride past the meter.
    """
    found = media.assets(input)
    if not found:
        return
    for key, most in media_units.REFERENCE_LIST_KEYS.items():
        held = input.get(key)
        if isinstance(held, list) and len(held) > most:
            raise MediaInputRefused(f"{key} holds {len(held)} references; at most {most} are priced")
    if sum(1 for _, kind, _ in found if kind != "audio") > media_units.MAX_REFERENCE_ASSETS:
        raise MediaInputRefused(
            f"the request carries more than the {media_units.MAX_REFERENCE_ASSETS} reference "
            f"assets that are priced (reference_assets)"
        )
    allowed = ({str(a).strip().lower() for a in accept} if accept is not None
               else set(media.ACCEPTABLE_TYPES))
    bounds = bounds or {}
    counts: dict[str, int] = {}
    clip_secs = 0
    true_total = 0

    for key, kind, asset in found:
        if not isinstance(asset, dict):
            raise MediaInputRefused(f"{key} is not a reference object")
        limits = bounds.get(kind) or {}
        counts[kind] = counts.get(kind, 0) + 1
        if "max_count" in limits and counts[kind] > limits["max_count"]:
            raise MediaInputRefused(
                f"{key}: this backend takes at most {limits['max_count']} {kind} "
                f"reference(s) (max_count)")

        media_type = str(asset.get("media_type") or "").strip().lower()[:64]
        is_sound = media_type in media.AUDIO_TYPES
        if media_type not in allowed or is_sound != (kind == "audio"):
            raise MediaInputRefused(
                f"{key} is {media_type or 'untyped'}, which this backend does not accept"
                + ("" if media_type not in allowed else f" as a {kind} reference"))
        try:
            raw = base64.b64decode(str(asset.get("b64") or ""), validate=True)
        except Exception as exc:  # noqa: BLE001 — any decode failure is one refusal
            raise MediaInputRefused(f"{key} is not valid base64: {exc}") from exc
        if "max_bytes" in limits and len(raw) > limits["max_bytes"]:
            raise MediaInputRefused(
                f"{key} is {len(raw)} bytes; this backend takes {limits['max_bytes']} (max_bytes)")

        if kind == "audio":
            if len(raw) > media_units.MAX_REFERENCE_AUDIO_BYTES:
                raise MediaInputRefused(
                    f"{key} is {len(raw)} bytes; the most a reference sound may be is "
                    f"{media_units.MAX_REFERENCE_AUDIO_BYTES} (reference_audio_bytes)")
            continue

        try:
            width, height, seconds = media.decode_dimensions(raw, media_type)
        except BackendError as exc:
            # Re-raised under this class so the outcome is metered as the client's
            # doing rather than as a backend that misbehaved.
            raise MediaInputRefused(f"{key}: {exc}") from exc
        if (seconds is not None) != (kind == "clip"):
            raise MediaInputRefused(f"{key} is {media_type}, which is not a {kind}")

        pixels = width * height
        if pixels > media_units.MAX_REFERENCE_PIXELS:
            raise MediaInputRefused(
                f"{key} is {width}x{height} = {pixels} pixels; the most a reference "
                f"may carry is {media_units.MAX_REFERENCE_PIXELS} (reference_pixels)"
            )
        if seconds is not None and seconds > media_units.MAX_REFERENCE_DURATION_S:
            raise MediaInputRefused(
                f"{key} runs {seconds}s; the longest reference clip is "
                f"{media_units.MAX_REFERENCE_DURATION_S}s (reference_duration_s)"
            )
        _within_bounds(key, limits, width, height, seconds)
        clip_secs += seconds or 0
        if "max_total_secs" in limits and clip_secs > limits["max_total_secs"]:
            raise MediaInputRefused(
                f"{key}: the reference clips run {clip_secs}s together; this backend takes "
                f"{limits['max_total_secs']}s (max_total_secs)")

        declared_w, declared_h = asset.get("width"), asset.get("height")
        declared_secs = asset.get("duration_secs")
        declared_secs = declared_secs if isinstance(declared_secs, int) and declared_secs > 0 else 1
        if not isinstance(declared_w, int) or not isinstance(declared_h, int):
            raise MediaInputRefused(f"{key} declares no usable dimensions")
        # Area, not each side: the order bills the product, and a header states a
        # frame before its orientation tag or rotation matrix is applied — so an
        # honest client that measured the picture the right way up declares the
        # transpose of what is read here.
        if pixels > declared_w * declared_h or (seconds or 1) > declared_secs:
            raise MediaInputRefused(
                f"{key} is {width}x{height}"
                + (f" for {seconds}s" if seconds is not None else "")
                + f", larger than the {declared_w}x{declared_h}"
                + (f" for {declared_secs}s" if seconds is not None else "")
                + " the order paid for"
            )
        true_total += pixels * (seconds or 1)

    paid = getattr(job, "units_in", None)
    if paid is not None and true_total > int(paid):
        # Reachable even with every asset inside its own declaration, when the
        # order's units_in was simply written smaller than the assets it names.
        raise MediaInputRefused(
            f"the references total {true_total} pixel-seconds and the order declared "
            f"{paid} input units"
        )


def _within_bounds(key: str, limits: dict, width: int, height: int, seconds: int | None) -> None:
    """One decoded asset against one backend's own limits, naming the first it breaks."""
    facts = {
        "min_side": (min(width, height), False), "max_side": (max(width, height), True),
        "min_ratio": (width / height, False), "max_ratio": (width / height, True),
        "min_pixels": (width * height, False), "max_pixels": (width * height, True),
    }
    if seconds is not None:
        facts.update(min_secs=(seconds, False), max_secs=(seconds, True))
    for bound, (value, is_ceiling) in facts.items():
        if bound in limits and (value > limits[bound] if is_ceiling else value < limits[bound]):
            raise MediaInputRefused(
                f"{key} is {width}x{height}" + (f" for {seconds}s" if seconds is not None else "")
                + f"; this backend takes {bound} {limits[bound]}")


#: Everything in a sealed envelope that is not the model input, generously.
#:
#: The client seals ``{v, owner, result_key, input}`` plus an optional
#: ``custom_id`` as canonical JSON. The fixed frame is 161 bytes — two braces,
#: four keys, an address, a result key and a version string — and ``custom_id`` is
#: an opaque caller label with no ceiling anywhere in the protocol. This is the
#: frame plus a generous allowance for that label, and it is subtracted from every
#: measurement, so the daemon's estimate of the input's own size can only come out
#: **low**. A floor that under-estimates never refuses an honest client, which is
#: the only failure mode that costs a provider work it wanted.
ENVELOPE_SLACK_BYTES = 4096

#: The modalities whose ``units_in`` is denominated in bytes of input.
#:
#: Text and embeddings are metered on the prompt, and the shipped clients declare
#: one unit per four bytes of it — which is the whole reason a byte measurement
#: can stand in as a floor. Image and video meter the *reference* in pixels, a
#: quantity with no relation to payload size: a sharper reference of identical
#: dimensions costs more bytes and buys exactly the same work. Weighing those
#: would decline honest media for carrying its own input, so they are held to
#: their declaration by re-deriving the reference after decryption instead, where
#: its real dimensions are knowable.
BYTE_METERED_MODALITIES = frozenset({"text", "embedding"})


def input_shortfall(job, payload_bytes: int, *, bytes_per_unit: int, modality: str | None) -> int:
    """Bytes of sealed payload past what this order's ``units_in`` pays for; 0 when covered.

    ``units_in`` is declared by the client and the chain bills the input leg at
    whatever it says — ``charge = min(cap, ceilDiv(rateIn·unitsIn + rateOut·tok,
    RATE_SCALE))`` — with no floor under it anywhere. On an ``embedding`` model,
    where settlement reports ``completion_tok = 0``, ``rateIn·unitsIn`` is the
    **entire** bill: one declared unit buys a megabyte of prompt for one atomic
    unit of value. And unlike ``units_out``, there is nothing to clamp — a
    provider cannot deliver less input than it was sent, so the only move against
    an under-declared bid is to decline it.

    **This deliberately does not reproduce what the clients compute.** They declare
    one unit per four bytes of canonical JSON; this daemon owns no canonical-JSON
    writer and must not grow one, because a third copy of that formula is a third
    thing to drift out of step. What it owns instead is a measurement it can take
    without a key — sealed bytes, from a container's own length before the claim
    and from the plaintext after it — and an operator's ceiling on how many of
    them one paid unit may buy. At the shipped default that ceiling is four times
    looser than the clients' own rate, and the slack is subtracted on top; a
    multiplicative tolerance that wide is what makes every additive unknown
    harmless, so the floor catches the absurd and never argues about the marginal.

    Zero on a bid whose window meters no input side: ``rate_in`` is unset there,
    the input leg bills nothing whatever ``units_in`` says, and judging it would
    decline honest media work over a number nobody is charged for. Zero too on a
    row carrying no ``units_in`` at all — a signed order term the coordinator has
    no reason to drop, so its absence means something changed upstream, and the
    worst possible answer to that is a daemon that quietly stops claiming.

    Zero as well on any modality whose input is not counted in bytes — see
    :data:`BYTE_METERED_MODALITIES`. Media declares reference pixel-seconds, which
    a payload's weight says nothing about.

    ``modality`` is a **parameter and not read off the job**, which is the one
    surprising thing here. ``EvmJob.modality`` is a placeholder until ``_ingest``
    stamps it after the claim, so a media job reads as ``"text"`` for the whole of
    the pre-claim sweep — exactly where this is most useful and where getting it
    wrong declines honest work. The caller knows: before the claim it is the model
    the sweep is polling for, after it the job's own stamped value.
    """
    if modality not in BYTE_METERED_MODALITIES:
        return 0
    if not job.rate_in:
        return 0
    declared = getattr(job, "units_in", None)
    if declared is None or bytes_per_unit <= 0:
        return 0
    paid_for = int(declared) * int(bytes_per_unit)
    return max(0, int(payload_bytes) - ENVELOPE_SLACK_BYTES - paid_for)


def _readable_list(value) -> list:
    """The list a Responses-style item promises, or the fail-closed error.

    A missing field is an empty one; a scalar where a list belongs is an object
    this cannot read, and guessing at it would seal a silently empty answer the
    client still pays for.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise BackendError("the backend answered with an unreadable output")
    return value


def responses_to_chat(final: dict) -> dict:
    """A finished Responses-style object as the chat-completion object a client reads.

    The answer is the concatenated ``output_text`` parts of the ``message``
    items; reasoning rides beside it as ``reasoning_content`` — the summary when
    the surface emits one, else the raw ``reasoning_text`` — exactly as a
    streamed chat completion carries it, and ``function_call`` items come back
    as chat ``tool_calls`` (with object arguments serialised, since the chat
    shape carries them as a string). The usage block is renamed field for
    field; without an output count there is no usage block, and extraction then
    reports no billable quantity.
    """
    texts: list[str] = []
    reasoning: list[str] = []
    reasoning_text: list[str] = []
    tool_calls: list[dict] = []
    for item in _readable_list(final.get("output")):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in _readable_list(item.get("content")):
                if isinstance(part, dict) and part.get("type") == "output_text" \
                        and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        elif item.get("type") == "reasoning":
            for part in _readable_list(item.get("summary")):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    reasoning.append(part["text"])
            for part in _readable_list(item.get("content")):
                if isinstance(part, dict) and part.get("type") == "reasoning_text" \
                        and isinstance(part.get("text"), str):
                    reasoning_text.append(part["text"])
        elif item.get("type") == "function_call":
            args = item.get("arguments")
            call: dict = {
                "type": "function",
                "function": {"name": item.get("name"),
                             "arguments": args if isinstance(args, str) else json.dumps(args)},
            }
            handle = item.get("call_id") or item.get("id")
            if handle is not None:
                call = {"id": handle, **call}
            tool_calls.append(call)
    message: dict = {"role": "assistant", "content": "".join(texts)}
    thoughts = reasoning or reasoning_text
    if thoughts:
        message["reasoning_content"] = "".join(thoughts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    out: dict = {
        "id": final.get("id"),
        "object": "chat.completion",
        "created": final.get("created_at"),
        "model": final.get("model"),
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
    }
    usage = final.get("usage") or {}
    if isinstance(usage, dict) and "output_tokens" in usage:
        out["usage"] = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage["output_tokens"],
            "total_tokens": usage.get("total_tokens"),
        }
        details = usage.get("output_tokens_details")
        if isinstance(details, dict):
            thinking = details.get("reasoning_tokens")
            if isinstance(thinking, int) and not isinstance(thinking, bool):
                out["usage"]["completion_tokens_details"] = {"reasoning_tokens": thinking}
    return out


@dataclass
class Normalized:
    kind: str  # "text" | "media" | "embedding"
    text: str | None = None
    completion_tokens: int | None = None
    media_urls: list[str] | None = None
    media_blobs: list[bytes] | None = None
    raw: dict | None = None  # the final backend response (text: the OpenAI response object)
    # Media facts that travel with the frames in the sealed result. Width and
    # height are the pixel dimensions billing multiplies, so they are stated
    # rather than recovered by decoding the output; duration_secs is set for
    # video only, and its presence is what marks a result as one.
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    duration_secs: int | None = None
    seed: int | None = None


@dataclass
class Mapping:
    request: dict           # {method, url, headers, body}
    response: dict          # {mode, poll?, result}
    health: dict | None
    params: dict            # base_url, api_key, model, extra_params, params_supported
    preset: str | None


def expand_preset(be: BackendConfig) -> Mapping:
    if be.preset not in ("openai-chat", "openai-embeddings", "openai-responses", "openai-batch"):
        raise BackendError(f"unknown preset: {be.preset!r}")
    p = be.params
    base = str(p["base_url"]).rstrip("/")
    headers = {"Content-Type": "application/json"}
    if p.get("api_key"):
        headers["Authorization"] = f"Bearer {p['api_key']}"
    if be.preset == "openai-embeddings":
        return Mapping(
            request={"method": "POST", "url": f"{base}/embeddings", "headers": headers},
            # No `completion_tokens` path, and its absence is the contract: an embeddings
            # response carries `usage.prompt_tokens` and nothing else, so there is no output
            # count to bill. `_extract` therefore hands back `completion_tokens=None`, the
            # scheduler settles at 0, and the charge is the input leg alone.
            response={"mode": "sync", "result": {"embeddings": "$.data"}},
            health=be.health,
            params=p,
            preset="openai-embeddings",
        )
    if be.preset == "openai-responses":
        # A background submit answered by an id, then polled by that id: the
        # generic resumable poll mapping, with the same auth on every tick. The
        # final object is normalised into the chat-completion shape before
        # extraction (`responses_to_chat`), so the result paths are the chat ones.
        return Mapping(
            request={"method": "POST", "url": f"{base}/responses", "headers": headers},
            response={
                "mode": "poll",
                "poll": {
                    "handle": "$.id",
                    "request": {"method": "GET", "url": f"{base}/responses/{{handle}}",
                                "headers": headers},
                    "status_field": "$.status",
                    "done_values": ["completed"],
                    "failed_values": ["failed", "cancelled", "incomplete"],
                    "max_polls": int(p.get("max_polls", 60)),
                },
                "result": {
                    "text": "$.choices[0].message.content",
                    "completion_tokens": "$.usage.completion_tokens",
                },
            },
            health=be.health,
            params=p,
            preset="openai-responses",
        )
    if be.preset == "openai-batch":
        # One job is one batch of one line. The submit is two calls (upload, then
        # create — `_batch_submit`); the batch id is the handle; the finished
        # batch names an output file whose one line carries the chat completion
        # (`_batch_collect`), so the result paths are the chat ones.
        return Mapping(
            request={"method": "POST", "url": f"{base}/batches", "headers": headers},
            response={
                "mode": "poll",
                "poll": {
                    "handle": "$.id",
                    "request": {"method": "GET", "url": f"{base}/batches/{{handle}}",
                                "headers": headers},
                    "status_field": "$.status",
                    "done_values": ["completed"],
                    "failed_values": ["failed", "cancelled", "expired"],
                    "max_polls": int(p.get("max_polls", 60)),
                },
                "result": {
                    "text": "$.choices[0].message.content",
                    "completion_tokens": "$.usage.completion_tokens",
                },
            },
            health=be.health,
            params=p,
            preset="openai-batch",
        )
    return Mapping(
        request={"method": "POST", "url": f"{base}/chat/completions", "headers": headers,
                 "stream": be.stream},
        response={
            "mode": "sync",
            "result": {
                "text": "$.choices[0].message.content",
                "completion_tokens": "$.usage.completion_tokens",
            },
        },
        health=be.health,
        params=p,
        preset="openai-chat",
    )


def _render_request(request: dict, ctx: dict) -> dict:
    """``render``, with a field the job does not carry turned into a refusal.

    An unresolved required token leaves ``render`` as a ``KeyError``. Let through,
    it reaches the scheduler's crash handler, which reports no ``fail`` — and the
    claim sits until the SLA reclaims it, at this provider's expense, over a
    client's omission. The message is the token's *name*, never a value.
    """
    try:
        return render(request, ctx)
    except KeyError as exc:
        raise BackendError(f"the request cannot be built: {exc.args[0]}") from None


def _raw_mapping(be: BackendConfig) -> Mapping:
    return Mapping(
        request=be.request or {},
        response=be.response or {},
        health=be.health,
        params=be.params,
        preset=None,
    )


class BackendDriver:
    def __init__(
        self,
        client: httpx.AsyncClient,
        mapping: Mapping,
        *,
        sleep=asyncio.sleep,
        clock=time.monotonic,
        retry_statuses: tuple[int, ...] = (),
    ) -> None:
        self._client = client
        self.mapping = mapping
        self._sleep = sleep
        self._clock = clock
        self._retry_statuses = frozenset(retry_statuses)

    @classmethod
    def from_config(cls, client: httpx.AsyncClient, be: BackendConfig, **kw) -> "BackendDriver":
        mapping = expand_preset(be) if be.is_preset else _raw_mapping(be)
        kw.setdefault("retry_statuses", tuple(getattr(be, "retry_statuses", ()) or ()))
        return cls(client, mapping, **kw)

    @property
    def resumable(self) -> bool:
        """Can a job at this backend be picked up again after a restart?

        Only a poll mapping that names the submit-response field identifying
        the job (``poll.handle``): that value is what a later boot polls with.
        """
        poll = self.mapping.response.get("poll") or {}
        return self.mapping.response.get("mode") == "poll" and "handle" in poll

    # -- execution -----------------------------------------------------------

    async def run(self, job, input: dict, *, timeout_s: float | None = None,
                  resume: str | None = None, on_handle=None) -> Normalized:
        """Execute one job.

        ``on_handle(handle)`` is called the moment a submit response yields the
        backend's handle for the job — before the first poll tick, so a crash
        anywhere after it can be resumed. ``resume`` is such a handle from a
        previous life: the submit is skipped and the poll starts from it.
        """
        m = self.mapping
        # `input` may carry a reference asset — megabytes of base64 belonging to a
        # client who sealed it precisely so nobody else would read it. **Nothing
        # built from this context is ever logged**, at any level: not the rendered
        # body, not the request, not an exception carrying either. The response
        # log below truncates for the same reason from the other direction. A
        # debug line added here would put a client's private reference into an
        # operator's log aggregator and keep it there.
        ctx = {"input": input,
               "job": {"id": job.job_id, "owner": job.owner,
                       "units_out": job.units_out, "modality": job.modality},
               **m.params}
        # The caller bounds the request by the job's SLA window (plus grace); a
        # completion that takes minutes must not be aborted before the deadline.
        timeout = httpx.Timeout(timeout_s, connect=10.0) if timeout_s is not None else None
        mode = m.response.get("mode", "sync")
        poll = m.response.get("poll") or {}
        sla = _sla_s(getattr(job, "sla", None))

        if resume is not None and self.resumable:
            final = await self._poll(poll, None, ctx, timeout=timeout, sla=sla, handle=resume)
        else:
            submit = await self._submit(m, job, input, ctx, timeout)
            handle = None
            if self.resumable:
                handle = extract(poll["handle"], submit)
                if isinstance(handle, bool) or not isinstance(handle, (str, int)) or handle == "":
                    # The work may or may not be running; a second submit would
                    # not tell us, so this is the job's shape, not a fault to retry.
                    raise BackendError("the submit response carried no poll handle")
                handle = str(handle)
                if on_handle is not None:
                    on_handle(handle)
            final = submit if mode == "sync" else await self._poll(
                poll, submit, ctx, timeout=timeout, sla=sla, handle=handle)
        if m.preset == "openai-responses":
            final = responses_to_chat(final)
        if m.preset == "openai-batch":
            batch = final
            try:
                final = await self._batch_collect(m, job, batch, timeout)
            finally:
                await self._batch_discard_input(m, batch)
        return self._extract(m.response["result"], final, input, getattr(job, "modality", None))

    async def _submit(self, m: Mapping, job, input: dict, ctx: dict, timeout) -> dict:
        """The submit call: the preset's body or the rendered raw request."""
        if m.preset == "openai-chat":
            method, url = "POST", m.request["url"]
            headers = m.request["headers"]
            body = self._openai_chat_body(input, job, m.params)
        elif m.preset == "openai-embeddings":
            method, url = "POST", m.request["url"]
            headers = m.request["headers"]
            body = self._openai_embeddings_body(input, m.params)
        elif m.preset == "openai-responses":
            method, url = "POST", m.request["url"]
            headers = m.request["headers"]
            body = self._openai_responses_body(input, job, m.params)
        elif m.preset == "openai-batch":
            return await self._batch_submit(m, job, input, timeout)
        else:
            ctx = {**ctx, "prepare": await self._prepare(m.request.get("prepare") or [], ctx, timeout)}
            req = _render_request({k: v for k, v in m.request.items() if k != "prepare"}, ctx)
            method = req.get("method", "POST")
            url = req["url"]
            headers = req.get("headers", {})
            body = req.get("body")
        if m.preset is not None and m.params.get("headers"):
            headers = {**headers, **_render_request(m.params["headers"], ctx)}

        if m.preset == "openai-chat" and m.request.get("stream"):
            # The operator's switch, set after the body is built: a client's own
            # `stream` never reaches the wire (NEVER_FORWARDED).
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            return await self._request_stream(method, url, headers, body, timeout=timeout)
        answer = await self._request_json(method, url, headers, body, timeout=timeout)
        return self._answered(answer) if m.preset is None else answer

    async def _prepare(self, steps: list, ctx: dict, timeout) -> dict:
        """Run the requests a raw submit depends on; answer ``{name: {key: value}}``.

        For a backend that takes a reference as a URL and offers an upload endpoint
        to mint one. Each step is an ordinary templated request, run in order and
        skipped when its ``when`` path resolves to nothing in the job; what its
        ``extract`` block pulls from the answer is in scope for the steps after it
        and for the submit, as ``{prepare.<name>.<key>}``. A skipped step leaves
        nothing behind, so the submit names it with a trailing ``?``.

        A step that ran and answered without what it was run for fails the job.
        The optional token downstream would otherwise drop the field, and the job
        would run as a different one from the one the client paid for.

        The same rule as the submit holds: nothing rendered here is ever logged.
        """
        async def one(step: dict, scope: dict) -> dict:
            req = _render_request({k: step[k] for k in ("method", "url", "headers", "body")
                                   if k in step}, scope)
            answer = self._answered(await self._request_json(
                req.get("method", "POST"), req["url"], req.get("headers", {}),
                req.get("body"), timeout=timeout))
            values = {key: extract(path, answer) for key, path in step["extract"].items()}
            missing = sorted(key for key, value in values.items() if value in (None, "", []))
            if missing:
                raise BackendError(
                    f"prepare step {step['name']!r} answered without {', '.join(missing)}")
            return values

        done: dict = {}
        for step in steps:
            scope = {**ctx, "prepare": done}
            if "for_each" in step:
                # Once per element of a listed reference, each in scope as {item};
                # the step's name then holds the list of what each run extracted.
                items = [m.value for m in parse_path(step["for_each"]).find(scope)]
                items = items[0] if items and isinstance(items[0], list) else []
                if items:
                    done[step["name"]] = [await one(step, {**scope, "item": item}) for item in items]
                continue
            if "when" in step and not resolves(step["when"], scope):
                continue
            done[step["name"]] = await one(step, scope)
        return done

    def _answered(self, answer):
        """A raw backend's JSON answer, or a refusal when the *body* says it failed.

        Some task APIs answer HTTP 200 for everything and carry the real status in
        a field. Read as the success its status line claims, a refused submit is a
        submit that "lost its handle" and a rate limit is a poll status nobody
        recognises. ``response.ok`` names the field, the values that mean success
        and the ones worth another attempt. An answer without the field is not a
        refusal. The backend's message is logged and never raised: it may quote
        the client's input, and a raised reason travels to the network.
        """
        ok = self.mapping.response.get("ok")
        if not ok or not isinstance(answer, (dict, list)):
            return answer
        status = extract(ok["field"], answer)
        if status is None or status in ok["values"]:
            return answer
        if "message" in ok:
            log.warning("backend answered %r in the body: %.200s", status, extract(ok["message"], answer))
        raise BackendError(f"the backend answered {status!r}", retryable=status in ok.get("retry", []))

    @staticmethod
    def _reasoning_defaults(input: dict, job, params: dict, allowed: set, rename: dict) -> dict:
        """Fill the reasoning controls the payload left unset, without ever
        overriding one it carries.

        A payload with no reasoning control at all gets the model's declared
        ``default_effort`` — the backend's own default is unpublished and often
        the top of the ladder, so the declared one is what a silent client is
        actually quoted against. An effort without an explicit budget then
        derives one as that effort's share of the charged cap
        (:func:`budget_for_effort`) — only where the dialect publishes a budget
        field (the map renames the canonical key, or forwards it verbatim);
        where it drops the key, nothing is derived just to be dropped.
        """
        if "reasoning_effort" not in input and "reasoning_max_tokens" not in input:
            default = params.get("default_effort")
            if default is not None and "reasoning_effort" in allowed:
                input = {**input, "reasoning_effort": default}
        effort = input.get("reasoning_effort")
        budget_spec = rename.get("reasoning_max_tokens")
        if (
            isinstance(effort, str)
            and "reasoning_max_tokens" not in input
            and "reasoning_max_tokens" in allowed
            and (budget_spec is None or isinstance(budget_spec, str))
        ):
            derived = budget_for_effort(effort, job.units_out)
            if derived is not None:
                input = {**input, "reasoning_max_tokens": derived}
        return input

    @staticmethod
    def _openai_embeddings_body(input: dict, params: dict) -> dict:
        """The embeddings request, which is `input` plus at most two knobs.

        Deliberately not built through the chat allowlist: this surface has no sampling,
        no messages and no reasoning: forwarding a chat param here is a 400 from the
        backend and a job the provider fails at its own cost.

        `encoding_format` defaults to base64 because a float32 vector is ~4x smaller that
        way than as JSON floats (a 3072-dim vector: ~16 KB against ~61 KB), and a batch of
        them is what the sealed result carries.

        `input_type` is the operator's, and the caller's only where the operator says so.
        Asymmetric retrieval models embed a query and a passage into different points, so
        the field decides which space a vector lands in. Which side a deployment serves is
        normally a property of the deployment — an index builder pins `passage`, a search
        path pins `query`, and the network id the caller submits against is what tells them
        apart. That is the default here, and it stays the default: a caller able to set the
        field on an id that *names* a side could make the id a lie.

        `input_type_overridable: true` opts one model out of that, for the case the id
        makes no such claim: a single published id, an operator-pinned side for callers who
        say nothing, and a caller who knows which side it needs able to ask. The operator's
        value is the floor, never bypassed — a caller naming something outside
        :data:`EMBEDDING_INPUT_TYPES` is dropped back onto it rather than forwarded into a
        `400`, since the job is already claimed by the time this body is built.

        Absent entirely when the operator names none and opens nothing. That is not a safe
        default, only a compatible one: on a model that *accepts* the omission rather than
        refusing it, the vector comes back well-formed from an untemplated space, the job
        succeeds, and only the caller's retrieval quality is destroyed. Operators should
        pin a side.
        """
        body: dict = {
            "model": params["model"],
            "input": input.get("input"),
            "encoding_format": input.get("encoding_format", "base64"),
        }
        if input.get("dimensions") is not None:
            body["dimensions"] = input["dimensions"]
        side = params.get("input_type")
        if params.get("input_type_overridable"):
            asked = input.get("input_type")
            # `isinstance` first: a sealed payload is arbitrary JSON, and an
            # unhashable value (a list, an object) raises on set membership.
            if isinstance(asked, str) and asked in EMBEDDING_INPUT_TYPES:
                side = asked
            elif asked is not None:
                log.info(
                    "dropped input_type %r: not one of %s",
                    asked, ", ".join(sorted(EMBEDDING_INPUT_TYPES)),
                )
        if side is not None:
            body["input_type"] = side
        return body

    def _openai_chat_body(self, input: dict, job, params: dict, *,
                          default_params: frozenset = OPENAI_CHAT_PARAMS) -> dict:
        # A backend that publishes a per-model schema gets exactly that schema:
        # forwarding a field the served model does not accept risks a 400, and
        # a rejected request is a job the provider fails at its own cost.
        # `default_params` is the vocabulary a silent entry falls back to — the
        # chat one here, the Responses one for that preset.
        supported = params.get("params_supported")
        allowed = set(supported if supported is not None else default_params)
        rename = params.get("param_map") or {}
        input = self._reasoning_defaults(input, job, params, allowed, rename)
        body: dict = {}
        dropped: list[str] = []
        for key, value in input.items():
            if key in ("model", "input"):
                continue  # "model": runtime model overrides; "input": consumed by the messages sugar below
            if key == "messages":
                body[key] = value
            elif key in NEVER_FORWARDED:
                dropped.append(key)
            elif key in allowed:
                # Budget-class clamp, on the canonical keys so it holds in every
                # dialect: thinking tokens are generated inside the charged
                # units_out, and a floor must fit inside it too. Anything but a
                # non-negative int is dropped, not coerced — a sealed payload is
                # untrusted, and several backends read a negative budget as an
                # "unlimited" sentinel that would bill the backend for tokens
                # settlement cannot pass on.
                if key in ("reasoning_max_tokens", "min_tokens"):
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        dropped.append(key)
                        continue
                    # The thinking budget clamps strictly below the cap — it
                    # spends units_out from the inside and the answer needs the
                    # remainder — while min_tokens == cap stays coherent
                    # ("generate exactly the cap").
                    if key == "reasoning_max_tokens":
                        value = min(value, max(job.units_out - 1, 0))
                    else:
                        value = min(value, job.units_out)
                if not _apply_param_map(body, key, value, rename.get(key)):
                    dropped.append(key)
            else:
                dropped.append(key)
        if dropped:
            log.info("dropped params not in the served schema: %s", ", ".join(sorted(dropped)))
        if "messages" not in body and "input" in input:
            raw = input["input"]
            body["messages"] = raw if isinstance(raw, list) else [{"role": "user", "content": raw}]
        body["model"] = params["model"]
        body["max_tokens"] = job.units_out
        for key, value in (params.get("extra_params") or {}).items():
            body[key] = value
        return body

    def _openai_responses_body(self, input: dict, job, params: dict) -> dict:
        """The chat body, respelled for a Responses-style endpoint.

        Same reasoning defaults and clamps, over the Responses vocabulary — then
        the payload is put into the Responses spelling
        (:func:`_responses_dialect`), the configured tier is added, and
        ``background`` asks for the id the daemon polls and can resume from.

        ``extra_params`` are held back from the body and respelled on their own,
        then merged last, so the operator's escape hatch still overrides
        everything — in either dialect. Respelling them together with the
        payload would instead let the payload's own keys land on top of them,
        the renames arriving after the merge (a preset ``max_tokens`` renamed
        over an operator's ``max_output_tokens``). ``reasoning`` and ``text``
        merge key by key, so naming one field of an object does not erase the
        rest of it.
        """
        body = self._openai_chat_body(input, job, {**params, "extra_params": None},
                                      default_params=OPENAI_RESPONSES_PARAMS)
        if "messages" not in body:
            # No prompt at all: the chat endpoint would answer 400 and the
            # scheduler would refund; here the same outcome, without a round trip.
            raise BackendError("the payload carries no messages or input")
        _responses_dialect(body)
        if params.get("service_tier"):
            body["service_tier"] = params["service_tier"]
        body["background"] = True
        for key, value in _responses_dialect(dict(params.get("extra_params") or {})).items():
            current = body.get(key)
            if key in ("reasoning", "text") and isinstance(current, dict) and isinstance(value, dict):
                body[key] = {**current, **value}
            else:
                body[key] = value
        return body

    @staticmethod
    def _batch_auth(m: Mapping) -> dict:
        """The preset's headers minus the JSON content type: the upload is multipart."""
        return {k: v for k, v in m.request["headers"].items() if k.lower() != "content-type"}

    async def _batch_submit(self, m: Mapping, job, input: dict, timeout) -> dict:
        """Upload a one-line JSONL file, then create the batch over it.

        The line body is the chat-completion body, verbatim — the batch surface
        runs each line against the chat endpoint. Both calls classify like any
        other: a refusal is the job's shape, a 5xx or transport fault is retried by
        the scheduler (no handle has been recorded yet).
        """
        p = m.params
        base = str(p["base_url"]).rstrip("/")
        endpoint = p.get("endpoint", "/v1/chat/completions")
        body = self._openai_chat_body(input, job, p)
        if "messages" not in body:
            # No prompt at all: a line the chat endpoint would answer 400 is not
            # worth an upload and a capacity slot held for the whole window.
            raise BackendError("the payload carries no messages or input")
        line = {"custom_id": _batch_custom_id(job), "method": "POST", "url": endpoint, "body": body}
        upload = await self._request_multipart(
            f"{base}/files", self._batch_auth(m), {"purpose": "batch"},
            {"file": ("vorqd.jsonl", json.dumps(line) + "\n", "application/jsonl")},
            timeout=timeout)
        file_id = upload.get("id") if isinstance(upload, dict) else None
        if not isinstance(file_id, str) or not file_id:
            raise BackendError("the batch upload answered with no file id")
        return await self._request_json(
            "POST", m.request["url"], m.request["headers"],
            {"input_file_id": file_id, "endpoint": endpoint,
             "completion_window": p.get("completion_window", "24h")},
            timeout=timeout)

    async def _batch_collect(self, m: Mapping, job, batch: dict, timeout) -> dict:
        """The job's own line out of the finished batch: its chat completion, or why not.

        A batch reports `completed` even when its one line failed, so the status
        alone settles nothing. The output file is read first; a line answered
        200 is the completion. Anything else — a non-200 line, a line in the
        error file, or no line at all — fails the job without a retry: the work
        was accepted and ran, and running it again would pay twice for the same
        refusal. The upstream detail goes to the operator log only.
        """
        base = str(m.params["base_url"]).rstrip("/")
        auth = self._batch_auth(m)
        wanted = _batch_custom_id(job)
        row = await self._batch_line(base, auth, batch.get("output_file_id"), wanted, timeout)
        if row is not None:
            resp = row.get("response")
            if isinstance(resp, dict) and resp.get("status_code") == 200 \
                    and isinstance(resp.get("body"), dict):
                return resp["body"]
            log.warning("batch line failed: %s — %.300s", wanted, json.dumps(row))
            raise BackendError("the batch line failed")
        row = await self._batch_line(base, auth, batch.get("error_file_id"), wanted, timeout)
        if row is not None:
            log.warning("batch line failed: %s — %.300s", wanted, json.dumps(row))
            raise BackendError("the batch line failed")
        log.warning("batch carried no line for the job: %s — %.300s", wanted, json.dumps(batch))
        raise BackendError("the batch reported no result for the job")

    async def _batch_line(self, base: str, auth: dict, file_id, custom_id: str, timeout) -> dict | None:
        """The JSONL row for `custom_id` in one content file, or None.

        A refusal answers None rather than propagating: one file is not the
        job's verdict. An output file a `completed` batch never wrote answers
        404 here, and re-raising it would end the job before the error file —
        which is where that batch's refusal actually is — had been read at all.
        `_batch_collect` reaches the verdict, over both files.

        A retryable fault is absorbed and the fetch tried again, never raised —
        the same invariant `_poll` holds: by this point the batch has run and
        been billed, and an error that reached the scheduler's retry would
        upload and create a second one for work already done. The scheduler's
        deadline bounds the loop.

        A surface may answer a slice of a large file and say so in
        ``X-Incomplete``, taking ``?skip=<rows>`` for the rest; the pages are
        walked until the row is found, the header stops saying `true`, or a page
        carries no rows at all — the last being the guard against a header that
        never clears.
        """
        if not isinstance(file_id, str) or not file_id:
            return None
        url = f"{base}/files/{file_id}/content"
        read = 0
        while True:
            try:
                text, headers = await self._batch_content(url, auth, read, timeout)
            except BackendError as exc:
                log.warning("batch content file unavailable: %s — %s", url, exc)
                return None
            rows = 0
            for raw in text.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rows += 1
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("custom_id") == custom_id:
                    return row
            if rows == 0 or str(headers.get("X-Incomplete", "")).lower() != "true":
                return None
            read += rows
            log.debug("batch content file is partial, reading from row %d: %s", read, url)

    async def _batch_content(self, url: str, auth: dict, skip: int, timeout) -> tuple[str, dict]:
        """One page of a content file, with the retryable faults absorbed."""
        params = {"skip": skip} if skip else None
        while True:
            try:
                return await self._request_text("GET", url, auth, params=params, timeout=timeout)
            except BackendError as exc:
                if not exc.retryable:
                    raise
                log.warning("batch content fetch failed, retrying: %s", exc)
                await self._sleep(max(5.0, exc.retry_after_s or 0))

    async def _batch_discard_input(self, m: Mapping, batch: dict) -> None:
        """Best-effort delete of the one-line input file the submit uploaded.

        The result has been collected, so the file is spent — and a file left
        behind per job is charged as storage at some surfaces and counts against
        a file-count quota at most. Nothing here may change the job's outcome:
        the delete runs on the failure paths too, and every fault it meets is
        the operator's to read. The output and error files are deliberately left
        for exactly that reading.
        """
        file_id = batch.get("input_file_id") if isinstance(batch, dict) else None
        if not isinstance(file_id, str) or not file_id:
            return
        base = str(m.params["base_url"]).rstrip("/")
        # Its own short wall, not the job's: a delete that hangs must not hold
        # an already-collected result until the deadline fires.
        try:
            await self._send("DELETE", f"{base}/files/{file_id}",
                             headers=self._batch_auth(m),
                             timeout=httpx.Timeout(_BATCH_DISCARD_TIMEOUT_S, connect=10.0))
        except Exception as exc:  # noqa: BLE001 — best-effort by contract
            log.info("batch input file was not deleted: %s — %s", file_id, exc)

    async def _send(self, method: str, url: str, *, timeout=None, **kwargs) -> httpx.Response:
        """One backend call. The scheduler owns retries; this classifies.

        A :class:`BackendError` says whether another attempt could succeed: a
        ``429``, a ``5xx`` and a transport fault (a timeout included) are
        ``retryable``, and a ``429``'s ``Retry-After`` rides along as
        ``retry_after_s``. Any other ``4xx`` is a request the backend would refuse
        again, so it is not.

        Two audiences, two channels. The operator's log gets the URL and the
        upstream's own words — the status alone hides whether it was capacity, a
        cold model, or maintenance. The :class:`BackendError` gets the status and
        whose fault it was, and nothing else: its message becomes the reason on the
        failure report the coordinator records against the client's job, and a
        backend URL can carry credentials in its query while an upstream error body
        can carry anything at all.
        """
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = await self._client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            log.warning("backend unreachable (%s): %s — %s", type(exc).__name__, url, exc)
            raise BackendError("the backend was unreachable", retryable=True) from exc
        self._refuse(resp.status_code, resp.headers.get("Retry-After"), url, resp.text[:300])
        return resp

    async def _request_json(self, method: str, url: str, headers: dict, body, *, timeout=None) -> dict:
        """One classified call answered by JSON. A body rides on anything but a GET."""
        kwargs: dict = {"headers": headers}
        if body is not None and method.upper() != "GET":
            kwargs["json"] = body
        resp = await self._send(method, url, timeout=timeout, **kwargs)
        return self._json_body(resp, url)

    async def _request_text(self, method: str, url: str, headers: dict, *, params=None,
                            timeout=None) -> tuple[str, dict]:
        """One classified call answered by text — a JSONL content file, not an
        object — as ``(text, headers)``: a paged file says so in a header."""
        resp = await self._send(method, url, headers=headers, params=params, timeout=timeout)
        return resp.text, resp.headers

    async def _request_multipart(self, url: str, headers: dict, data: dict, files: dict,
                                 *, timeout=None) -> dict:
        """One classified multipart upload, answered by JSON. The content type is
        the encoder's: `headers` must carry none of its own."""
        resp = await self._send("POST", url, headers=headers, data=data, files=files,
                                timeout=timeout)
        return self._json_body(resp, url)

    @staticmethod
    def _json_body(resp: httpx.Response, url: str) -> dict:
        """The 2xx's parsed body, or a retryable refusal.

        A 200 whose body is not JSON is an interposed gateway answering for the
        backend — an HTML error page, a truncated read — not the backend's own
        answer. Another attempt may well reach it, so this is retryable; the
        body itself is the operator's, capped, and never the client's.
        """
        try:
            return resp.json()
        except ValueError as exc:
            log.warning("backend answered with an unreadable body: %s — %.300s", url, resp.text)
            raise BackendError("the backend answered with an unreadable body",
                               retryable=True) from exc

    def _refuse(self, status: int, retry_after: str | None, url: str, text: str) -> None:
        """Raise the :class:`BackendError` a non-2xx status classifies to; return on a 2xx."""
        if status in self._retry_statuses:
            log.warning("backend failed (HTTP %s, retried by configuration): %s — %s",
                        status, url, text)
            raise BackendError(f"the backend failed with HTTP {status}", retryable=True)
        if status >= 500:
            log.warning("backend failed (HTTP %s): %s — %s", status, url, text)
            raise BackendError(f"the backend failed with HTTP {status}", retryable=True)
        if status == 429:
            retry_after_s = parse_retry_after(retry_after)
            log.warning("backend throttled (HTTP 429, retry-after %s): %s — %s",
                        retry_after_s, url, text)
            raise BackendError("the backend throttled the request with HTTP 429",
                               retryable=True, retry_after_s=retry_after_s)
        if status >= 400:
            log.warning("backend refused (HTTP %s): %s — %s", status, url, text)
            raise BackendError(f"the backend refused the request with HTTP {status}")

    async def _request_stream(self, method: str, url: str, headers: dict, body, *,
                              timeout=None) -> dict:
        """One streamed chat completion, reassembled into the object the plain
        call returns, so extraction and the settled result are the same shape.

        The read timeout bounds the silence between two chunks, not the whole
        answer: a generation that keeps producing outlives a gateway that closes
        a request after a fixed wall, which is what streaming is for. The
        attempt as a whole ends at the scheduler's SLA wall. The usage block is
        asked for on the final chunk because the settled count comes from it; a
        stream that ends without a finish reason and without `[DONE]`, or
        without the block, is an incomplete answer and retryable, as is a
        fault the backend reports mid-stream.
        """
        kwargs: dict = {"headers": headers, "json": body}
        if timeout is not None:
            kwargs["timeout"] = timeout
        head: dict = {}
        content: list[str] = []
        reasoning: list[str] = []
        finish = None
        usage = None
        done = False
        try:
            async with self._client.stream(method, url, **kwargs) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread())[:300].decode(errors="replace")
                    self._refuse(resp.status_code, resp.headers.get("Retry-After"), url, text)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        log.warning("backend stream carried an unreadable chunk: %s — %.200s", url, data)
                        raise BackendError("the backend stream was unreadable", retryable=True)
                    if not isinstance(chunk, dict):
                        continue
                    if "error" in chunk and "choices" not in chunk:
                        log.warning("backend failed mid-stream: %s — %.300s", url, data)
                        raise BackendError("the backend failed mid-stream", retryable=True)
                    if not head:
                        head = {k: chunk[k] for k in ("id", "created", "model", "system_fingerprint")
                                if k in chunk}
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except httpx.TransportError as exc:
            log.warning("backend stream broke (%s): %s — %s", type(exc).__name__, url, exc)
            raise BackendError("the backend was unreachable" if not head else "the backend stream broke",
                               retryable=True) from exc
        if not done and finish is None:
            log.warning("backend stream ended early (no finish reason, no [DONE]): %s", url)
            raise BackendError("the backend closed the stream early", retryable=True)
        if usage is None:
            # The block was asked for and is what settles; a stream that ended
            # without it is an answer that cannot be billed, not one to abandon.
            log.warning("backend stream ended without its usage block: %s", url)
            raise BackendError("the backend stream carried no usage block", retryable=True)
        message: dict = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        final: dict = {
            **head,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        if usage is not None:
            final["usage"] = usage
        return final

    async def _poll(self, poll: dict, submit: dict | None, ctx: dict, *, timeout=None,
                    sla: float | None = None, handle: str | None = None) -> dict:
        """Wait for a submitted job to finish.

        A status tick that fails retryably is absorbed here, never raised: the
        work is already submitted, and an error that reached the scheduler's
        retry would submit it again. ``timeout`` is the caller's per-request
        ceiling and applies to every tick. The poll's own wall (``timeout_s``)
        is optional — without one, the scheduler's deadline bounds the wait.

        The cadence is ``interval_s`` when set, else the job's SLA window
        divided by ``max_polls`` (default 60) and never under a second: a 24h
        job is asked about every 24 minutes, a 1h job every minute. ``submit``
        is ``None`` on a resumed job, which has only ``{handle}`` in scope.
        """
        status_field = poll["status_field"]
        done = set(poll.get("done_values", []))
        failed = set(poll.get("failed_values", []))
        interval = poll.get("interval_s")
        if interval is None:
            interval = max(1.0, sla / int(poll.get("max_polls", 60))) if sla else 2
        wall_s = poll.get("timeout_s")
        start = self._clock()

        status_url = extract(poll["status_url"], submit) if "status_url" in poll and submit is not None else None
        poll_ctx = {**ctx, "submit": submit if submit is not None else {}, "handle": handle}

        while True:
            if wall_s is not None and self._clock() - start > wall_s:
                raise BackendError(f"backend poll timed out after {wall_s}s", retryable=True)
            try:
                if status_url is not None:
                    body = await self._request_json("GET", status_url, {}, None, timeout=timeout)
                else:
                    req = _render_request(poll["request"], poll_ctx)
                    body = await self._request_json(
                        req.get("method", "POST"), req["url"], req.get("headers", {}),
                        req.get("body"), timeout=timeout,
                    )
                body = self._answered(body)
            except BackendError as exc:
                if not exc.retryable:
                    raise
                log.warning("backend poll tick failed, polling on: %s", exc)
                await self._sleep(max(interval, exc.retry_after_s or 0))
                continue
            status = extract(status_field, body)
            if status in failed:
                # The backend's own code travels in the reason — 'fail' alone reads
                # the same for a content refusal and an outage. Its message goes to
                # the log only: it may quote the client's prompt.
                code = extract(poll["failure_code"], body) if "failure_code" in poll else None
                if "failure_message" in poll:
                    log.warning("backend task failed (%r): %.200s", code,
                                extract(poll["failure_message"], body))
                raise BackendError(f"backend reported failure status: {status!r}"
                                   + (f" ({code})" if code not in (None, "") else ""))
            if status in done:
                return body
            await self._sleep(interval)

    def _extract(self, result: dict, final: dict, input: dict | None = None,
                 modality: str | None = None) -> Normalized:
        if "embeddings" in result:
            # `completion_tokens` stays None deliberately — see `expand_preset`. The whole
            # response is the result: the client is handed the OpenAI `EmbeddingResponse`
            # verbatim, `usage.prompt_tokens` included, sealed like any other.
            return Normalized(kind="embedding", completion_tokens=None, raw=final)
        if "text" in result:
            ct = extract(result["completion_tokens"], final) if "completion_tokens" in result else None
            return Normalized(kind="text", text=extract(result["text"], final), completion_tokens=ct, raw=final)
        media = self._media_facts(result, final, input or {}, modality)
        if "media_urls" in result:
            return Normalized(kind="media", media_urls=_media_urls(extract(result["media_urls"], final)),
                              raw=final, **media)
        if "media_b64" in result:
            return Normalized(kind="media", media_blobs=_media_blobs(extract(result["media_b64"], final)),
                              raw=final, **media)
        raise BackendError("response.result declares no extractable output")  # pragma: no cover

    @staticmethod
    def _media_facts(result: dict, final: dict, input: dict, modality: str | None) -> dict:
        """The facts that travel with the frames inside the sealed media result.

        Width and height come from the request the job was priced against — the
        same numbers, read the same way, that ``_frame_pixels`` multiplied into
        the settled unit count — so a client is handed a frame labelled with
        exactly the pixels it paid for. ``duration_secs`` is a video fact and is
        set only for that modality.

        ``seed`` and ``content_type`` are the backend's to report: a mapping that
        names a path for either takes the value from the response, and a request
        that pinned its own seed gets that one back when the backend states none.
        Both are coerced to the shape the sealed frame promises — an int and a
        string — so a path that resolves to anything else states nothing.
        """
        width, height = _frame_dims(input)
        facts: dict = {
            "width": width,
            "height": height,
            "content_type": _as_str(extract(result["content_type"], final)) if "content_type" in result else None,
            "seed": _as_int(extract(result["seed"], final) if "seed" in result else input.get("seed")),
        }
        if modality == "video":
            facts["duration_secs"] = _duration_secs(input)
        return facts

    # -- health --------------------------------------------------------------

    async def healthy(self) -> bool:
        h = self.mapping.health
        if not h:
            return True
        path = h["path"]
        if path.startswith("http"):
            url = path
        elif self.mapping.params.get("base_url"):
            url = str(self.mapping.params["base_url"]).rstrip("/") + path
        else:
            parts = urlsplit(self.mapping.request.get("url", ""))
            url = f"{parts.scheme}://{parts.netloc}{path}"
        try:
            resp = await self._client.get(url)
            return resp.status_code < 400
        except httpx.TransportError:
            return False
