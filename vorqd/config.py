"""Load and validate ``vorqd.yaml`` into typed config objects.

The daemon is driven entirely by this file. Loading fails fast (naming the
offending key or environment variable) so a misconfigured deployment never
reaches the poll loop. Validation follows mapping-reference §6.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ConfigError
from .limits import Throttle, window_seconds
from .reasoning import extract_default_effort, resolve_reasoning
from .money import parse_usd

_ENV_RE = re.compile(r"env:([A-Za-z_][A-Za-z0-9_]*)")
_RESULT_KINDS = ("text", "media_urls", "media_b64")

# A model name is free-form text. The market's ``e2ee-`` naming convention
# (``org/e2ee-model:quant``) is presentation only — no code anywhere inspects a
# name for it, and it carries zero semantics. ``confidential: true`` is the only
# switch that makes a model confidential.


# --------------------------------------------------------------------------- #
# Typed config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SlaRate:
    """One window's ask: USD per 1M units of work, as the decimal strings the
    config wrote. ``rate_in`` is ``None`` on a side the model does not meter."""

    rate_in: str | None
    rate_out: str


#: The furthest under the published ask the acceptance floor may ever sit.
#:
#: The cap is what keeps a floor a floor. Two allowances can sum past it, and an
#: unclamped percentage over 100 would invert the floor outright — a negative
#: price accepts a bid paying less than nothing. It lives here, with the knobs it
#: bounds, so :func:`_build_pricing` and :func:`vorqd.pricing.discounted_units`
#: enforce one number rather than two copies of it.
MAX_FLOOR_DISCOUNT_PCT = 90


@dataclass(frozen=True)
class PricingConfig:
    """The private acceptance floor, off by default.

    None of this is published. The ask book keeps advertising the ``slas`` rates
    exactly as configured; these knobs only decide how far **under** that
    advertised price this daemon is willing to claim, and that is deliberate — a
    floor the network can see is a floor bids converge onto.

    ``max_discount_pct`` is the whole allowance at a completely idle backend,
    shrinking linearly to nothing as load rises. ``bid_tolerance_pct`` is a
    further allowance, granted only while load is under ``low_load_pct``.

    Each is capped at :data:`MAX_FLOOR_DISCOUNT_PCT` by the loader, and the two
    together are capped at it again inside
    :func:`vorqd.pricing.discounted_units`, which is where a summed percentage
    first becomes a price: a floor discounted to zero would accept a bid that
    pays nothing, and one discounted past that would accept less than nothing.
    """

    max_discount_pct: int = 0        # 0 keeps the configured rates as the floor
    bid_tolerance_pct: int = 0       # 0 grants nothing extra at low load
    low_load_pct: int = 30


@dataclass(frozen=True)
class LoadProbeConfig:
    """Where to read one backend's load, and how to read it.

    Any runtime that exports Prometheus text works: name the endpoint and the
    series. ``scale`` is the divisor that turns that series into 0..1 — ``1.0``
    for a ratio gauge, ``100`` for a percentage.

    ``source: occupancy`` is the other shape, for a backend that exports
    nothing: the load is the entry's own occupancy — jobs held over what the
    entry may hold, attempts started over the longest ``rate_limit`` window —
    and there is no endpoint to name.
    """

    url: str | None = None      # absolute; the daemon does not assume the metrics
                                # endpoint shares a host with the inference API
    metric: str | None = None
    scale: float = 1.0
    source: str = "probe"       # "probe" (url + metric) or "occupancy"


@dataclass(frozen=True)
class BackendConfig:
    preset: str | None            # e.g. "openai-chat"; None for a raw mapping
    params: dict = field(default_factory=dict)   # preset params (base_url, model, ...)
    request: dict | None = None   # raw submit mapping
    response: dict | None = None  # raw response mapping (mode/poll/result)
    health: dict | None = None
    # openai-chat only: stream the completion and reassemble it, so a long
    # generation outlives a gateway that closes a request after a fixed wall.
    stream: bool = False
    # Limits — how many attempts a job gets, how they are spaced, and how much
    # of this backend the daemon may use at once. See `_build_limits`.
    retries: int = 0
    # Statuses retried on top of 429 and 5xx — a gateway that answers 404 while
    # a pool scales, say. Any other 4xx still fails the job at once.
    retry_statuses: tuple[int, ...] = ()
    retry_backoff_s: float = 30.0
    retry_backoff_max_s: float = 900.0
    timeout_s: float | None = None
    concurrency: int | None = None
    # Window -> attempts started inside it. The window at the entry's shortest
    # SLA is also what the entry may hold: a day's budget on a `24h` entry is a
    # day's worth of claims, started `concurrency` at a time.
    rate_limit: dict[str, int] = field(default_factory=dict)
    # Consecutive jobs the backend failed outright before the model's asks are
    # withdrawn, and for how long. 0 never withdraws.
    trip_after: int = 3
    trip_cooldown_s: float = 60.0

    @property
    def is_preset(self) -> bool:
        return self.preset is not None

    @property
    def params_supported(self) -> list[str] | None:
        """The exact params the served model accepts, replacing the preset's
        default set. ``None`` keeps the preset default."""
        return self.params.get("params_supported")

    @property
    def param_map(self) -> dict | None:
        """The effective map — a named ``reasoning:`` preset already resolved and
        merged with any inline ``param_map`` at load time (see `_build_backend`)."""
        return self.params.get("param_map")

    @property
    def param_caps(self) -> dict | None:
        return self.params.get("param_caps")

    @property
    def reference_accept(self) -> list[str] | None:
        """The reference media types this backend takes, or ``None`` for "whatever
        this daemon can read".

        Narrows :data:`vorqd.media.READABLE_TYPES` and never widens it — a type
        outside that set has no reader, so accepting it would mean claiming jobs
        only to hand them straight back.
        """
        return (self.params.get("reference") or {}).get("accept")

    @property
    def media_policy(self) -> dict:
        """Everything this backend says about the media it takes and renders, as
        the keyword arguments :func:`vorqd.backend.plan_media_units` reads — one
        place, so a key added here cannot be forgotten at the call site."""
        reference = self.params.get("reference") or {}
        return {
            "accept": reference.get("accept"),
            "bounds": {kind: reference[kind] for kind in _REFERENCE_KINDS if kind in reference},
            "resolutions": self.params.get("resolutions"),
            "durations": self.params.get("durations"),
            "auto_duration": self.params.get("auto_duration"),
            "adaptive_aspect": self.params.get("adaptive_aspect"),
        }

    @property
    def durations(self) -> list[int] | None:
        """The clip lengths, in whole seconds, this backend renders — or ``None``
        for one that renders any. A request is clamped down to the greatest listed
        length its order covers, and handed back when there is none."""
        return self.params.get("durations")

    @property
    def default_effort(self) -> str | None:
        """The canonical effort injected when a payload carries no reasoning
        control at all — declared in the model's ``reasoning:`` spec."""
        return self.params.get("default_effort")


@dataclass(frozen=True)
class ModelConfig:
    model: str
    slas: dict[str, SlaRate]
    backend: BackendConfig
    confidential: bool = False
    # Local fallback for the catalog's modality, used only when the curated
    # catalog cannot be read. Modality decides whether a job is metered by tokens
    # or by planned pixels, so the daemon refuses to run a model whose modality it
    # cannot establish from either source — declaring it here keeps a provider
    # serving through a catalog outage.
    modality: str | None = None
    #: Where this model's backend reports its load, if it does. ``None`` — the
    #: default — leaves the model's acceptance floor at its configured rates.
    load: LoadProbeConfig | None = None
    #: SLA window -> the backend that serves the model at that window, each one
    #: ``backend`` with the window's patch merged over it. Empty — the default —
    #: serves every window from ``backend``.
    sla_backends: dict[str, BackendConfig] = field(default_factory=dict)

    def backend_for(self, sla: str | None) -> BackendConfig:
        """The backend this model is served through at ``sla``."""
        return self.sla_backends.get(sla, self.backend)


#: The job registry's own ``FAIL_GRACE``, in seconds, mirrored here.
#:
#: A provider that hands a claimed job back inside this window of its claim pays
#: no penalty — the client is refunded immediately and nobody is punished for an
#: escrow key that is gone or a payload that will not open. The daemon holds no
#: RPC, so it cannot read the contract's constant; it is duplicated across a
#: language boundary and the e2e suite asserts the two agree, which is the one
#: test that fails when they drift.
FAIL_GRACE_SECONDS = 300

#: The most bytes of sealed payload this daemon will run per declared input unit.
#:
#: ``units_in`` is the client's own count and the chain bills the input leg at
#: whatever it says, so without a floor a bid can declare one unit for a megabyte
#: and buy the work for one atomic unit — on an embedding model, where the output
#: leg settles at zero, that is the whole bill.
#:
#: The shipped client SDKs declare one unit per four bytes of canonical JSON. This
#: default is deliberately four times looser: the daemon does not reproduce their
#: formula and must never decline an honest bid, so the gap is the margin that
#: absorbs everything it cannot see from outside a sealed container. Lower it to
#: tighten the floor; ``0`` turns the check off, which is the escape hatch if a
#: client ever trips it.
MAX_INPUT_BYTES_PER_UNIT = 16


@dataclass(frozen=True)
class ProviderConfig:
    wallet_key: str | None
    api_url: str
    #: What the daemon requests from the network at boot; the network grants
    #: effective capacity. Derived by the loader — the sum of what the model
    #: entries may hold — unless the operator sets it.
    capacity: int
    box_key: str | None              # Curve25519 payload-decryption key: required, except in
                                     # confidential mode where it is ephemeral per boot
    #: Where ``POST /release`` lives. Defaults to ``api_url``: the coordinator
    #: node serves the escrow today, and an operator who runs a separate escrow
    #: deployment points this at it. It is a URL the daemon must reach, never a
    #: second identity — the same wallet signs the release request.
    escrow_url: str | None = None
    metrics_port: int = 9090
    log_level: str = "info"
    #: Must stay well under the coordinator's lease window (20 s by default):
    #: a lease the daemon does not see before it lapses is a job it forfeits.
    poll_interval_s: float = 5
    safety_margin_s: float = 60
    #: The window after a claim in which handing the job back is penalty-free.
    #: See :data:`FAIL_GRACE_SECONDS` — this is a mirror of a chain constant, so
    #: an operator who lowers it only makes their own daemon more cautious.
    fail_grace_s: float = 300
    #: How many bytes of sealed payload one declared ``units_in`` may buy before
    #: this daemon declines the bid. See :data:`MAX_INPUT_BYTES_PER_UNIT`; ``0``
    #: disables the check.
    max_input_bytes_per_unit: int = MAX_INPUT_BYTES_PER_UNIT
    #: Where the handles of jobs in flight at an async backend are kept, so a
    #: restart resumes them instead of submitting them again. The loader
    #: defaults it to a file in the working directory (a nulled key too);
    #: ``None`` (a config built in code) keeps them in memory only.
    state_db: str | None = None
    #: The private acceptance floor. The default instance is inert: no discount,
    #: no tolerance, so the configured rates are the floor exactly as before.
    pricing: PricingConfig = field(default_factory=PricingConfig)
    #: Extra narrowing the operator wants on the open-book poll, validated by
    #: :func:`_build_bid_filter`. Empty by default: the daemon then asks for
    #: whatever its own floor and free capacity allow, and nothing further.
    bid_filter: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VorqdConfig:
    provider: ProviderConfig
    models: list[ModelConfig]

    @property
    def confidential(self) -> bool:
        return any(m.confidential for m in self.models)


# --------------------------------------------------------------------------- #
# env: indirection
# --------------------------------------------------------------------------- #


def resolve_env(value):
    """Recursively expand ``env:NAME`` references (whole or embedded).

    Raises :class:`ConfigError` naming the first variable that is unset.
    """
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ConfigError(
                    f"environment variable {name!r} referenced in config is not set"
                )
            return os.environ[name]

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


#: Where the daemon looks when started with neither `--config` nor `VORQD_CONFIG`:
#: the path the container image documents for a mounted file.
DEFAULT_CONFIG_PATH = "/etc/vorqd/vorqd.yaml"
#: The VORQ coordinator, when the config names none.
DEFAULT_API_URL = "https://api.vorq.co"


def load_config(path) -> VorqdConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {path}")
    return load_config_text(p.read_text())


def load_config_text(text: str) -> VorqdConfig:
    """The document itself, as `VORQD_CONFIG` carries it where no file can be mounted."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    raw = resolve_env(raw)  # fail fast on any missing env var
    models = _build_models(raw.get("models"))
    confidential = any(m.confidential for m in models)
    return VorqdConfig(
        provider=_build_provider(raw.get("provider"), confidential=confidential, models=models),
        models=models,
    )


def _require(d: dict, key: str, where: str):
    if key not in d or d[key] is None:
        raise ConfigError(f"{where}: missing required field '{key}'")
    return d[key]


def _build_provider(d, *, confidential: bool, models) -> ProviderConfig:
    if not isinstance(d, dict):
        raise ConfigError("config: missing 'provider' block")
    capacity = d.get("capacity")
    if isinstance(capacity, str) and capacity.strip().isdigit():
        capacity = int(capacity)          # `env:` resolves to a string
    if capacity is None:
        capacity = _derived_capacity(models)
    elif isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ConfigError("provider: 'capacity' must be a positive integer")
    if confidential and d.get("box_key") is not None:
        raise ConfigError(
            "provider: in confidential mode the box key is ephemeral — generated in "
            "guest memory each boot, never configured; remove 'box_key'"
        )
    box_key = d.get("box_key") if confidential else _require(d, "box_key", "provider")
    bpu = d.get("max_input_bytes_per_unit", MAX_INPUT_BYTES_PER_UNIT)
    if isinstance(bpu, str) and bpu.strip().isdigit():
        bpu = int(bpu)                    # `env:` resolves to a string
    if isinstance(bpu, bool) or not isinstance(bpu, int) or bpu < 0:
        # Zero is legal and means "off"; a negative would invert the floor into a
        # bonus, so it is refused rather than clamped.
        raise ConfigError("provider: 'max_input_bytes_per_unit' must be a non-negative integer")
    return ProviderConfig(
        # A `provider.id` key, if present, is silently ignored: the daemon's id is
        # admin-issued and discovered at the session handshake, never configured.
        wallet_key=d.get("wallet_key"),  # optional: a fresh wallet is generated when omitted
        box_key=box_key,  # operator-held Curve25519 identity; ephemeral in confidential mode
        api_url=d.get("api_url") or DEFAULT_API_URL,
        escrow_url=d.get("escrow_url"),
        capacity=capacity,
        metrics_port=int(d.get("metrics_port", 9090)),
        log_level=d.get("log_level", "info"),
        poll_interval_s=float(d.get("poll_interval_s", 5)),
        safety_margin_s=float(d.get("safety_margin_s", 60)),
        fail_grace_s=float(d.get("fail_grace_s", FAIL_GRACE_SECONDS)),
        max_input_bytes_per_unit=bpu,
        state_db=str(d.get("state_db") or "vorqd-state.sqlite"),
        pricing=_build_pricing(d.get("pricing")),
        bid_filter=_build_bid_filter(d.get("bid_filter")),
    )


def _derived_capacity(models) -> int:
    """The sum of what the entries may hold — a `provider.capacity` left unset.

    Each entry's share is its `rate_limit` window at its shortest SLA, else its
    `concurrency`. An entry with neither could hold anything, so the operator
    has to say.
    """
    total = 0
    for m in models:
        share = Throttle.from_model(m).capacity
        if share is None:
            raise ConfigError(
                f"provider: 'capacity' is required — model {m.model!r} declares neither a "
                "'concurrency' nor a 'rate_limit' window at its SLA, so what it may hold "
                "cannot be derived"
            )
        total += share
    return max(1, total)


_PRICING_KEYS = frozenset({"max_discount_pct", "bid_tolerance_pct", "low_load_pct"})


def _build_pricing(d) -> PricingConfig:
    if d is None:
        return PricingConfig()
    if not isinstance(d, dict):
        raise ConfigError("provider.pricing: must be a mapping")
    unknown = sorted(set(d) - _PRICING_KEYS)
    if unknown:
        raise ConfigError(
            f"provider.pricing: unknown key(s): {', '.join(repr(k) for k in unknown)}"
        )
    values = {}
    for key in ("max_discount_pct", "bid_tolerance_pct"):
        value = int(d.get(key, 0))
        if not 0 <= value <= MAX_FLOOR_DISCOUNT_PCT:
            raise ConfigError(
                f"provider.pricing: {key} must be between 0 and {MAX_FLOOR_DISCOUNT_PCT} "
                f"(got {value}). A floor discounted any further would accept a bid that "
                "pays nothing"
            )
        values[key] = value
    low = int(d.get("low_load_pct", 30))
    if not 0 <= low <= 100:
        raise ConfigError(f"provider.pricing: low_load_pct must be between 0 and 100 (got {low})")
    return PricingConfig(
        max_discount_pct=values["max_discount_pct"],
        bid_tolerance_pct=values["bid_tolerance_pct"],
        low_load_pct=low,
    )


_BID_FILTER_KEYS = frozenset({"min_age_s"})


def _build_bid_filter(d) -> dict:
    """Operator-set narrowing on the open-book poll.

    A validated passthrough rather than a raw one: an unknown key is a typo the
    loader names, not a parameter silently forwarded to a node that will ignore
    it — and never a way to override the filters the scheduler computes itself.
    ``min_rate_out``, ``min_rate_in`` and ``limit`` carry the private floor and
    this daemon's free capacity; they are decisions taken per sweep, not
    settings, so they are unknown keys here like any other.

    ``min_age_s`` stays in **seconds** all the way to the scheduler. Bid age on
    this book is a block count, and neither the block time nor the block the
    last listing was answered at is knowable from a config file.
    """
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ConfigError("provider.bid_filter: must be a mapping")
    unknown = sorted(set(d) - _BID_FILTER_KEYS)
    if unknown:
        raise ConfigError(
            f"provider.bid_filter: unknown key(s): {', '.join(repr(k) for k in unknown)}"
        )
    out: dict = {}
    if d.get("min_age_s") is not None:
        min_age_s = int(d["min_age_s"])
        if min_age_s < 0:
            raise ConfigError(
                f"provider.bid_filter: min_age_s must not be negative (got {min_age_s})"
            )
        out["min_age_s"] = min_age_s
    return out


def _build_models(items) -> list[ModelConfig]:
    if not isinstance(items, list) or not items:
        raise ConfigError("config: 'models' must be a non-empty list")
    return [_build_model(m) for m in items]


def _build_model(d) -> ModelConfig:
    if not isinstance(d, dict):
        raise ConfigError("models: each entry must be a mapping")
    name = _require(d, "model", "model")
    # `load:` belongs to the model, beside `backend:`, not inside it. Refused
    # here because the alternative is silence: `backend:` forwards keys it does
    # not recognise to the runtime as request params, deliberately — that is how
    # a preset carries a nonstandard knob — so a `load:` nested one level too
    # deep loads without complaint, publishes, claims, and simply never
    # discounts. Every gauge and log line looks exactly as it does when the
    # feature is off, because it is off. One key, named, rather than a blanket
    # unknown-key rejection that would break the open params dict.
    backend_block = d.get("backend")
    if isinstance(backend_block, dict) and "load" in backend_block:
        raise ConfigError(
            f"model {name!r}: 'load' is a model-level key, a sibling of 'backend' "
            f"and not a member of it — move the block out one level, or the load "
            f"probe is silently ignored and the acceptance floor never moves"
        )
    # The listing name is free-form: only this flag makes a model confidential,
    # and it changes nothing about the backend — that is the daemon's attested
    # identity, published at startup.
    confidential = bool(d.get("confidential", False))
    backend = _build_backend(d.get("backend"), name)
    modality = d.get("modality")
    if modality is not None and modality not in ("text", "image", "video", "embedding"):
        raise ConfigError(
            f"model {name!r}: 'modality' must be one of text, image, video, embedding "
            f"(got {modality!r})"
        )
    slas = _build_slas(d.get("slas"), name)
    _check_hold_pace(backend, slas, name)
    return ModelConfig(
        model=name,
        slas=slas,
        backend=backend,
        confidential=confidential,
        modality=modality,
        load=_build_load(d.get("load"), name, backend=backend),
        sla_backends=_build_sla_backends(d.get("sla_backends"), backend_block, name, slas),
    )


def _check_hold_pace(backend: BackendConfig, slas: dict, model: str) -> None:
    """A day's budget the minute window cannot start inside the day is a
    promise the daemon would break: the window at the shortest SLA is what the
    entry may hold, so every shorter window must be able to start that many
    attempts inside the SLA. Refused at load, naming both windows."""
    if not slas or not backend.rate_limit:
        return
    sla_w = min(slas, key=window_seconds)
    sla_s = window_seconds(sla_w)
    hold = backend.rate_limit.get(sla_w)
    if hold is None:
        # Any window of the same length spelled differently counts too.
        hold = next((n for w, n in backend.rate_limit.items() if window_seconds(w) == sla_s), None)
    if hold is None:
        return
    for w, n in backend.rate_limit.items():
        secs = window_seconds(w)
        if secs < sla_s and n * (sla_s / secs) < hold:
            raise ConfigError(
                f"model {model!r} backend: rate_limit[{sla_w!r}] of {hold} is what the entry "
                f"may hold, but rate_limit[{w!r}] of {n} starts at most "
                f"{int(n * sla_s // secs)} inside {sla_w!r} — lower the budget or raise the pace"
            )


def _build_sla_backends(d, base, model: str, slas: dict) -> dict[str, BackendConfig]:
    """One backend per SLA window, each a patch over the model's `backend`.

    A patch is merged key by key — its keys win, a `null` removes a key — and the
    result is validated as a backend of its own. Naming a `preset` drops an
    inherited `request`/`response` and vice versa, so a window may change the
    shape, not only the params. The limit keys stay per model: the throttle and
    the breaker guard the model's asks as one, so a limit on a patch is refused
    rather than read and ignored.
    """
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ConfigError(
            f"model {model!r}: 'sla_backends' must be a mapping of SLA window to backend patch"
        )
    out: dict[str, BackendConfig] = {}
    for window, patch in d.items():
        if str(window) not in slas:
            raise ConfigError(
                f"model {model!r} sla_backends: {str(window)!r} is not one of this model's slas"
            )
        if not isinstance(patch, dict):
            raise ConfigError(f"model {model!r} sla_backends {str(window)!r}: must be a mapping")
        limits = sorted(set(patch) & _LIMIT_KEYS)
        if limits:
            raise ConfigError(
                f"model {model!r} sla_backends {str(window)!r}: {', '.join(limits)} are "
                f"per-model limits and belong on 'backend'"
            )
        merged = dict(base)
        if "preset" in patch:
            merged.pop("request", None)
            merged.pop("response", None)
        if "request" in patch:
            merged.pop("preset", None)
        for key, value in patch.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        out[str(window)] = _build_backend(merged, f"{model} [{window}]")
    return out


#: The most fraction digits a configured rate may carry at load. The token's own
#: ``decimals`` bound it exactly once the chain context is read; no token has more.
MAX_RATE_DECIMALS = 18


def _check_usd_rate(value, model: str, window: str, field: str) -> None:
    """A configured rate is a quoted USD decimal string, settled here at load.

    An unquoted YAML number is refused: a float cannot say which decimal was
    meant, and an integer reads as dollars where the old atomic spelling meant
    micro-units. ``rate_in`` is optional: a side the model does not meter is
    simply absent.
    """
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(
            f"model {model!r} sla {window!r}: {field}={value!r} must be a quoted USD decimal "
            f"string, USD per 1M units of work (e.g. {field}: \"0.16\")"
        )
    try:
        parse_usd(value, MAX_RATE_DECIMALS)
    except ValueError as exc:
        raise ConfigError(f"model {model!r} sla {window!r}: {field}: {exc}") from None


def _build_slas(d, model: str) -> dict[str, SlaRate]:
    if not isinstance(d, dict) or not d:
        raise ConfigError(f"model {model!r}: 'slas' must be a non-empty mapping")
    out: dict[str, SlaRate] = {}
    for window, rates in d.items():
        if not isinstance(rates, dict) or "rate_out" not in rates:
            raise ConfigError(
                f"model {model!r} sla {window!r}: 'rate_out' is required on every window"
            )
        for field in ("rate_in", "rate_out"):
            _check_usd_rate(rates.get(field), model, str(window), field)
        out[str(window)] = SlaRate(rate_in=rates.get("rate_in"), rate_out=rates["rate_out"])
    return out


_LOAD_KEYS = frozenset({"url", "metric", "scale", "source"})


def _build_load(d, model: str, *, backend=None) -> LoadProbeConfig | None:
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ConfigError(f"model {model!r}: 'load' must be a mapping")
    unknown = sorted(set(d) - _LOAD_KEYS)
    if unknown:
        raise ConfigError(
            f"model {model!r} load: unknown key(s): {', '.join(repr(k) for k in unknown)}"
        )
    source = str(d.get("source", "probe"))
    if source not in ("probe", "occupancy"):
        raise ConfigError(
            f"model {model!r} load: 'source' must be 'probe' or 'occupancy' (got {source!r})"
        )
    if source == "occupancy":
        if set(d) & {"url", "metric", "scale"}:
            raise ConfigError(
                f"model {model!r} load: 'source: occupancy' reads the entry's own limits and "
                "takes no 'url', 'metric' or 'scale'"
            )
        if backend is not None and backend.concurrency is None and not backend.rate_limit:
            raise ConfigError(
                f"model {model!r} load: 'source: occupancy' needs a 'concurrency' or a "
                "'rate_limit' on the backend — an unbounded entry has no fullness to report"
            )
        return LoadProbeConfig(source="occupancy")
    url = str(_require(d, "url", f"model {model!r} load"))
    if not url.startswith(("http://", "https://")):
        raise ConfigError(
            f"model {model!r} load: 'url' must be an absolute URL (got {url!r})"
        )
    metric = str(_require(d, "metric", f"model {model!r} load"))
    scale = float(d.get("scale", 1.0))
    if scale <= 0:
        raise ConfigError(f"model {model!r} load: 'scale' must be positive (got {scale})")
    return LoadProbeConfig(url=url, metric=metric, scale=scale)


# The keys every `backend:` shape shares, whatever it points at: how many
# attempts a job gets and how they are spaced, and how much of the backend the
# daemon may use at once. They are read here, never forwarded as params.
_LIMIT_KEYS = frozenset({
    "retries", "retry_statuses", "retry_backoff_s", "retry_backoff_max_s", "timeout_s",
    "concurrency", "rate_limit", "trip_after", "trip_cooldown_s",
})


def _build_limits(d: dict, model: str) -> dict:
    """The limit fields of a `backend:` block, validated, as `BackendConfig` kwargs.

    Every field is optional. A backend with a request quota — so many requests
    per rolling window, so many in flight, a ceiling on one request's wall time —
    is described here, and the scheduler keeps the daemon inside those numbers
    before it claims: the poll offers the coordinator no more `free` than the
    tightest of them allows.
    """
    where = f"model {model!r} backend"

    def non_negative_int(key: str, default: int) -> int:
        value = d.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"{where}: '{key}' must be a non-negative integer")
        return value

    def positive_number(key: str, default):
        value = d.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{where}: '{key}' must be a positive number")
        return float(value)

    retries = non_negative_int("retries", 0)
    raw_statuses = d.get("retry_statuses") or []
    if not isinstance(raw_statuses, list) or any(
        isinstance(v, bool) or not isinstance(v, int) or not 400 <= v <= 499 for v in raw_statuses
    ):
        raise ConfigError(
            f"{where}: 'retry_statuses' must be a list of 4xx status codes — 429 and every 5xx "
            "are retried already"
        )
    backoff = positive_number("retry_backoff_s", 30.0)
    backoff_max = positive_number("retry_backoff_max_s", 900.0)
    if backoff_max < backoff:
        raise ConfigError(f"{where}: 'retry_backoff_max_s' must be at least 'retry_backoff_s'")
    timeout = positive_number("timeout_s", None)
    concurrency = d.get("concurrency")
    if concurrency is not None and (
        isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1
    ):
        raise ConfigError(f"{where}: 'concurrency' must be a positive integer")
    if "queue" in d:
        raise ConfigError(
            f"{where}: 'queue' is not a key — what the entry may hold is its 'rate_limit' "
            "window at its shortest SLA (a day's budget on a '24h' entry), started "
            "'concurrency' at a time"
        )
    raw_limit = d.get("rate_limit") or {}
    if not isinstance(raw_limit, dict):
        raise ConfigError(f"{where}: 'rate_limit' must be a mapping of window to a request count")
    rate_limit: dict[str, int] = {}
    for window, count in raw_limit.items():
        try:
            window_seconds(str(window))
        except ValueError as exc:
            raise ConfigError(f"{where}: rate_limit {exc}") from None
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ConfigError(
                f"{where}: rate_limit[{window!r}] must be a positive integer request count"
            )
        rate_limit[str(window)] = count
    return {
        "retries": retries,
        "retry_statuses": tuple(sorted(set(raw_statuses))),
        "retry_backoff_s": backoff,
        "retry_backoff_max_s": backoff_max,
        "timeout_s": timeout,
        "concurrency": concurrency,
        "rate_limit": rate_limit,
        "trip_after": non_negative_int("trip_after", 3),
        "trip_cooldown_s": positive_number("trip_cooldown_s", 60.0),
    }


def _build_backend(d, model: str) -> BackendConfig:
    if not isinstance(d, dict):
        raise ConfigError(f"model {model!r}: missing 'backend' block")

    has_preset = "preset" in d
    has_request = "request" in d
    if has_preset == has_request:  # both or neither
        raise ConfigError(
            f"model {model!r} backend: must contain exactly one of 'preset' or 'request'"
        )

    limits = _build_limits(d, model)
    if has_preset:
        stream = d.get("stream", False)
        if not isinstance(stream, bool):
            raise ConfigError(f"model {model!r} backend: 'stream' must be true or false")
        if stream and d["preset"] != "openai-chat":
            raise ConfigError(
                f"model {model!r} backend: 'stream' applies to the openai-chat preset only"
            )
        params = {k: v for k, v in d.items()
                  if k not in ("preset", "stream") and k not in _LIMIT_KEYS}
        _validate_param_map(params.get("param_map"), model)
        _validate_param_caps(params.get("param_caps"), model)
        _validate_reference(params.get("reference"), model)
        _validate_durations(params.get("durations"), model)
        _validate_render_policy(params, model)
        _validate_param_list(params.get("params_supported"), "params_supported", model)
        _validate_preset_headers(params.get("headers"), d["preset"], model)
        _resolve_reasoning_params(params, model)
        if d["preset"] == "openai-responses":
            tier = params.get("service_tier")
            if tier is not None and tier not in ("flex", "priority"):
                raise ConfigError(
                    f"model {model!r} backend: 'service_tier' must be 'flex' or 'priority' "
                    f"(got {tier!r})"
                )
            if "max_polls" in params:
                _validate_poll_resume({"max_polls": params["max_polls"]}, model)
        if d["preset"] == "openai-batch":
            window = params.get("completion_window", "24h")
            if not isinstance(window, str) or not window.strip():
                raise ConfigError(
                    f"model {model!r} backend: 'completion_window' must be a window string "
                    f"such as \"24h\" (got {window!r})"
                )
            endpoint = params.get("endpoint", "/v1/chat/completions")
            if not isinstance(endpoint, str) or not endpoint.startswith("/"):
                raise ConfigError(
                    f"model {model!r} backend: 'endpoint' must be the request path each batch "
                    f"line names, e.g. \"/v1/chat/completions\" (got {endpoint!r})"
                )
            if "max_polls" in params:
                _validate_poll_resume({"max_polls": params["max_polls"]}, model)
        return BackendConfig(
            preset=str(d["preset"]),
            params=params,
            health=d.get("health"),
            stream=stream,
            **limits,
        )

    if "stream" in d:
        raise ConfigError(f"model {model!r} backend: 'stream' applies to the openai-chat preset only")
    response = d.get("response")
    _validate_raw_response(response, model)
    # Raw mappings carry no preset params, but `param_caps` applies to them the
    # same way — media jobs (steps/fps ceilings, the caps' main use) are served
    # by raw backends, so the caps must survive this branch too.
    raw_params: dict = {}
    if "param_caps" in d:
        _validate_param_caps(d["param_caps"], model)
        raw_params["param_caps"] = d["param_caps"]
    if "reference" in d:
        # Same reason as `param_caps`: reference-conditioned media is served by
        # raw mappings, so the policy has to survive this branch too.
        _validate_reference(d["reference"], model)
        raw_params["reference"] = d["reference"]
    if "durations" in d:
        _validate_durations(d["durations"], model)
        raw_params["durations"] = d["durations"]
    _validate_render_policy(d, model)
    for key in _RENDER_POLICY_KEYS:
        if key in d:
            raw_params[key] = d[key]
    _validate_prepare(d["request"], model)
    return BackendConfig(
        preset=None,
        params=raw_params,
        request=d["request"],
        response=response,
        health=d.get("health"),
        **limits,
    )


#: The kinds of reference asset a backend may bound, as the request's keys sort them.
_REFERENCE_KINDS = ("still", "clip", "audio")

#: Backend keys that say what a media backend renders, beside ``durations``.
_RENDER_POLICY_KEYS = ("resolutions", "auto_duration", "adaptive_aspect")


def _validate_render_policy(d: dict, model: str) -> None:
    """``resolutions``, ``auto_duration`` and ``adaptive_aspect``.

    ``resolutions`` is checked against the shared table — a tier the network does
    not price can never arrive. The other two are the backend's *own* spelling of
    "the model chooses", so any scalar is a legitimate value and only a container
    is a mistake.
    """
    from . import media_units

    tiers = d.get("resolutions")
    if tiers is not None and (not isinstance(tiers, list) or not tiers
                              or any(t not in media_units.RESOLUTIONS for t in tiers)):
        raise ConfigError(
            f"model {model!r} backend: 'resolutions' must be a non-empty list drawn from "
            f"{', '.join(media_units.RESOLUTIONS)} (got {tiers!r})")
    for key in ("auto_duration", "adaptive_aspect"):
        value = d.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            raise ConfigError(
                f"model {model!r} backend: {key!r} is the value this backend is sent when the "
                f"model is to choose — a string or a whole number (got {value!r})")


_PREPARE_KEYS = {"name", "when", "for_each", "method", "url", "headers", "body", "extract"}


def _validate_prepare(request, model: str) -> None:
    """``request.prepare``: the requests a raw submit depends on.

    Checked at load because every mistake here otherwise surfaces on a claimed
    job: a step with nothing to extract contributes nothing the submit can name,
    and a name that is not a plain identifier cannot be addressed as
    ``{prepare.<name>.<key>}`` at all.
    """
    steps = request.get("prepare") if isinstance(request, dict) else None
    if steps is None:
        return
    where = f"model {model!r} request.prepare"
    if not isinstance(steps, list) or not steps:
        raise ConfigError(f"{where}: must be a non-empty list of steps")
    seen: set = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ConfigError(f"{where}: each step must be a mapping")
        unknown = set(step) - _PREPARE_KEYS
        if unknown:
            raise ConfigError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")
        name = step.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ConfigError(f"{where}: each step needs a 'name' that is a plain identifier "
                              f"(got {name!r})")
        if name in seen:
            raise ConfigError(f"{where}: step name {name!r} is used twice")
        seen.add(name)
        if not isinstance(step.get("url"), str) or not step["url"]:
            raise ConfigError(f"{where}: step {name!r} needs a 'url'")
        if "for_each" in step:
            if "when" in step or not isinstance(step["for_each"], str) or not step["for_each"].strip():
                raise ConfigError(f"{where}: step {name!r} 'for_each' must be a dot-path to a "
                                  f"list such as input.reference_images, and replaces 'when'")
        if "when" in step and (not isinstance(step["when"], str) or not step["when"].strip()):
            raise ConfigError(f"{where}: step {name!r} 'when' must be a dot-path such as "
                              f"input.image")
        pulled = step.get("extract")
        if (not isinstance(pulled, dict) or not pulled
                or any(not isinstance(k, str) or not k.isidentifier()
                       or not isinstance(v, str) or not v.startswith("$")
                       for k, v in pulled.items())):
            raise ConfigError(f"{where}: step {name!r} needs an 'extract' mapping of "
                              f"name -> '$.' JSONPath into its answer")


def _validate_durations(durations, model: str) -> None:
    """The clip lengths a backend renders: a non-empty list of positive whole seconds.

    Checked at load for the same reason as ``reference``: a list the clamp cannot
    compare against would fail every video job after its claim.
    """
    if durations is None:
        return
    if (not isinstance(durations, list) or not durations
            or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in durations)):
        raise ConfigError(
            f"model {model!r} backend: 'durations' must be a non-empty list of positive "
            f"whole seconds (got {durations!r}); omit it for a backend that renders any length"
        )


def _validate_reference(block, model: str) -> None:
    """The operator's per-backend reference policy: one key, and it must be real.

    Checked at load rather than on the first job that carries a reference. An
    accept list naming a type this daemon has no reader for would take every such
    job to a claim and then refuse it — a cost the client and the chain both pay
    for a typo.
    """
    from .backend import REFERENCE_BOUNDS
    from .media import ACCEPTABLE_TYPES

    if block is None:
        return
    if not isinstance(block, dict):
        raise ConfigError(f"model {model!r} backend: 'reference' must be a mapping")
    unknown = set(block) - {"accept", *_REFERENCE_KINDS}
    if unknown:
        raise ConfigError(
            f"model {model!r} backend: unknown key(s) in 'reference': "
            f"{', '.join(sorted(unknown))}"
        )
    accept = block.get("accept")
    if not isinstance(accept, list) or not accept:
        # `accept: []` is a backend that takes no reference at all, which omitting
        # the block already says — and reading it as "no opinion" would silently
        # mean the opposite of how it reads.
        raise ConfigError(
            f"model {model!r} backend: 'reference.accept' must be a non-empty list "
            f"of media types; omit the block entirely to accept everything readable"
        )
    for entry in accept:
        if not isinstance(entry, str) or entry.strip().lower() not in ACCEPTABLE_TYPES:
            raise ConfigError(
                f"model {model!r} backend: 'reference.accept' names {entry!r}, which this "
                f"daemon cannot read; it reads {', '.join(ACCEPTABLE_TYPES)}"
            )
    for kind in _REFERENCE_KINDS:
        limits = block.get(kind)
        if limits is None:
            continue
        if not isinstance(limits, dict) or not limits:
            raise ConfigError(f"model {model!r} backend: 'reference.{kind}' must be a mapping of bounds")
        for bound, value in limits.items():
            if bound not in REFERENCE_BOUNDS:
                raise ConfigError(
                    f"model {model!r} backend: 'reference.{kind}' has no bound {bound!r}; "
                    f"it takes {', '.join(REFERENCE_BOUNDS)}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(
                    f"model {model!r} backend: 'reference.{kind}.{bound}' must be a "
                    f"non-negative number (got {value!r})")


def _resolve_reasoning_params(params: dict, model: str) -> None:
    """Resolve ``reasoning: <preset>`` + inline param_map into the single effective
    map stored on the config — the driver never sees preset names. A declared
    ``default_effort`` is split out first and carried beside the map: it is applied
    to the payload, not to a key. Mutates ``params`` in place.
    """
    reasoning = params.pop("reasoning", None)
    if reasoning is None:
        return
    try:
        reasoning, default_effort = extract_default_effort(reasoning)
        params["param_map"] = resolve_reasoning(reasoning, params.get("param_map"))
    except ValueError as exc:
        raise ConfigError(f"model {model!r} backend: {exc}") from exc
    if default_effort is not None:
        params["default_effort"] = default_effort


def _validate_param_map(pm, model: str) -> None:
    """A ``param_map`` entry is a str (rename, dotted for nesting) or an object
    of ``{to, values, unmapped}`` — the value-mapping form (mapping-reference §4)."""
    if pm is None:
        return
    if not isinstance(pm, dict):
        raise ConfigError(f"model {model!r} backend: 'param_map' must be a mapping")
    for key, spec in pm.items():
        if not isinstance(key, str):
            raise ConfigError(f"model {model!r} backend: 'param_map' keys must be strings")
        if isinstance(spec, str):
            continue
        if not isinstance(spec, dict):
            raise ConfigError(
                f"model {model!r} backend: 'param_map' entry {key!r} must be a string "
                f"rename or an object of to/values/unmapped"
            )
        unknown = set(spec) - {"to", "values", "unmapped"}
        if unknown:
            raise ConfigError(
                f"model {model!r} backend: 'param_map' entry {key!r} has unknown "
                f"key(s): {', '.join(sorted(unknown))}"
            )
        if "to" in spec and not isinstance(spec["to"], str):
            raise ConfigError(f"model {model!r} backend: 'param_map' entry {key!r}: 'to' must be a string")
        if "values" in spec and not isinstance(spec["values"], dict):
            raise ConfigError(f"model {model!r} backend: 'param_map' entry {key!r}: 'values' must be a mapping")
        if spec.get("unmapped", "pass") not in ("pass", "drop"):
            raise ConfigError(
                f"model {model!r} backend: 'param_map' entry {key!r}: 'unmapped' must be 'pass' or 'drop'"
            )


def _validate_param_caps(caps, model: str) -> None:
    if caps is None:
        return
    if not isinstance(caps, dict) or not all(
        isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
        for k, v in caps.items()
    ):
        raise ConfigError(f"model {model!r} backend: 'param_caps' must be a mapping of str -> number")


def _validate_param_list(value, key: str, model: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"model {model!r} backend: {key!r} must be a list of param names")


def _validate_preset_headers(value, preset: str, model: str) -> None:
    """Extra submit headers, templated per job and merged over the preset's own."""
    if value is None:
        return
    if preset == "openai-batch":
        raise ConfigError(f"model {model!r} backend: 'headers' does not apply to openai-batch")
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ConfigError(f"model {model!r} backend: 'headers' must be a mapping of str -> str")


def _validate_raw_response(response, model: str) -> None:
    if not isinstance(response, dict):
        raise ConfigError(f"model {model!r} backend: raw mapping requires a 'response' block")

    ok = response.get("ok")
    if ok is not None:
        if (not isinstance(ok, dict) or set(ok) - {"field", "values", "retry", "message"}
                or not isinstance(ok.get("field"), str) or not ok["field"].startswith("$")
                or not isinstance(ok.get("values"), list) or not ok["values"]
                or not isinstance(ok.get("retry", []), list)
                or not str(ok.get("message", "$")).startswith("$")):
            raise ConfigError(
                f"model {model!r} response.ok: needs 'field' (a '$.' JSONPath into every answer) "
                f"and a non-empty 'values' list meaning success; optionally 'retry' (values "
                f"worth another attempt) and 'message' (a '$.' JSONPath, logged only)")
    for key in ("failure_code", "failure_message"):
        path = (response.get("poll") or {}).get(key)
        if path is not None and (not isinstance(path, str) or not path.startswith("$")):
            raise ConfigError(f"model {model!r} response.poll: {key!r} must be a '$.' JSONPath")

    result = response.get("result")
    if not isinstance(result, dict):
        raise ConfigError(f"model {model!r} response: missing 'result' block")
    present = [k for k in _RESULT_KINDS if k in result]
    if len(present) != 1:
        raise ConfigError(
            f"model {model!r} response.result: must contain exactly one of "
            f"{', '.join(_RESULT_KINDS)} (found: {present or 'none'})"
        )
    # A text job is billed by the backend's reported output-token count, so that
    # count MUST be mapped — a text result without 'completion_tokens' has no
    # billable quantity and is rejected at load (the openai-chat preset maps it
    # implicitly, so preset configs never hit this).
    if "text" in result and "completion_tokens" not in result:
        raise ConfigError(
            f"model {model!r} response.result: a 'text' result must also map "
            f"'completion_tokens' (the billed output-token count)"
        )

    mode = response.get("mode")
    if mode not in ("sync", "poll"):
        raise ConfigError(f"model {model!r} response.mode: must be 'sync' or 'poll'")
    if mode == "poll":
        poll = response.get("poll")
        if not isinstance(poll, dict):
            raise ConfigError(f"model {model!r} response: 'poll' is required when mode is 'poll'")
        poll_present = [k for k in ("status_url", "request") if k in poll]
        if len(poll_present) != 1:
            raise ConfigError(
                f"model {model!r} response.poll: must contain exactly one of "
                f"'status_url' or 'request' (found: {poll_present or 'none'})"
            )
        _validate_poll_resume(poll, model)


def _validate_poll_resume(poll: dict, model: str) -> None:
    """The two keys that make a poll resumable across a restart.

    ``handle`` names the submit-response field that identifies the job at the
    backend; it is only usable with a templated ``request`` (a resumed job has
    no submit response to pull a ``status_url`` from). ``max_polls`` sets the
    tick cadence as a share of the SLA window.
    """
    if "handle" in poll:
        handle = poll["handle"]
        if not isinstance(handle, str) or not handle.startswith("$"):
            raise ConfigError(
                f"model {model!r} response.poll: 'handle' must be a JSONPath into the submit "
                f"response, e.g. \"$.id\""
            )
        if "request" not in poll:
            raise ConfigError(
                f"model {model!r} response.poll: 'handle' needs a templated 'request' (the "
                f"handle is in scope there as {{handle}}); a 'status_url' cannot be resumed"
            )
        if _templates_submit(poll["request"]):
            raise ConfigError(
                f"model {model!r} response.poll: a resumable poll request cannot use "
                f"{{submit.*}} — a resumed job has no submit response; use {{handle}}"
            )
    elif "request" in poll and _templates_handle(poll["request"]):
        # The request names the handle and nothing declares one, so the render
        # would substitute an empty string and poll the backend's job list
        # forever. A poll that cannot be resumed at all is a valid mapping; one
        # that spells the resume and never records it is not.
        raise ConfigError(
            f"model {model!r} response.poll: the poll request uses {{handle}} but 'handle' "
            f"names no submit-response field"
        )
    if "max_polls" in poll:
        n = poll["max_polls"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ConfigError(f"model {model!r} response.poll: 'max_polls' must be a positive integer")


#: ``{submit.id}`` as the template engine actually reads it: whitespace inside
#: the braces is stripped, so ``{ submit.id }`` is the same reference and a
#: literal substring match would let it through the resume guard below.
_SUBMIT_REF_RE = re.compile(r"\{\s*submit\.")

#: ``{handle}``, same reason.
_HANDLE_REF_RE = re.compile(r"\{\s*handle\s*\}")


def _templates_submit(node) -> bool:
    """Whether any string anywhere under ``node`` templates the submit response."""
    if isinstance(node, str):
        return _SUBMIT_REF_RE.search(node) is not None
    if isinstance(node, dict):
        return any(_templates_submit(v) for v in node.values())
    if isinstance(node, list):
        return any(_templates_submit(v) for v in node)
    return False


def _templates_handle(node) -> bool:
    """Whether any string anywhere under ``node`` references ``{handle}``."""
    if isinstance(node, str):
        return _HANDLE_REF_RE.search(node) is not None
    if isinstance(node, dict):
        return any(_templates_handle(v) for v in node.values())
    if isinstance(node, list):
        return any(_templates_handle(v) for v in node)
    return False
