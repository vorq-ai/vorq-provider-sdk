"""Exception hierarchy for vorqd."""

from __future__ import annotations


class VorqdError(Exception):
    """Base class for all vorqd errors."""


class ConfigError(VorqdError):
    """Raised when ``vorqd.yaml`` is missing, malformed, or violates a rule."""


class ChainConflict(VorqdError):
    """A ``409`` from the ``/evm/*`` surface.

    ``type`` carries the wire ``error.type`` (``state_conflict`` for a lost
    race / wrong-state transition, ``reclaim`` for a past-deadline settle).
    """

    def __init__(self, type_: str, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message or type_)
        self.type = type_
        self.code = code


class OpRejected(VorqdError):
    """A ``403`` from ``POST /evm/ops`` or ``PUT /evm/asks``.

    The signature recovered to nobody the registries know — a wrong key, a wrong
    domain, or a wallet that is not a registered provider. Never retryable: the
    identical bytes can only ever be rejected again.
    """


class OpRefused(VorqdError):
    """A ``409`` from ``POST /evm/ops`` — **the chain refused the op**.

    ``reason`` is the contract's own error name, decoded from the revert:
    ``NotOpen``, ``AtCapacity``, ``StaleOp``, ``NotDesignated`` and so on, or the
    literal ``"unknown"`` when the revert came from outside the two registries'
    ABIs (a payment-token failure inside ``claim``), in which case ``raw``
    carries the undecodable bytes and is ``None`` otherwise.

    Typed so no call site has to reach into a body: a refusal is a verdict on
    chain state as it stands, and re-sending the same op cannot change it.
    """

    def __init__(self, reason: str, *, raw: str | None = None, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.raw = raw


class UploadInvalid(VorqdError):
    """``POST /v1/files`` answered ``2xx`` with no ``vorq.cid`` in the body.

    The cid is the whole reference: a settle naming ``result_cid: null`` is a
    body the node answers ``result_required`` to, which reads as "this daemon
    forgot the result". Typed so the settle path can hand the job back like any
    other delivery failure instead of raising ``KeyError`` past every handler
    around it, which would leave the job claimed with no fail report.
    """


class UnknownModel(ConfigError):
    """A configured model name the coordinator's catalog does not carry.

    A :class:`ConfigError`, because it is one: the ops and the ask book address
    models by ``uint32`` id, the config addresses them by name, and a name with
    no id cannot be served, quoted or claimed against. The daemon refuses to
    start rather than skipping the model silently and running at a capacity the
    operator did not choose.
    """


class CatalogUnresolved(VorqdError):
    """A job listing was read before the model catalog was resolved.

    The job book answers ``model_id`` and ``sla_secs``; turning those back into
    the operator's own model name and SLA window needs the catalog, which is
    fetched once at startup. Raised rather than guessed: a job whose model the
    daemon cannot name is one it must not claim.
    """


class BackendError(VorqdError):
    """The operator's inference backend failed or timed out for a job.

    ``retryable`` says whether another attempt could succeed: a ``429``, a
    ``5xx`` or a transport fault (a timeout included) is a backend that may
    answer later; any other ``4xx`` is a request the backend will refuse again.
    ``retry_after_s`` carries a ``Retry-After`` the backend sent, in seconds.
    The scheduler owns the retry loop and reads both.
    """

    def __init__(self, message: str = "", *, retryable: bool = False,
                 retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class MediaInputRefused(BackendError):
    """The reference this job carries is not one this daemon may run for it.

    Its own kind of refusal, because it is the only one that is squarely the
    *client's* doing: the reference is larger than the units bought for it, or of
    a type this backend does not take, or not readable as the media it claims.
    Never retryable — another attempt reads the same bytes — and never the
    backend's fault, so the circuit breaker is untouched by it.

    The message reaches the network through ``_report_fail``, so it carries
    dimensions, counts and media types and never the reference itself.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message, retryable=False)


class BackendExhausted(BackendError):
    """Every configured attempt failed retryably; the job is given back."""


class BackendGone(BackendError):
    """The backend answered 404 or 410: what this entry points at is not there.

    Never retryable — the job is failed back at once — but it is the backend's
    fault and not the job's, so it counts toward the circuit breaker. A model
    the host has withdrawn, or whose gateway is away, answers every job this
    way; without the count its ask would stand and every claim would fail.
    An entry that knows its gateway comes and goes lists 404 in
    ``retry_statuses`` instead, and the retries are what it then exhausts.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message, retryable=False)


class DeadlineExceeded(BackendError):
    """The next wait or attempt would end past the job's SLA deadline."""


class NotRegisteredError(VorqdError):
    """A ``403 not_registered`` from ``POST /auth/session`` for a provider-role handshake.

    A retryable provisioning state — the wallet is not (yet) registered as a
    provider on the coordinator — distinct from a bad-signature ``401``.
    Callers may retry the handshake once registration completes.
    """


class SlaExpired(VorqdError):
    """A job's SLA deadline (minus safety margin) passed before settlement."""
