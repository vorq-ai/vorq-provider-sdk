"""The shapes that cross the daemon's seams, and nothing that speaks HTTP.

``EvmJob`` and ``ChainContext`` are shapes, not surfaces: they belong to whoever
reads and writes them rather than to one client. :mod:`vorqd.node` produces both
from a coordinator node's answers and :mod:`vorqd.opsig` signs against the
context, so putting either in one of those modules would make the other import a
client in order to name a dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: The largest body the node's two byte-carrying doors read. The node's
#: ``MAX_BODY_BYTES``: one ceiling for both doors, and the only number here with
#: a choice behind it — :data:`INLINE_MAX_BYTES` is derived from it.
MAX_BODY_BYTES = 20 * 1024 * 1024

#: What the inline threshold leaves free for everything in a settle that is not
#: the result — the job id, the counts, a 65-byte signature. Far larger than
#: those need, because reserving too little means inlining a result the door then
#: refuses with a ``413``.
ENVELOPE_RESERVE_BYTES = 64 * 1024


def base64_length(n: int) -> int:
    """How many bytes base64 costs to encode ``n``, as the encoder counts them."""
    return 4 * ((n + 2) // 3)


#: A sealed result at or under this many bytes rides inline as base64; a bigger
#: one is uploaded first.
#:
#: **Derived, not chosen.** This was a second constant, hand-written as 7 MiB
#: here, in both client SDKs and in the spec, with nothing holding the four
#: copies together — editing one left every suite green while the parties
#: silently disagreed about where the line was. It was never a second decision:
#: base64 costs a third again, which is the whole reason a result cannot be
#: inlined into a body its own size. Every party derives it the same way from the
#: same ceiling.
INLINE_MAX_BYTES = ((MAX_BODY_BYTES - ENVELOPE_RESERVE_BYTES) // 4) * 3


@dataclass(frozen=True)
class ChainContext:
    """The chain a coordinator node relays to: its id and its four contracts.

    All four, and that is a rule rather than a convenience. The two registries
    each compute their own EIP-712 ``DOMAIN_SEPARATOR`` over ``address(this)``,
    and each declares its **own** domain name — ``VORQ Jobs`` and
    ``VORQ Providers`` — at the same version ``2`` and the same chain id. The
    distinct names are load-bearing: they make the two separators differ
    whatever the addresses are, so a configuration whose two address slots
    resolve to the same contract cannot produce digests indistinguishable from
    legitimate ones. A signature made under the wrong ``verifyingContract`` is
    not an error anywhere: it recovers a different address, and the contract
    refuses it silently. Carrying one address and
    reusing it for every op is therefore a bug with no symptom until a
    provider's registry ops stop landing, which is why the whole set is read
    once and kept together.

    ``decimals`` is the payment token's: the coordinator speaks money as USD
    decimal strings and the ask snapshot signs atomic integers, so it is the one
    shift between the two (:mod:`vorqd.money`).

    What is deliberately **not** here: ``head_block``, ``block_time_ms``,
    ``token_domain`` and ``fee_bps``.
    ``GET /evm/chain`` answers them all and the daemon has no use for any — it never
    builds, funds or broadcasts a transaction, and it signs no payment
    artifact, so the token's own EIP-712 domain is the client's business.
    Keeping a gas field on this object would be the first RPC concept back in a
    daemon that holds none.
    """

    chain_id: int
    job_registry: str
    provider_registry: str
    ask_registry: str
    usdc: str
    decimals: int

    @classmethod
    def from_wire(cls, d: dict) -> "ChainContext":
        contracts = d.get("contracts") or {}
        names = ("job_registry", "provider_registry", "ask_registry", "usdc")
        missing = [name for name in names if not contracts.get(name)]
        if missing:
            raise ValueError(
                "GET /evm/chain answered without " + ", ".join(missing) +
                "; every op signs against one of these addresses"
            )
        if not isinstance(d.get("decimals"), int):
            raise ValueError("GET /evm/chain answered without decimals; every rate converts by it")
        return cls(
            chain_id=int(d["chain_id"]),
            decimals=d["decimals"],
            **{name: contracts[name] for name in names},
        )


@dataclass
class EvmJob:
    job_id: str
    model: str
    state: str
    sla: str
    created_at: int
    owner: str | None = None
    provider: int | None = None
    rate_in: int | None = None     # atomic units per RATE_SCALE units; the wire's USD, converted
    rate_out: int | None = None
    expires_at: int | None = None
    claimed_at: int | None = None
    settled_at: int | None = None
    ended_at: int | None = None       # the one instant the job went terminal
    ended_because: str | None = None  # settled | cancelled | provider_fail | reclaim
    task_cid: str | None = None       # CID of the raw task bytes, named by the order
    result_cid: str | None = None     # CID of the sealed result bytes, named at settle
    units_in: int | None = None
    units_out: int | None = None              # cap in the model's out unit (media: exact pixels / pixel-seconds)
    completion_tokens: int | None = None      # actual delivered output units at settle (≤ units_out)
    # The pinned provider of a designated bid. **``0`` is the open-order
    # sentinel, not ``None``** — it is what the registry stores and what the node
    # serves, verbatim. The default moves with the guard that reads it: a
    # scheduler skipping on ``designated is not None and designated != mine``
    # treats every open job as a foreign designation and claims nothing at all,
    # which is a daemon that looks healthy and earns nothing.
    designated: int = 0
    # There is no key field on a job row. The DEK is derived from the seed inside
    # the container's own ``seed_wrap``, sealed to this provider (designated) or
    # to the escrow (open) — never delivered by the surface that answers a claim.
    # Not an order term and never on the wire: modality is a catalog fact about
    # the model, resolved by the scheduler from the curated catalog (falling back
    # to the operator's own config). This "text" is a placeholder for a job whose
    # modality has not been resolved yet, NOT a safe default: running media work
    # as text skips the units_out clamp and settles at the full cap, which is
    # exactly the underpay/overcharge the design refuses. A model the scheduler
    # cannot name is not claimed at all (``_poll_model`` skips it before the
    # claim), so no job ever reaches execution on this placeholder.
    modality: str = "text"

    @classmethod
    def from_wire(cls, d: dict) -> "EvmJob":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# There is deliberately no ``Ask`` dataclass. An ask is not a shape this daemon
# holds — it is one row inside a signed ``AskSnapshot``, and the whole snapshot is
# what crosses the wire and what the signature covers. A dataclass here would be a
# second spelling of the four members ``AskRegistry.ASK_TYPEHASH`` fixes, and the
# scheduler would have to convert it back into the dict it signs, which is exactly
# where a book that was signed and a book that was pushed come apart.
