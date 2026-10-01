"""EIP-712 signing for everything a provider authorises, and nothing else.

This module holds no transport. It turns an op's fields into the exact typed
data the deployed registries verify, hands it to ``eth-account``, and returns a
65-byte ``r ‖ s ‖ v`` signature as hex. :mod:`vorqd.node` carries it to the
coordinator node, which relays it and pays the gas; the signature is the whole
of the authority and ``msg.sender`` is never read on chain, so any relayer may
land any of these.

**The domain is chosen here, by what is being signed, and is never a
parameter.** ``Claim``, ``Settle`` and ``Fail`` verify against the
**JobRegistry**; ``SetIdentity`` and ``RequestCapacity`` verify against the
**ProviderRegistry**; the ``AskSnapshot`` verifies against the **AskRegistry**.
Each declares its own domain name, so a signature made under the wrong one is
now distinguishable — but only to something that knows all three names, and it
is still not an *error*: it recovers a different address and the contract simply
refuses it. Letting a caller pass a domain would make that mistake reachable;
letting the artifact decide makes it unreachable, which is the guarantee that
does not depend on any address being configured correctly.

**There is no ``reclaim`` here, deliberately.** ``reclaim`` is permissionless
and unsigned: a stranded provider's slot is freed by its node's keeper or by any
raw tooling, and an SDK signing path for it would imply an authority the op does
not have.

**``Settle`` carries no ``resultCid``**, and it cannot: the name does not exist
when the op is signed. The claimant hands the result bytes to the node, the node
pins them and learns the name from the storage service, and that name is
submitted alongside the signature and recorded — never attested. The signed
members are the job, the delivered count and the issue time, which is what stops
a relayer editing what it settles.

**Every signature this module produces is canonical: ``v`` in {27, 28} and
``s <= N/2``, and there is a test that says so.** The registries recover with
bare ``ecrecover`` and impose no canonical-``s`` bound, so the high-``s`` twin
``(r, N - s, 55 - v)`` of any op signed here recovers the same operator address
and is accepted just as readily — signature **malleability**, deliberate on the
chain side, because replay is stopped by the job state machine and by the
monotonic ``issuedAt`` floors rather than by signature bytes (see
``vorq-evm-contracts/README.md``, "Signature caveats").

The rule that leaves on this side is one line: **a signature is never an
identifier.** A relay that deduplicated on the signature bytes would relay one
claim twice and pay the gas for the second; a keeper that remembered "already
signed" by them would forget it the moment a twin arrived. Identity is the op's
own fields — the job id, the provider id, the ``issuedAt``.

Every digest this module produces is pinned by ``tests/vectors/signing-v3.json``,
which was generated from the deployed contracts and whose every case was handed
back to the contract that owns it and accepted. The tests assert :data:`SIGNED_TYPES`
against that file's own ``types`` blocks by direct list equality, which is why no
type STRING is derived anywhere in this module.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak

from .money import parse_usd
from .types import ChainContext

#: Each registry declares its **own** EIP-712 domain name. They used to share
#: ``"VORQ"`` and differ only in ``verifyingContract``, which left one real
#: failure invisible: a chain context whose two address slots resolve to the same
#: contract produces digests indistinguishable from legitimate ones. Distinct
#: names make the separators differ whatever the addresses are.
DOMAIN_NAMES: dict[str, str] = {
    "job_registry": "VORQ Jobs",
    "provider_registry": "VORQ Providers",
    "ask_registry": "VORQ Asks",
}

#: Version **2**, and not the ``1`` of the session handshake: the handshake is an
#: off-chain auth artifact and lives in a namespace the chain will never accept,
#: so a captured login can never be replayed as an op.
DOMAIN_VERSION = "2"

#: The five op types, member for member as the contracts' typehash strings spell
#: them. A member renamed, retyped or reordered recovers a different address and
#: the op is refused — silently — so these are transcribed from the contract
#: source and pinned against ``signing-v3.json``'s type tables by the tests.
OP_TYPES: dict[str, dict[str, list[dict[str, str]]]] = {
    "Claim": {
        "Claim": [
            {"name": "jobId", "type": "bytes32"},
            {"name": "issuedAt", "type": "uint64"},
        ]
    },
    "Settle": {
        "Settle": [
            {"name": "jobId", "type": "bytes32"},
            {"name": "completionTok", "type": "uint32"},
            {"name": "issuedAt", "type": "uint64"},
        ]
    },
    # Members identical to ``Claim``'s. Only the typehash separates the two
    # digests, so a signer that reuses ``Claim``'s type for a fail produces a
    # signature the contract silently refuses.
    "Fail": {
        "Fail": [
            {"name": "jobId", "type": "bytes32"},
            {"name": "issuedAt", "type": "uint64"},
        ]
    },
    "SetIdentity": {
        "SetIdentity": [
            {"name": "boxKey", "type": "bytes32"},
            # Dynamic bytes: enters the struct hash as ``keccak256(evidence)``,
            # never inline, and ``keccak256("")`` is a real hash rather than a
            # zero word.
            {"name": "evidence", "type": "bytes"},
            {"name": "issuedAt", "type": "uint64"},
        ]
    },
    "RequestCapacity": {
        "RequestCapacity": [
            {"name": "n", "type": "uint32"},
            {"name": "issuedAt", "type": "uint64"},
        ]
    },
}

#: Which registry verifies each op. This mapping is the Q6 defence: it is the
#: only place an op is associated with an address, and there is no way to sign
#: an op without going through it.
OP_REGISTRY: dict[str, str] = {
    "Claim": "job_registry",
    "Settle": "job_registry",
    "Fail": "job_registry",
    "SetIdentity": "provider_registry",
    "RequestCapacity": "provider_registry",
}

#: EIP-712 primary type → the ``op`` name ``POST /evm/ops`` answers to.
OP_NAMES: dict[str, str] = {
    "Claim": "claim",
    "Settle": "settle",
    "Fail": "fail",
    "SetIdentity": "set_identity",
    "RequestCapacity": "request_capacity",
}

#: The ask book's typed data, transcribed from ``AskRegistry.SNAPSHOT_TYPEHASH``
#: and ``AskRegistry.ASK_TYPEHASH``. Two structs, because ``quotes`` is an array
#: of them: EIP-712 hashes each element under ``Ask``'s own typehash and puts
#: ``keccak256`` of the concatenated element hashes in the member's slot.
#:
#: ``rateIn``/``rateOut`` are ``uint128`` and are **scaled integers** — atomic
#: token units per ``RATE_SCALE`` units of work — not decimals. A price that is
#: not a whole number is not a price this book can hold.
ASK_SNAPSHOT_TYPES: dict[str, list[dict[str, str]]] = {
    "AskSnapshot": [
        {"name": "providerId", "type": "uint32"},
        {"name": "signedAt", "type": "uint64"},
        {"name": "quotes", "type": "Ask[]"},
    ],
    "Ask": [
        {"name": "modelId", "type": "uint32"},
        {"name": "sla", "type": "uint32"},
        {"name": "rateIn", "type": "uint128"},
        {"name": "rateOut", "type": "uint128"},
    ],
}

#: Every EIP-712 artifact this daemon signs → its types. The ops door's five
#: (:data:`OP_TYPES`) plus the ask book's one, which is **not** an op: it does
#: not go to ``POST /evm/ops``, it has no entry in :data:`OP_NAMES`, and it is
#: pushed whole to ``PUT /evm/asks``.
SIGNED_TYPES: dict[str, dict[str, list[dict[str, str]]]] = {
    **OP_TYPES,
    "AskSnapshot": ASK_SNAPSHOT_TYPES,
}

#: Which contract verifies each artifact — the whole of the domain defence, and
#: the only place an artifact is associated with an address.
SIGNED_REGISTRY: dict[str, str] = {**OP_REGISTRY, "AskSnapshot": "ask_registry"}


@runtime_checkable
class TypedDataSigner(Protocol):
    """Holds the operator wallet key and signs EIP-712 messages with it."""

    address: str

    def sign_typed_data(
        self, domain: dict[str, Any], types: dict[str, list[dict[str, str]]], message: dict[str, Any]
    ) -> str: ...


def _bytes32(value: str | bytes, field: str) -> bytes:
    """A ``bytes32`` member as bytes, from ``0x`` hex or raw bytes.

    Normalised here rather than passed through, so the encoder is handed one
    representation and a 31-byte value fails loudly instead of being padded into
    a different digest.
    """
    raw = value if isinstance(value, bytes) else bytes.fromhex(str(value).removeprefix("0x"))
    if len(raw) != 32:
        raise ValueError(f"{field} must be 32 bytes, got {len(raw)}")
    return raw


def _dyn_bytes(value: str | bytes | None) -> bytes:
    """A dynamic ``bytes`` member, from ``0x`` hex, raw bytes, or nothing."""
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return bytes.fromhex(str(value).removeprefix("0x"))


def domain_for(primary_type: str, ctx: ChainContext) -> dict[str, Any]:
    """The EIP-712 domain ``primary_type`` verifies against, from ``ctx``."""
    try:
        contract = SIGNED_REGISTRY[primary_type]
    except KeyError:
        raise ValueError(f"no VORQ op is called {primary_type!r}") from None
    return {
        "name": DOMAIN_NAMES[contract],
        "version": DOMAIN_VERSION,
        "chainId": ctx.chain_id,
        "verifyingContract": getattr(ctx, contract),
    }


def typed_data(primary_type: str, message: dict[str, Any], ctx: ChainContext) -> dict[str, Any]:
    """The full EIP-712 payload for one op — domain, types, primary type, message."""
    return {
        "domain": domain_for(primary_type, ctx),
        "types": SIGNED_TYPES[primary_type],
        "primaryType": primary_type,
        "message": message,
    }


def digest(data: dict[str, Any]) -> str:
    """``keccak256(0x1901 ‖ domainSeparator ‖ structHash)`` for a typed payload.

    Built from ``eth-account``'s own encoding rather than re-derived, so the
    bytes asserted against the vectors are the bytes that get signed.
    """
    signable = encode_typed_data(full_message=data)
    return "0x" + keccak(b"\x19" + signable.version + signable.header + signable.body).hex()


def ask_snapshot_message(snapshot: dict[str, Any], decimals: int) -> dict[str, Any]:
    """The wire snapshot as the typed-data message the AskRegistry verifies.

    The rates travel as USD decimal strings and this is the one place they
    become the atomic ``uint128`` members, at the token's ``decimals``. Reading the
    signature's inputs out of the exact dict that is about to be pushed is
    deliberate: a snapshot signed from one object and pushed from another is the
    one way the two can disagree, and the chain would answer that by recovering
    a stranger.
    """
    return {
        "providerId": int(snapshot["provider_id"]),
        "signedAt": int(snapshot["signed_at"]),
        "quotes": [
            {
                "modelId": int(q["model_id"]),
                "sla": int(q["sla"]),
                "rateIn": parse_usd(q["rate_in"], decimals),
                "rateOut": parse_usd(q["rate_out"], decimals),
            }
            for q in snapshot["quotes"]
        ],
    }


class OpSigner:
    """Signs the five provider ops with the operator wallet key.

    Every method takes the op's fields and a :class:`~vorqd.types.ChainContext`
    and returns a hex signature. None of them takes a domain, a contract address
    or a type string, and that is the point: the caller cannot get the domain
    wrong because the caller never states it.
    """

    def __init__(self, signer: TypedDataSigner) -> None:
        self._signer = signer

    @property
    def address(self) -> str:
        """The operator address these signatures recover to."""
        return self._signer.address

    def _sign(self, primary_type: str, message: dict[str, Any], ctx: ChainContext) -> str:
        data = typed_data(primary_type, message, ctx)
        return self._signer.sign_typed_data(data["domain"], data["types"], data["message"])

    # -- job registry ops -------------------------------------------------- #

    def sign_claim(self, job_id: str | bytes, issued_at: int, ctx: ChainContext) -> str:
        """``Claim(bytes32 jobId,uint64 issuedAt)`` — accepting a job."""
        return self._sign(
            "Claim", {"jobId": _bytes32(job_id, "job_id"), "issuedAt": int(issued_at)}, ctx
        )

    def sign_settle(
        self, job_id: str | bytes, completion_tok: int, issued_at: int, ctx: ChainContext
    ) -> str:
        """``Settle(bytes32 jobId,uint32 completionTok,uint64 issuedAt)`` — delivery.

        **There is no ``result_cid`` parameter and there never will be.** The
        result's name is minted by the node after it pins the bytes this op's
        payload carries; the daemon does not know it and does not sign it.
        """
        return self._sign(
            "Settle",
            {
                "jobId": _bytes32(job_id, "job_id"),
                "completionTok": int(completion_tok),
                "issuedAt": int(issued_at),
            },
            ctx,
        )

    def sign_fail(self, job_id: str | bytes, issued_at: int, ctx: ChainContext) -> str:
        """``Fail(bytes32 jobId,uint64 issuedAt)`` — the abort inside the grace window."""
        return self._sign(
            "Fail", {"jobId": _bytes32(job_id, "job_id"), "issuedAt": int(issued_at)}, ctx
        )

    # -- provider registry ops --------------------------------------------- #

    def sign_set_identity(
        self,
        box_key: str | bytes,
        evidence: str | bytes | None,
        issued_at: int,
        ctx: ChainContext,
    ) -> str:
        """``SetIdentity(bytes32 boxKey,bytes evidence,uint64 issuedAt)``.

        A **ProviderRegistry**-domain op. ``evidence`` is opaque bytes on chain,
        so it is signed as bytes and never as text — a second interpretation of
        the same field would make the one the signature covers a coin toss.
        """
        return self._sign(
            "SetIdentity",
            {
                "boxKey": _bytes32(box_key, "box_key"),
                "evidence": _dyn_bytes(evidence),
                "issuedAt": int(issued_at),
            },
            ctx,
        )

    def sign_request_capacity(self, n: int, issued_at: int, ctx: ChainContext) -> str:
        """``RequestCapacity(uint32 n,uint64 issuedAt)`` — asking for concurrency.

        The other ProviderRegistry-domain op, and the one whose fields say
        nothing at all about which contract it belongs to.
        """
        return self._sign("RequestCapacity", {"n": int(n), "issuedAt": int(issued_at)}, ctx)

    # -- ask registry ------------------------------------------------------ #

    def sign_ask_snapshot(self, snapshot: dict[str, Any], ctx: ChainContext) -> str:
        """``AskSnapshot(uint32 providerId,uint64 signedAt,Ask[] quotes)`` — the book.

        A **third** domain: the AskRegistry computes its own
        ``DOMAIN_SEPARATOR`` over its own address, so a snapshot signed against
        either registry recovers a stranger and ``setAsks`` silently skips the
        entry rather than reverting.

        ``snapshot`` is the wire body itself, so what is signed and what is
        pushed cannot drift apart. The snapshot names its
        own ``provider_id``: the contract compares it against ``idOf(signer)``
        and skips a mismatch, which is what stops a real operator publishing a
        price under somebody else's id.
        """
        return self._sign("AskSnapshot", ask_snapshot_message(snapshot, ctx.decimals), ctx)


def recover(data: dict[str, Any], signature: str) -> str:
    """The address that signed ``data``, for verifying a signature locally."""
    return Account.recover_message(encode_typed_data(full_message=data), signature=signature)
