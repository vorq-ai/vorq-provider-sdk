"""The open-bid DEK release: the escrow seam, and nothing else.

A container's ``seed_wrap`` is sealed to whoever is meant to open it. On a
**designated** bid that is this daemon's own box key, and the daemon unseals it
in-process — no network, no third party — recovering a **seed** it then derives
the DEK from (:func:`vorqd.container.derive_dek`). On an **open** bid the wrap is
sealed to the coordinator's attested escrow key, and the DEK is obtained by
asking the escrow for it (``POST /release``) once the claim is on record: the
escrow unseals, **derives against the owner it reads from chain**, re-seals the
result to a response key this request carries, and never sees the ciphertext.

So the two paths differ in one more way than "who holds the key": what comes back
from ``/release`` is the **already-derived DEK**, and deriving it a second time
here would produce a key of a key. The derivation belongs to whoever unseals the
wrap, and on this path that is not us.

This is deliberately a seam of its own. It is **not** the retired "DEK in the
claim response" path: that one had the coordinator hand a key to whoever claimed,
on a surface with no attestation and no per-request binding, and it is gone.
What lives here is a request against an attested escrow, refused by a frozen,
typed vocabulary.

A daemon with **no release client configured refuses open bids with
``escrow_unavailable``** — the escrow's own code for "this endpoint cannot serve
you" — and gives the claim back immediately, so the client is refunded now rather
than at SLA expiry. It never guesses, and it never falls back to a key delivered
by anything else.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Awaitable, Callable, Protocol

import httpx
from nacl.encoding import HexEncoder
from nacl.public import PrivateKey, SealedBox

from .container import SEED_WRAP_BYTES

#: The escrow's frozen refusal vocabulary — **ten** codes. Each is reported
#: verbatim as the job's failure reason and metered under that label; they are
#: stable tokens, and an operator tells a lost key from a wrong wallet by them.
#:
#: ``escrow_handed_over`` is the tenth and it is not decoration: a holder that has
#: handed its keys to a successor cannot be described by the other nine without
#: lying about retryability — it is neither "the key is gone" nor "you asked the
#: wrong wallet".
RELEASE_REFUSALS = (
    "bad_container",
    "wrap_mismatch",
    "no_claim",
    "not_claimed",
    "wrong_wallet",
    "stale_issued_at",
    "unseal_failed",
    "escrow_key_lost",
    "escrow_handed_over",
    "escrow_unavailable",
)

#: The one refusal that says the key existed and is permanently gone. It is the
#: daemon's cue to hand the job back **now**, inside the penalty-free grace
#: window, rather than sit on a claim whose payload can never be opened.
ESCROW_KEY_LOST = "escrow_key_lost"

#: How far ``issued_at`` may sit from the escrow's clock, in seconds. The daemon
#: stamps ``now`` on every request and never reuses one, so this is stated only
#: so the bound has a name where the request is built.
RELEASE_SKEW_SECONDS = 600

#: The escrow's EIP-712 domain — a **third** namespace, and deliberately neither
#: the session handshake's ``{VORQ, 1}`` nor the registries' ``{VORQ, 2,
#: verifyingContract}``: a captured login must not be a release, and a captured
#: op must not be either. There is no ``verifyingContract`` because no contract
#: verifies this — the escrow is an API. The chain id is bound because one
#: provider wallet is registered on more than one deployment and a job id is
#: derivable on all of them.
RELEASE_DOMAIN_NAME = "VORQ Escrow"
RELEASE_DOMAIN_VERSION = "1"

#: ``responsePubkey`` is **inside** the signature, which is what makes a replay
#: harmless: a captured request can only ever re-deliver to the recipient its
#: original signer chose. Leave that field out and the request becomes a bearer
#: token for the DEK.
#:
#: There is **no ``owner``**. ``jobId`` is ``keccak256(owner ‖ c)``, so the owner
#: is already committed by the id this struct signs, and the escrow reads it off
#: chain to derive under — a copy in the request could only ever disagree with the
#: copy that counts, and the coordinator carried a narrow-signature function of
#: its own purely to keep the body's owner out of scope of the KDF. Deleting the
#: field deletes what that defence was defending against.
#:
#: This must match the coordinator character for character. EIP-712 hashes the
#: **type**, so a rename or a dropped member the two sides do not make together
#: is a signature that stops verifying with no other symptom:
#: ``Release(bytes32 jobId,bytes seedWrap,bytes32 ctHash,bytes32 responsePubkey,uint64 issuedAt)``
RELEASE_TYPES: dict[str, list[dict[str, str]]] = {
    "Release": [
        {"name": "jobId", "type": "bytes32"},
        {"name": "seedWrap", "type": "bytes"},
        {"name": "ctHash", "type": "bytes32"},
        {"name": "responsePubkey", "type": "bytes32"},
        {"name": "issuedAt", "type": "uint64"},
    ]
}


def release_domain(chain_id: int) -> dict[str, Any]:
    return {"name": RELEASE_DOMAIN_NAME, "version": RELEASE_DOMAIN_VERSION, "chainId": int(chain_id)}


class ReleaseRefused(Exception):
    """The escrow will not release this DEK.

    ``code`` is normally one of :data:`RELEASE_REFUSALS` and is carried
    **verbatim** either way: a code this build does not know is surfaced as
    itself rather than collapsed into a neighbour, because the neighbour would
    misdescribe it to the operator reading the log. Unknown or not, a refusal is
    never retryable — the escrow has pronounced on chain state as it stands, and
    the identical request can only be refused again.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code

    @property
    def known(self) -> bool:
        """Is this one of the frozen ten? Diagnostic only — nothing branches on
        it, because an unknown code must be reported, not guessed at."""
        return self.code in RELEASE_REFUSALS

    @property
    def retryable(self) -> bool:
        return False


class EscrowRelease(Protocol):
    """Obtain the DEK an open bid's wrap seeds, for a job this daemon has claimed."""

    async def release(self, job: Any, seed_wrap: bytes, ct_hash: bytes) -> bytes:
        """The 32-byte DEK — **already derived** — or :class:`ReleaseRefused`.

        ``ct_hash`` is ``keccak256(ciphertext)``: the escrow re-derives the
        commitment from it and refuses a wrap that belongs to another order
        without ever seeing the payload.
        """
        ...


class TypedDataSigner(Protocol):
    address: str

    def sign_typed_data(
        self, domain: dict[str, Any], types: dict[str, list[dict[str, str]]], message: dict[str, Any]
    ) -> str: ...


def _hex32(value: str | bytes) -> str:
    raw = bytes(value) if isinstance(value, (bytes, bytearray)) else bytes.fromhex(
        value[2:] if value[:2].lower() == "0x" else value
    )
    if len(raw) != 32:
        raise ValueError(f"expected a 32-byte word, got {len(raw)} bytes")
    return "0x" + raw.hex()


class HttpEscrowRelease:
    """``POST {escrow_url}/release`` — the real seam.

    One request holds a **fresh** X25519 keypair, generated here and dropped when
    the call returns. The public half is signed into the request, so the escrow
    can only ever deliver to the key this signer chose, and the secret half never
    leaves this coroutine — a DEK re-sealed to a long-lived key would be readable
    by anyone who ever obtains that key, including from a core dump taken long
    after the job settled.

    The escrow is a **key oracle, not a chain door**: it answers a refusal
    vocabulary of its own and touches no transaction, which is why it is not a
    method on the node client.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        escrow_url: str,
        signer: TypedDataSigner,
        chain_id: int | Callable[[], Awaitable[int]],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._http = http
        self._url = escrow_url.rstrip("/")
        self._signer = signer
        # An int when the caller already knows it, or an awaitable that answers
        # one: the chain id comes from `GET /evm/chain`, which the wiring cannot
        # read synchronously at construction time. Resolved once and cached — a
        # node pointed at another deployment is another node.
        self._chain_id = chain_id
        self._resolved: int | None = chain_id if isinstance(chain_id, int) else None
        self._clock = clock

    async def chain_id(self) -> int:
        if self._resolved is None:
            source = self._chain_id
            assert not isinstance(source, int)  # resolved in __init__ when it is
            self._resolved = int(await source())
        return self._resolved

    async def release(self, job: Any, seed_wrap: bytes, ct_hash: bytes) -> bytes:
        if len(seed_wrap) != SEED_WRAP_BYTES:
            raise ReleaseRefused(
                "bad_container",
                f"a container v1 seed_wrap is {SEED_WRAP_BYTES} bytes, and this one is {len(seed_wrap)}",
            )
        response_key = PrivateKey.generate()
        response_pubkey = response_key.public_key.encode(HexEncoder).decode()
        issued_at = int(self._clock())
        job_id = _hex32(job.job_id)
        # No owner, in the message or in the body: `jobId` is keccak256(owner ‖ c)
        # and the escrow reads the owner off chain to derive under. A second copy
        # on the wire could only ever disagree with the one that counts.
        message = {
            "jobId": job_id,
            "seedWrap": "0x" + seed_wrap.hex(),
            "ctHash": _hex32(ct_hash),
            "responsePubkey": "0x" + response_pubkey,
            "issuedAt": issued_at,
        }
        signature = self._signer.sign_typed_data(
            release_domain(await self.chain_id()), RELEASE_TYPES, message
        )
        body = {
            "job_id": job_id,
            "seed_wrap": base64.b64encode(seed_wrap).decode(),
            "ct_hash": _hex32(ct_hash),
            "response_pubkey": response_pubkey,
            "issued_at": issued_at,
            "signature": signature,
        }
        resp = await self._http.post(f"{self._url}/release", json=body)
        if resp.status_code != 200:
            code = _refusal_code(resp)
            if code is None:
                # No code at all: the node's own failure to serve, not the
                # escrow's verdict — a 429, a 503 with no chain, a gateway. It
                # propagates as a transport error rather than being dressed up
                # as a refusal, because a refusal tells the caller to stop.
                resp.raise_for_status()
            raise ReleaseRefused(str(code), _message(resp))
        sealed = base64.b64decode(resp.json()["dek_sealed"], validate=True)
        try:
            dek = SealedBox(response_key).decrypt(sealed)
        except Exception as exc:  # noqa: BLE001 — one refusal for any crypto failure
            raise ReleaseRefused(
                "unseal_failed", "the escrow's answer does not open with this request's response key"
            ) from exc
        if len(dek) != 32:
            raise ReleaseRefused("bad_container", f"a DEK is 32 bytes, and this one is {len(dek)}")
        return dek


def _body(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _refusal_code(resp: httpx.Response) -> str | None:
    error = _body(resp).get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return str(code) if code else None


def _message(resp: httpx.Response) -> str:
    error = _body(resp).get("error")
    return str(error.get("message", "")) if isinstance(error, dict) else ""


__all__ = [
    "ESCROW_KEY_LOST",
    "EscrowRelease",
    "HttpEscrowRelease",
    "RELEASE_DOMAIN_NAME",
    "RELEASE_DOMAIN_VERSION",
    "RELEASE_REFUSALS",
    "RELEASE_SKEW_SECONDS",
    "RELEASE_TYPES",
    "ReleaseRefused",
    "release_domain",
]
