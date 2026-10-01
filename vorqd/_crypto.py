"""Session signing.

The daemon's identity on the network is its wallet address; it proves control of
the wallet by signing the coordinator's nonce as an **EIP-712** ``VorqSession``
message. Only the signature and the address travel — the wallet key never leaves
the process.

``WalletSigner`` holds a secp256k1 wallet key and produces the signature a
coordinator verifies against exactly the ``domain`` / ``types`` / message shapes
defined below. The wallet key is **required** — it is the daemon's payable
identity — and is read from ``$VORQ_WALLET_KEY`` (or passed explicitly); the
daemon refuses to start without it. Only the session credential is derived under
the hood: the ``vorq_sess_…`` token is minted by signing the coordinator's nonce.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from eth_account import Account
from nacl.encoding import HexEncoder
from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.secret import SecretBox

# -- canonical EIP-712 contract -------------------------------------------------
#
# Single source of truth for what a coordinator verifies for a session handshake.
# Must match the client SDK's VorqSession definition exactly.

DOMAIN_NAME = "VORQ Session"
DOMAIN_VERSION = "1"

#: The session handshake's chain id is the deployment's, announced by ``GET
#: /auth/nonce`` as ``chain_id``. It used to be pinned at 1: a browser wallet
#: refuses to sign a typed-data domain whose chain is not the one it is on, so
#: the pin locked every real wallet out, and every SDK moved together.

SESSION_TYPES: dict[str, list[dict[str, str]]] = {
    "VorqSession": [
        {"name": "address", "type": "address"},
        {"name": "nonce", "type": "string"},
    ]
}


def vorq_domain(chain_id: int) -> dict[str, Any]:
    """The off-chain VORQ domain — the session handshake's, and no contract's.

    ``chain_id`` is the deployment's, which ``GET /auth/nonce`` announces as
    ``chain_id``. Version ``1`` against the on-chain artifacts' ``2``, with no
    ``verifyingContract`` at all, keeps a login in a namespace the chain will
    never accept.
    """
    return {"name": DOMAIN_NAME, "version": DOMAIN_VERSION, "chainId": int(chain_id)}


def _to_0x_hex(value: bytes) -> str:
    to_0x_hex = getattr(value, "to_0x_hex", None)
    if callable(to_0x_hex):
        return to_0x_hex()
    raw = value.hex()
    return raw if raw.startswith("0x") else "0x" + raw


# -- protocol -------------------------------------------------------------------


@runtime_checkable
class Signer(Protocol):
    """Holds the wallet key; signs session nonces locally."""

    address: str

    def sign_nonce(self, nonce: str, chain_id: int) -> str: ...


# -- implementation -------------------------------------------------------------


class WalletSigner:
    """EIP-712 session signer backed by a secp256k1 wallet key.

    The key is taken from ``private_key`` or, failing that, the ``key_env``
    environment variable — so production keeps the secret out of source. It is
    **required**: with neither set the constructor raises, and the daemon refuses
    to start. Use a dedicated wallet funded with your provider earnings, never a
    main wallet's key.
    """

    def __init__(
        self,
        private_key: str | None = None,
        *,
        key_env: str = "VORQ_WALLET_KEY",
    ) -> None:
        key = private_key or os.environ.get(key_env)
        if not key:
            raise ValueError(f"no wallet key: set ${key_env} or pass private_key")
        self._account = Account.from_key(key)
        self.address: str = self._account.address

    @classmethod
    def generate(cls) -> "WalletSigner":
        """Create a signer over a freshly generated wallet (tests, ephemeral use)."""
        return cls(Account.create().key.hex())

    def sign_nonce(self, nonce: str, chain_id: int) -> str:
        """Sign a session-handshake nonce under the deployment's chain; returns a 0x EIP-712 signature."""
        message = {"address": self.address, "nonce": nonce}
        signed = Account.sign_typed_data(
            self._account.key, vorq_domain(chain_id), SESSION_TYPES, message
        )
        return _to_0x_hex(signed.signature)

    def sign_typed_data(
        self, domain: dict[str, Any], types: dict[str, list[dict[str, str]]], message: dict[str, Any]
    ) -> str:
        """Sign an arbitrary EIP-712 message; returns a 0x signature.

        The one door the wallet key opens besides the session handshake. It is
        deliberately generic and deliberately dumb: it chooses no domain and
        knows no op. :mod:`vorqd.opsig` owns which domain each op belongs to,
        because that choice is the one a wrong answer to fails silently — a
        signature under the wrong ``verifyingContract`` recovers a stranger
        rather than raising.
        """
        signed = Account.sign_typed_data(self._account.key, domain, types, message)
        return _to_0x_hex(signed.signature)


# -- payload confidentiality (§03) ----------------------------------------------
#
# The daemon holds a Curve25519 box keypair: it publishes the public key on
# registration so clients can seal a container's ``seed_wrap`` to it (designated
# bids), and opens those sealed boxes with its private key. On an open bid the
# wrap is sealed to the coordinator's escrow instead and the DEK comes from a
# release; either way ``open_dek`` opens the container's SecretBox with it.
# ``seal_to`` seals the *result* back to the order's ``result_key``.


class BoxCipher:
    """libsodium sealed-box keypair — the daemon's payload-decryption identity.

    The private key is read from ``private_key`` or the ``key_env`` environment
    variable. ``public_key`` is published on registration; ``decrypt`` opens boxes
    sealed to it.
    """

    def __init__(self, private_key: str | None = None, *, key_env: str = "VORQ_BOX_KEY") -> None:
        key = private_key or os.environ.get(key_env)
        if not key:
            raise ValueError(f"no box key: pass private_key or set ${key_env}")
        self._private_hex = key
        self._private = PrivateKey(key.encode(), encoder=HexEncoder)

    @classmethod
    def generate(cls) -> "BoxCipher":
        """Create a cipher over a freshly generated keypair (tests, ephemeral use)."""
        return cls(PrivateKey.generate().encode(HexEncoder).decode())

    @property
    def public_key(self) -> str:
        """This cipher's Curve25519 public key as hex — publish it to receive."""
        return self._private.public_key.encode(HexEncoder).decode()

    def decrypt(self, data: bytes) -> bytes:
        """Open a sealed box addressed to this cipher's key — a designated bid's
        ``seed_wrap``, which is how the seed reaches a daemon that was named."""
        return SealedBox(self._private).decrypt(data)


def open_dek(data: bytes, dek: bytes) -> bytes:
    """Open a container's ciphertext with the DEK derived from the seed its
    ``seed_wrap`` sealed (``SecretBox``, nonce-prefixed)."""
    return SecretBox(dek).decrypt(data)


def seal_to(recipient_public_key: str, data: bytes) -> bytes:
    """Seal ``data`` to a recipient's Curve25519 public key (the envelope's result_key)."""
    box = SealedBox(PublicKey(recipient_public_key.encode(), encoder=HexEncoder))
    return bytes(box.encrypt(data))
