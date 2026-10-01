"""WalletSigner: EIP-712 VorqSession signing; wallet key is required."""

from __future__ import annotations

import pytest
from eth_account import Account
from eth_account.messages import encode_typed_data

from vorqd._crypto import SESSION_TYPES, WalletSigner, vorq_domain

KEY = "0x" + "4a" * 32


def test_missing_wallet_key_raises(monkeypatch):
    monkeypatch.delenv("VORQ_WALLET_KEY", raising=False)
    with pytest.raises(ValueError, match="wallet key"):
        WalletSigner()


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("VORQ_WALLET_KEY", KEY)
    assert WalletSigner().address == Account.from_key(KEY).address


def test_sign_nonce_is_recoverable():
    signer = WalletSigner(KEY)
    sig = signer.sign_nonce("nonce-123", 84532)
    msg = encode_typed_data(vorq_domain(84532), SESSION_TYPES, {"address": signer.address, "nonce": "nonce-123"})
    assert Account.recover_message(msg, signature=sig) == signer.address


# --- payload confidentiality (box cipher + DEK) ------------------------------

from vorqd._crypto import BoxCipher, open_dek, seal_to  # noqa: E402


def test_box_cipher_seal_roundtrip():
    box = BoxCipher.generate()
    ct = seal_to(box.public_key, b"payload")
    assert ct != b"payload"
    assert box.decrypt(ct) == b"payload"


def test_box_cipher_public_key_is_hex():
    box = BoxCipher.generate()
    assert isinstance(box.public_key, str) and len(box.public_key) == 64


def test_box_cipher_reads_env(monkeypatch):
    box = BoxCipher.generate()
    monkeypatch.setenv("VORQ_BOX_KEY", box._private_hex)
    assert BoxCipher().public_key == box.public_key


def test_missing_box_key_raises(monkeypatch):
    monkeypatch.delenv("VORQ_BOX_KEY", raising=False)
    with pytest.raises(ValueError, match="box key"):
        BoxCipher()


def test_open_dek_roundtrip():
    from nacl.secret import SecretBox
    from nacl.utils import random as nacl_random

    dek = nacl_random(SecretBox.KEY_SIZE)
    ct = bytes(SecretBox(dek).encrypt(b"broad payload"))
    assert open_dek(ct, dek) == b"broad payload"


def test_wrong_recipient_cannot_open():
    a, b = BoxCipher.generate(), BoxCipher.generate()
    ct = seal_to(a.public_key, b"secret")
    with pytest.raises(Exception):
        b.decrypt(ct)
