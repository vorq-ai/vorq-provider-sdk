"""Container v1 against the cross-repo vectors.

``tests/vectors/container-v1.json`` is copied verbatim from the file the
coordinator generates from its shipped implementation. The client SDK, the
coordinator and this daemon each split and commit these bytes with their own
code; the vectors are what makes their agreement checkable rather than
coincidental. Nothing here re-derives the format from prose — every expected
value is read out of the file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from vorqd._crypto import BoxCipher
from vorqd.container import (
    CONTAINER_VERSION,
    DEK_INFO_PREFIX,
    DEK_SALT,
    MIN_CONTAINER_BYTES,
    PLAINTEXT_OVERHEAD_BYTES,
    SECRETBOX_OVERHEAD_BYTES,
    SEED_LEN,
    SEED_WRAP_BYTES,
    ContainerError,
    commitment,
    content_job_id,
    derive_dek,
    sealed_plaintext_bytes,
    split_container,
    verify_container,
)

from .conftest import seal_container

#: Somebody to seal to. The identity these tests pin is about widths, so which
#: key holds the seed is irrelevant — only that a real wrap was produced.
_RECIPIENT = BoxCipher.generate()

VECTORS = json.loads((Path(__file__).parent / "vectors" / "container-v1.json").read_text())
CASES = VECTORS["cases"]
REFUSALS = VECTORS["refusals"]
KDF = VECTORS["kdf"]


def _b(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str[2:] if hex_str.startswith("0x") else hex_str)


def test_constants_match_the_vector_file():
    assert CONTAINER_VERSION == VECTORS["constants"]["version"]
    assert "0x" + bytes([CONTAINER_VERSION]).hex() == VECTORS["constants"]["version_byte"]
    assert SEED_WRAP_BYTES == VECTORS["constants"]["wrap_bytes"]
    assert MIN_CONTAINER_BYTES == VECTORS["constants"]["min_container_bytes"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_split_is_at_the_fixed_offsets(case):
    seed_wrap, ciphertext = split_container(_b(case["container"]))
    assert seed_wrap == _b(case["seed_wrap"])
    assert ciphertext == _b(case["ciphertext"])
    assert len(seed_wrap) == SEED_WRAP_BYTES


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_commitment_matches_the_vector(case):
    assert commitment(_b(case["container"])) == case["c"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_job_id_matches_the_vector(case):
    assert content_job_id(case["owner"], case["c"]) == case["job_id"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_verify_accepts_the_bytes_the_job_names(case):
    seed_wrap, ciphertext = verify_container(case["owner"], case["job_id"], _b(case["container"]))
    assert seed_wrap == _b(case["seed_wrap"]) and ciphertext == _b(case["ciphertext"])


@pytest.mark.parametrize("bad", REFUSALS, ids=[r["name"] for r in REFUSALS])
def test_labelled_refusals_are_refused_with_their_fault(bad):
    container = _b(bad["container"])
    if bad["fault"] == "commitment_mismatch":
        # Well formed, and it commits to *something* — just never to this job.
        owner = CASES[0]["owner"]
        job_id = content_job_id(owner, bad["c"])
        with pytest.raises(ContainerError) as exc:
            verify_container(owner, job_id, container)
    else:
        with pytest.raises(ContainerError) as exc:
            commitment(container)
    assert exc.value.fault == bad["fault"]


def test_a_wrap_lifted_from_another_order_commits_to_something_else():
    # The property that justifies hashing the wrap into `c` at all: identical
    # ciphertext under a stolen wrap is a different commitment, so it is a
    # different job id, so it can never be claimed under the order it was lifted
    # into. This is the check that makes an untrusted gateway safe.
    swap = VECTORS["wrap_swap"]
    assert commitment(_b(swap["container"])) == swap["c"] != swap["differs_from"]


def test_the_owner_is_inside_the_job_id():
    c = CASES[0]["c"]
    other = "0x" + "22" * 20
    assert content_job_id(CASES[0]["owner"], c) != content_job_id(other, c)
    # Address case is not part of the derivation — the bytes are.
    assert content_job_id(CASES[0]["owner"].lower(), c) == CASES[0]["job_id"]


def test_a_commitment_is_never_hashed_twice():
    # `content_job_id` takes the commitment, not the container: handing it the
    # raw bytes must not silently produce some other job's name.
    with pytest.raises(ValueError, match="32 bytes"):
        content_job_id(CASES[0]["owner"], _b(CASES[0]["container"]))


# --- the sealed 32 bytes are a SEED, and the DEK is derived from it ----------
#
# The vectors above pin the wrap as an opaque blob and say nothing about what it
# holds: the wrap is still 80 bytes and no case pins the sealed plaintext, so
# every assertion in this file still passes under an implementation that treats
# the wrap's plaintext as the DEK itself. That defect decrypts nothing, on a job
# that is already paid for, and only in production — so the derivation is pinned
# here as arithmetic, and against a digest produced by the OTHER implementation.


OWNER = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"


def _hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-SHA256, rebuilt from `hmac`/`hashlib` here.

    Deliberately a second implementation rather than a call into the module
    under test: a round trip through one implementation passes under any KDF at
    all, including one that ignores the owner entirely.
    """
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def test_the_seed_is_thirty_two_bytes_and_the_wrap_did_not_widen():
    assert SEED_LEN == 32
    # The whole point of deriving rather than widening the wrap: the wrap's width
    # did not move, so the vector file above is still the contract.
    assert SEED_WRAP_BYTES == 80 == VECTORS["constants"]["wrap_bytes"]


def test_the_info_prefix_is_utf8_vorq_dek_unpadded():
    """Stated as bytes, because the coordinator states it as bytes.

    ``Buffer.from("vorq-dek", "utf8")`` is 8 bytes. A side that padded it to a
    block, or encoded it UTF-16, would derive a different key and fail only at
    runtime, in this daemon, on a job the client already paid for.

    There is no ``-v1`` on it: the container's version byte is the one version
    namespace, and two namespaces declaring the same fact is what C3 removed.
    """
    assert DEK_INFO_PREFIX == b"vorq-dek"
    assert len(DEK_INFO_PREFIX) == 8
    # The vectors carry the label as text, not hex, so this is the encoding step
    # itself — the one the UTF-16 mistake above would fail.
    assert DEK_INFO_PREFIX == KDF["info_prefix"].encode("utf-8")


def test_the_salt_is_zero_length():
    """RFC 5869 §2.2 substitutes HashLen zeros for an absent salt.

    An empty salt and 32 zero bytes are genuinely the same HMAC key — the
    padding makes them so — and that equivalence is asserted rather than
    guessed at, because a reader who assumes otherwise reaches for the wrong
    fix. What is *not* the same operation is putting zeros in the **ikm**, which
    produces a well-formed key that decrypts nothing.
    """
    assert DEK_SALT == b""
    assert len(DEK_SALT) == 0
    assert DEK_SALT == _b(KDF["salt"])
    info = DEK_INFO_PREFIX + bytes.fromhex(OWNER[2:])
    assert derive_dek(bytes(range(32)), OWNER) == _hkdf(bytes(range(32)), b"\x00" * 32, info, 32)
    assert derive_dek(bytes(range(32)), OWNER) != _hkdf(b"\x00" * 32, b"", info, 32)


def test_the_derivation_matches_hkdf_sha256_input_for_input():
    seed = bytes(range(32))
    owner_raw = bytes.fromhex(OWNER[2:])
    assert derive_dek(seed, OWNER) == _hkdf(
        seed, b"", b"vorq-dek" + owner_raw, 32
    )


def test_the_derivation_is_pinned_to_the_vectors_kdf_block():
    """The value itself, produced by the implementation on the other side.

    Every other assertion here rebuilds the expectation from the same inputs
    this module uses, so a coordinated edit to both would keep them green. This
    digest is ``node:crypto.hkdfSync("sha256", seed, <empty>, info, 32)`` — the
    exact call the coordinator's ``/release`` makes — and it is the one
    assertion a Python-only mistake cannot survive.

    It lives **inside ``container-v1.json``**, in the same file as the layout it
    belongs to, instead of being transcribed by hand into three test bodies.
    """
    assert derive_dek(_b(KDF["seed"]), KDF["owner"]) == _b(KDF["dek"])
    assert KDF["info_prefix"].encode("utf-8") == DEK_INFO_PREFIX
    assert _b(KDF["salt"]) == DEK_SALT


def test_the_owner_enters_as_twenty_raw_bytes_and_never_as_a_string():
    """Checksum casing is display-only; a KDF over the text form is not.

    This daemon reads the owner off a job row and the client holds a checksummed
    address from `eth-account`. If either spelling reached the digest the two
    would derive different keys for one address.
    """
    seed = bytes(range(32))
    assert derive_dek(seed, OWNER) == derive_dek(seed, OWNER.lower())
    assert derive_dek(seed, OWNER) == derive_dek(seed, bytes.fromhex(OWNER[2:]))
    assert derive_dek(seed, OWNER) != _hkdf(seed, b"", DEK_INFO_PREFIX + OWNER.encode(), 32)


def test_a_different_owner_derives_a_different_key():
    """The wrap-lifting defence, in one assertion.

    An attacker lifts a wrap verbatim, mints a fresh commitment around it and
    posts their own job under their own address. Every field of that job is
    honest, so nothing over public data refuses it — the derivation does,
    because the info string carries the owner and the attacker's is not the
    victim's. It holds identically on this daemon's designated path, where no
    coordinator is in the loop at all.
    """
    seed = bytes(range(32))
    assert derive_dek(seed, OWNER) != derive_dek(seed, "0x" + "22" * 20)


def test_a_seed_of_the_wrong_width_is_refused_rather_than_stretched():
    with pytest.raises(ValueError, match="32 bytes"):
        derive_dek(b"\x01" * 31, OWNER)


def test_an_owner_of_the_wrong_width_is_refused_rather_than_padded():
    with pytest.raises(ValueError, match="20-byte"):
        derive_dek(bytes(range(32)), "0x1234")


# --- how much envelope a container carries, without opening it ----------------
#
# The size of a sealed payload is knowable from the container's own length, and
# that is what lets the daemon judge a bid's declared `units_in` BEFORE it claims
# anything — no key, no decryption, and nothing spent on a refusal. The identity
# is only worth anything if it is exact, so both halves of it are pinned here:
# the cipher's contribution against the cipher itself, and the whole sum against
# real sealed bytes.


def test_the_secretbox_overhead_constant_is_the_ciphers_own():
    """`container.py` must not import nacl — its docstring promises arithmetic over
    bytes and nothing else, which is what keeps a refusal free of any key. So the
    cipher's framing is a constant there, and this is the one test that fails if
    it ever drifts from the library actually doing the sealing.
    """
    from nacl.secret import SecretBox

    assert SECRETBOX_OVERHEAD_BYTES == SecretBox.NONCE_SIZE + SecretBox.MACBYTES


def test_the_plaintext_overhead_is_the_version_byte_the_wrap_and_the_framing():
    assert PLAINTEXT_OVERHEAD_BYTES == MIN_CONTAINER_BYTES + SECRETBOX_OVERHEAD_BYTES
    assert PLAINTEXT_OVERHEAD_BYTES == 121


@pytest.mark.parametrize("size", [0, 1, 20, 181, 100_000])
def test_a_container_carries_exactly_the_plaintext_its_length_names(size):
    """The identity the pre-claim gate rests on, against bytes a client really sealed."""
    plaintext = b"x" * size
    container = seal_container(plaintext, recipient=_RECIPIENT.public_key, owner=OWNER)

    assert sealed_plaintext_bytes(len(container)) == size


def test_bytes_too_short_to_be_a_container_carry_no_payload():
    """Never negative. The caller is `_poll_model`, which subtracts this from a
    declared allowance — a negative would read as a payload smaller than nothing
    and quietly turn the guard off for the one input that is malformed.
    """
    assert sealed_plaintext_bytes(0) == 0
    assert sealed_plaintext_bytes(PLAINTEXT_OVERHEAD_BYTES - 1) == 0
    assert sealed_plaintext_bytes(PLAINTEXT_OVERHEAD_BYTES) == 0
