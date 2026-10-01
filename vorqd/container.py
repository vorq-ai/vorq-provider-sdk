"""Container v1: the bytes an order names, and the one check that makes an
untrusted fetch safe.

::

    container = version ‖ seed_wrap ‖ ciphertext   version   = 0x01, 1 byte
                                                  seed_wrap = seal(recipient, SEED), 80 bytes
    c         = keccak256(version ‖ seed_wrap ‖ keccak256(ciphertext))
    job_id    = keccak256(owner ‖ c)
    dek       = HKDF-SHA256(ikm = seed, salt = b"", info = b"vorq-dek" ‖ owner20, L = 32)

The daemon fetches these bytes from a public gateway by CID. A CID is a locator
anyone may pin anything under, so the fetch proves nothing on its own — the job's
own name does. Re-derive ``keccak256(owner ‖ c)`` from the bytes that arrived and
compare it to the job id: **substituted ciphertext derives a different id, and a
``seed_wrap`` lifted from another order derives a different id too** (which is why
the wrap is hashed into ``c`` and not only the ciphertext). Both are refused
before a single byte is decrypted.

**The sealed 32 bytes are a seed, not the DEK.** A container is public, so its
``seed_wrap`` can be lifted verbatim into a fresh commitment under an attacker's
own address: every field of the resulting order is honest and nothing checkable
over public data refuses it. What refuses it is the derivation — the working key
is ``HKDF-SHA256`` of the seed under an ``info`` string carrying the **job's
owner**, read from the job and never from a request field, so a wrap opened
under the wrong owner yields a key that decrypts nothing. The cost of that fix
was zero change to the wrap: it is still 80 bytes, and
``tests/vectors/container-v1.json`` pins it as an opaque blob and asserts
nothing about its plaintext.

The whole module is arithmetic over ``bytes``. It reads no configuration and
opens no socket, so a refusal costs two keccaks and happens before any key is
touched. The wire format is pinned by ``tests/vectors/container-v1.json``, the
cross-repo vector file every implementation of this format is checked against;
the derivation, which no vector can pin because no vector carries a sealed
plaintext, is pinned against a digest the coordinator's own implementation
produced.
"""

from __future__ import annotations

import hashlib
import hmac

from eth_utils import keccak

#: Container v1. Byte 0 of every container, and the first byte of ``c``'s
#: preimage — so a v2 container can never be read as a v1 one by a v1 reader, and
#: flipping the byte fails the commitment before a claim is spent.
#:
#: There is deliberately no length field in the bytes: the version implies the
#: wrap's kind and its width together. A length in the container is
#: attacker-supplied data needing validation on every parse; a width in the code
#: is a constant with nothing to lie about.
CONTAINER_VERSION = 1

#: The width this build reads and writes: a sealed-box wrap of the 32-byte seed —
#: 32-byte ephemeral public key, the ciphertext, and the 16-byte MAC. Fixed width
#: is what makes the split a split.
SEED_WRAP_BYTES = 80

#: The shortest thing that is a container at all: the version byte and the wrap.
#: The ciphertext may legitimately be empty — ``keccak256(b"")`` is a real hash
#: and the vectors pin that case — but bytes too short to split are not a
#: container.
MIN_CONTAINER_BYTES = 1 + SEED_WRAP_BYTES

#: What the ciphertext costs over the plaintext it carries: a 24-byte nonce
#: prefixed and a 16-byte authenticator. A fact about the v1 format — the body is
#: ``SecretBox(dek).encrypt(plaintext)`` — and not about any one library, which is
#: why it is spelled here rather than imported. This module opens no box and must
#: keep it that way: the whole point of the arithmetic below is that it costs no
#: key. ``tests/test_container.py`` pins it against the cipher actually doing the
#: sealing, which is the one test that fails if the framing ever drifts.
SECRETBOX_OVERHEAD_BYTES = 40

#: Everything a container adds to the envelope inside it: the version byte, the
#: wrap, the nonce and the tag.
PLAINTEXT_OVERHEAD_BYTES = MIN_CONTAINER_BYTES + SECRETBOX_OVERHEAD_BYTES


#: The width of the sealed secret. Unchanged at 32 bytes — what changed is what
#: those bytes *are*: a seed the DEK is derived from, never the DEK itself.
SEED_LEN = 32

#: The HKDF ``info`` prefix, as **bytes**, spelled out at the one place the
#: derivation happens. It is a cross-language contract: the coordinator writes it
#: ``Buffer.from("vorq-dek", "utf8")``, and a side that padded it to a block or
#: re-encoded it would derive a different key and fail only at runtime, in a
#: provider, on a job that is already paid for.
#:
#: It carries **no version**. The container's version byte is the one version
#: namespace: it travels with the bytes it describes and it is committed by ``c``.
#: Seeds are freshly random per job, so no seed ever appears under two formats and
#: the label's only job is domain separation.
DEK_INFO_PREFIX = b"vorq-dek"

#: RFC 5869's ``salt`` for this derivation: **zero length, explicitly**. §2.2
#: then substitutes HashLen zero bytes inside HMAC's own key padding, which is
#: not the same thing as passing 32 zero bytes and not the same thing as
#: omitting the argument to a library whose default is something else.
DEK_SALT = b""


class ContainerError(Exception):
    """These bytes are not the container this job names.

    ``fault`` is the short, stable token the refusal is reported and metered
    under: ``too_short``, ``bad_version`` or ``commitment_mismatch``.
    """

    def __init__(self, fault: str, message: str) -> None:
        super().__init__(message)
        self.fault = fault


def split_container(container: bytes) -> tuple[bytes, bytes]:
    """``(seed_wrap, ciphertext)`` at the offsets byte 0 names.

    Length and version are checked **before** the split: Python slicing clamps out
    of range rather than raising, so without them an 80-byte buffer would split
    cleanly into a short wrap and an empty ciphertext and hash to a
    plausible-looking commitment. The wrap width is a code constant and never
    comes from the bytes.
    """
    if len(container) < MIN_CONTAINER_BYTES:
        raise ContainerError(
            "too_short",
            f"a container is at least {MIN_CONTAINER_BYTES} bytes — a version byte and an "
            f"{SEED_WRAP_BYTES}-byte seed_wrap — and this one is {len(container)}",
        )
    if container[0] != CONTAINER_VERSION:
        raise ContainerError(
            "bad_version",
            f"container byte 0 is 0x{container[0]:02x}; this build reads "
            f"0x{CONTAINER_VERSION:02x}",
        )
    return (
        container[1 : 1 + SEED_WRAP_BYTES],
        container[1 + SEED_WRAP_BYTES :],
    )


def sealed_plaintext_bytes(container_len: int) -> int:
    """How many bytes of envelope a v1 container of this length carries.

    Exact, not an estimate: every width in the format is a code constant, so
    ``len(plaintext) == len(container) - 121`` for every container this build
    reads. That is what makes the size of a sealed payload knowable **before a
    claim**, from bytes anyone may fetch and nobody need decrypt — the daemon can
    weigh a bid's declared ``units_in`` against what it actually carries while
    refusing is still free.

    Version-pinned by the only caller that matters: a container reaching the
    scheduler has already been through :func:`split_container`, which refuses any
    byte 0 that is not this build's version. A v2 with a different wrap width or a
    compressed body could never be measured here, because it would never get here.

    Floored at zero. The caller subtracts this from a declared allowance, and a
    negative would read as a payload smaller than nothing — quietly exempting the
    one input that is malformed.
    """
    return max(0, int(container_len) - PLAINTEXT_OVERHEAD_BYTES)


def commitment(container: bytes) -> str:
    """``c`` for these bytes: ``keccak256(version ‖ seed_wrap ‖ keccak256(ciphertext))``.

    Never one hash over the whole file. The bulk enters the commitment through
    its digest, so the preimage is always exactly ``1 + 80 + 32 = 113`` bytes
    whatever the payload weighs.

    ``split_container`` has already refused any byte 0 that is not this build's
    version, so the byte in the preimage is both the one the bytes carry and the
    one this build writes.
    """
    seed_wrap, ciphertext = split_container(container)
    return "0x" + keccak(bytes([CONTAINER_VERSION]) + seed_wrap + keccak(ciphertext)).hex()


def ciphertext_hash(ciphertext: bytes) -> bytes:
    """``keccak256(ciphertext)`` — how the bulk enters ``c``, and the only thing
    the escrow needs about a payload it must never see."""
    return keccak(ciphertext)


def content_job_id(owner: str, c: str | bytes) -> str:
    """The job id this commitment carries for this owner: ``keccak256(owner ‖ c)``.

    Twenty raw address bytes then the 32-byte word, packed — no padding, and no
    inner hash: ``c`` is already the commitment, and hashing it again would name
    a job nobody posted.
    """
    addr = owner[2:] if owner.lower().startswith("0x") else owner
    word = bytes.fromhex(c[2:] if str(c).lower().startswith("0x") else c) if isinstance(c, str) else c
    if len(word) != 32:
        raise ValueError(f"a commitment is 32 bytes, not {len(word)}")
    return "0x" + keccak(bytes.fromhex(addr) + word).hex()


def owner_bytes(owner: str | bytes) -> bytes:
    """The owner as 20 **raw** bytes, never as a hex string.

    Checksum casing is display-only, so a derivation over the text form would
    produce two different keys for one address depending on how each side
    spelled it — and the sides do spell it differently: this daemon reads a job
    row, the client holds a checksummed address from its wallet library, the
    coordinator reads a lowercase one off the chain.
    """
    raw = (
        bytes(owner)
        if isinstance(owner, (bytes, bytearray))
        else bytes.fromhex(owner[2:] if owner[:2].lower() == "0x" else owner)
    )
    if len(raw) != 20:
        raise ValueError(f"owner must be a 20-byte address, got {len(raw)} bytes")
    return raw


def _hkdf_sha256(*, ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-SHA256, extract-then-expand, in the stdlib.

    Written out rather than pulled from a dependency so the two steps are
    readable next to the inputs they are pinned against. ``length`` is never
    more than one block here, but the counter loop is the real thing anyway — a
    truncated implementation that happens to be right for L ≤ 32 is a trap for
    the next caller.
    """
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out = b""
    block = b""
    counter = 1
    while len(out) < length:
        # T(n) = HMAC(PRK, T(n-1) ‖ info ‖ n). `info` is inside every block and
        # not only the first: an expand that drops it produces a perfectly
        # well-formed 32 bytes that no other implementation reproduces.
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def derive_dek(seed: bytes, owner: str | bytes) -> bytes:
    """``HKDF-SHA256(ikm=seed, salt=b"", info=b"vorq-dek" ‖ owner20, L=32)``.

    The DEK for a container whose wrap sealed ``seed`` to a job owned by
    ``owner``. Run on the **designated** path, where this daemon unseals the
    wrap with its own box key: the owner comes off the job, never off a request
    field, so an attacker who lifted this wrap into their own order derives a
    different key and opens nothing.

    Not run on the open path — ``POST /release`` answers with the DEK already
    derived, because the coordinator derives it against the owner it read from
    chain. Deriving a second time there would produce a key of a key.

    Nothing local can check this function's output. Both sides of a job run it
    and never exchange the result, so the inputs are the contract; they are
    pinned byte for byte in ``tests/test_container.py`` against a digest the
    coordinator's implementation produced.
    """
    if len(seed) != SEED_LEN:
        raise ValueError(f"a seed is {SEED_LEN} bytes, got {len(seed)}")
    return _hkdf_sha256(
        ikm=bytes(seed),
        salt=DEK_SALT,
        info=DEK_INFO_PREFIX + owner_bytes(owner),
        length=32,
    )


def verify_container(owner: str, job_id: str, container: bytes) -> tuple[bytes, bytes]:
    """The gate: ``(seed_wrap, ciphertext)``, or a refusal — nothing in between.

    Call this before touching a key. Bytes that do not re-derive the job's own id
    are somebody else's, or nobody's: a gateway that substituted them, a pin that
    was replaced, or a wrap lifted from another order to make this job decrypt
    under a stranger's DEK. All three land here, and none reaches a cipher.
    """
    seed_wrap, ciphertext = split_container(container)
    c = commitment(container)
    derived = content_job_id(owner, c)
    if derived.lower() != str(job_id).lower():
        raise ContainerError(
            "commitment_mismatch",
            f"these bytes name job {derived}, not {job_id}: keccak256(owner ‖ c) over the "
            "fetched container does not reproduce the job's id",
        )
    return seed_wrap, ciphertext
