"""Shared test helpers for the vorqd suite.

Provides a routed ``httpx.MockTransport`` factory so unit tests can stand in for
the coordinator/chain surface (``/evm/*``, ``/v1/files``, ``/auth/*``) without a
live emulator, mirroring the client SDK's test conventions.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

import httpx


def mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def json_response(status: int, body, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers or {})


def error_response(status: int, type_: str, message: str = "", *, code: str | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": message or type_, "type": type_, "param": None, "code": code}},
        headers={"x-vorq-retryable": "false"},
    )


def req_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def req_form(request: httpx.Request) -> tuple[dict[str, str], tuple[str, bytes] | None]:
    """The fields of a ``multipart/form-data`` body, in order, and its one file part.

    Asserts the order the files door lives by — every field ahead of the file
    part — so a body that breaks it fails here rather than in the node.
    """
    ctype = request.headers["content-type"]
    assert ctype.startswith("multipart/form-data; boundary="), ctype
    delim = b"--" + ctype.split("boundary=", 1)[1].encode()
    assert request.content.endswith(delim + b"--\r\n")
    fields: dict[str, str] = {}
    file: tuple[str, bytes] | None = None
    for part in request.content.split(delim)[1:-1]:
        head, _, data = part.removeprefix(b"\r\n").partition(b"\r\n\r\n")
        disposition = next(h for h in head.split(b"\r\n") if h.lower().startswith(b"content-disposition"))
        name = re.search(rb'name="([^"]*)"', disposition).group(1).decode()
        assert file is None, f"{name} arrived after the file part"
        if b"filename=" in disposition:
            file = (name, data.removesuffix(b"\r\n"))
        else:
            fields[name] = data.removesuffix(b"\r\n").decode()
    return fields, file


def req_op(request: httpx.Request) -> dict:
    """One ``POST /evm/ops`` body as the flat dict the node reads: JSON for every
    op, a settle included — its ``result`` rides as a base64 string."""
    return req_json(request)


def seal_container(plaintext: bytes, *, recipient: str, owner: str, seed: bytes | None = None) -> bytes:
    """The container a client pins: ``version ‖ seal(recipient, SEED) ‖ SecretBox(DEK, plaintext)``.

    ``recipient`` is the Curve25519 public key the **seed** is sealed to — the
    provider's own box key on a designated bid, the coordinator's escrow key on
    an open one. Same bytes either way; only the holder differs.

    The sealed 32 bytes are a seed and the working key is
    ``derive_dek(seed, owner)``, so ``owner`` is required here: a helper that
    encrypted under the seed itself would produce perfectly valid containers
    that every commitment test in this suite accepts and that no correct
    provider can decrypt — the exact defect the derivation exists to make
    impossible, and one no vector file can catch, because no vector pins a
    sealed plaintext.
    """
    from nacl.secret import SecretBox
    from nacl.utils import random as nacl_random

    from vorqd._crypto import seal_to
    from vorqd.container import CONTAINER_VERSION, SEED_LEN, SEED_WRAP_BYTES, derive_dek

    seed = seed if seed is not None else nacl_random(SEED_LEN)
    wrap = seal_to(recipient, seed)
    assert len(wrap) == SEED_WRAP_BYTES
    dek = derive_dek(seed, owner)
    return bytes([CONTAINER_VERSION]) + wrap + bytes(SecretBox(dek).encrypt(plaintext))


def job_id_of(owner: str, container: bytes) -> str:
    """The job id these container bytes name for this owner."""
    from vorqd.container import commitment, content_job_id

    return content_job_id(owner, commitment(container))


def fake_cid(raw: bytes) -> str:
    """A content-derived opaque locator for tests. Real names are minted by the
    filestore that pins the bytes — nothing in the SDK computes or parses one —
    so tests need only a stable per-content string, not any CID encoding."""
    return "cid-test-" + hashlib.sha256(raw).hexdigest()
