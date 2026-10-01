"""The open-bid release seam: the frozen vocabulary, and the request it signs."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from eth_account import Account
from eth_account.messages import encode_typed_data
from nacl.encoding import HexEncoder
from nacl.public import PrivateKey, PublicKey, SealedBox

from vorqd._crypto import WalletSigner
from vorqd.container import SEED_WRAP_BYTES, ciphertext_hash, derive_dek
from vorqd.escrow import (
    RELEASE_REFUSALS,
    RELEASE_TYPES,
    HttpEscrowRelease,
    ReleaseRefused,
    release_domain,
)
from vorqd.types import EvmJob

ESCROW_URL = "https://escrow.test"
OWNER = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"
JOB_ID = "0x" + "ab" * 32
CHAIN_ID = 31337


def a_job() -> EvmJob:
    return EvmJob(job_id=JOB_ID, model="m", state="Claimed", sla="1h", created_at=0,
                  owner=OWNER, designated=0)


# --- the vocabulary ---------------------------------------------------------


def test_the_vocabulary_is_ten_codes_including_escrow_handed_over():
    """Nine described a holder that still has its keys. The tenth describes one
    that handed them to a successor — which none of the other nine can say
    without lying about what the caller should do next."""
    assert len(RELEASE_REFUSALS) == 10
    assert "escrow_handed_over" in RELEASE_REFUSALS
    assert set(RELEASE_REFUSALS) == {
        "bad_container", "wrap_mismatch", "no_claim", "not_claimed", "wrong_wallet",
        "stale_issued_at", "unseal_failed", "escrow_key_lost", "escrow_handed_over",
        "escrow_unavailable",
    }
    assert len(set(RELEASE_REFUSALS)) == len(RELEASE_REFUSALS)


def test_every_refusal_is_non_retryable_and_carries_its_code_verbatim():
    for code in RELEASE_REFUSALS:
        exc = ReleaseRefused(code, "detail")
        assert exc.code == code and exc.known and not exc.retryable
        assert code in str(exc)


def test_an_unknown_code_is_surfaced_verbatim_and_not_collapsed():
    """A code this build has never heard of must reach the operator as itself.

    Mapping it onto the nearest known one would report a condition that did not
    happen, and the nearest known one is exactly where an operator would look
    for a cause that is not there.
    """
    exc = ReleaseRefused("escrow_on_fire")
    assert exc.code == "escrow_on_fire"
    assert not exc.known          # this build does not know it...
    assert not exc.retryable      # ...so it is not retried, per the ruling
    assert exc.code not in RELEASE_REFUSALS


# --- the request ------------------------------------------------------------


class FakeEscrowServer:
    """The escrow door: records what it was asked, answers what it is told to."""

    def __init__(self, *, seed: bytes | None = None, status: int = 200, code: str | None = None,
                 body: dict | None = None, owner: str = OWNER):
        self.cipher = PrivateKey.generate()
        self.seed = seed if seed is not None else bytes(range(32))
        self.status = status
        self.code = code
        self.body = body
        # The owner the escrow derives under. The real one reads it off CHAIN,
        # from the row `jobId` was minted from — so this fake holds its own rather
        # than believing the body, which is the whole reason the body no longer
        # carries one.
        self.owner = owner
        self.requests: list[dict] = []

    @property
    def public_key(self) -> str:
        return self.cipher.public_key.encode(HexEncoder).decode()

    def wrap(self) -> bytes:
        return bytes(SealedBox(self.cipher.public_key).encrypt(self.seed))

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/release"
        payload = json.loads(req.content)
        self.requests.append(payload)
        if self.status != 200:
            if self.body is not None:
                return httpx.Response(self.status, json=self.body)
            return httpx.Response(
                self.status,
                json={"error": {"message": "no", "type": "invalid_request_error", "code": self.code}},
                headers={"x-vorq-retryable": "false"},
            )
        # What the real escrow does: unseal the seed, derive against the owner it
        # read from CHAIN, and re-seal the derived key to the response pubkey.
        seed = SealedBox(self.cipher).decrypt(base64.b64decode(payload["seed_wrap"]))
        dek = derive_dek(seed, self.owner)
        recipient = PublicKey(payload["response_pubkey"].encode(), encoder=HexEncoder)
        sealed = bytes(SealedBox(recipient).encrypt(dek))
        return httpx.Response(200, json={"dek_sealed": base64.b64encode(sealed).decode()})


def a_client(server: FakeEscrowServer, signer: WalletSigner | None = None, *, clock=lambda: 1_700_000_000.0):
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return HttpEscrowRelease(http, ESCROW_URL, signer or WalletSigner.generate(), CHAIN_ID, clock=clock)


async def test_release_returns_the_already_derived_dek_and_never_derives_again():
    """The open path does not derive. The coordinator did, against the owner it
    read from chain; deriving here would produce a key of a key."""
    server = FakeEscrowServer()
    dek = await a_client(server).release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    assert dek == derive_dek(server.seed, OWNER)
    assert dek != derive_dek(dek, OWNER)   # the mistake this test exists to catch


async def test_the_request_carries_the_wrap_and_the_digest_and_never_the_payload():
    server = FakeEscrowServer()
    ct = b"a payload the escrow must never see"
    wrap = server.wrap()
    await a_client(server).release(a_job(), wrap, ciphertext_hash(ct))
    sent = server.requests[0]
    assert set(sent) == {"job_id", "seed_wrap", "ct_hash", "response_pubkey",
                         "issued_at", "signature"}
    assert "owner" not in sent
    assert base64.b64decode(sent["seed_wrap"], validate=True) == wrap
    assert len(wrap) == SEED_WRAP_BYTES
    assert sent["ct_hash"] == "0x" + ciphertext_hash(ct).hex()
    assert base64.b64encode(ct).decode() not in json.dumps(sent)
    assert ct.decode() not in json.dumps(sent)


async def test_the_signature_recovers_to_the_provider_over_the_escrow_domain():
    """Third namespace, on purpose: a captured session login must not be a
    release and a captured chain op must not be either."""
    server = FakeEscrowServer()
    signer = WalletSigner.generate()
    await a_client(server, signer).release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    sent = server.requests[0]
    message = {
        "jobId": sent["job_id"],
        "seedWrap": "0x" + base64.b64decode(sent["seed_wrap"]).hex(),
        "ctHash": sent["ct_hash"],
        "responsePubkey": "0x" + sent["response_pubkey"],
        "issuedAt": sent["issued_at"],
    }
    recovered = Account.recover_message(
        encode_typed_data(release_domain(CHAIN_ID), RELEASE_TYPES, message),
        signature=sent["signature"],
    )
    assert recovered == signer.address
    # And the domain is not the registries' — no verifyingContract, its own name.
    assert "verifyingContract" not in release_domain(CHAIN_ID)
    assert release_domain(CHAIN_ID)["name"] == "VORQ Escrow"


async def test_the_response_key_is_fresh_per_request_and_is_inside_the_signature():
    """A long-lived response key would make every past answer readable by anyone
    who ever obtains it; a response key outside the signature would make a
    captured request a bearer token for the DEK."""
    server = FakeEscrowServer()
    client = a_client(server)
    await client.release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    await client.release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    first, second = server.requests
    assert first["response_pubkey"] != second["response_pubkey"]
    assert first["signature"] != second["signature"]
    assert [f["name"] for f in RELEASE_TYPES["Release"]] == [
        "jobId", "seedWrap", "ctHash", "responsePubkey", "issuedAt"]


def test_the_release_type_string_is_the_coordinators_and_carries_no_owner():
    """EIP-712 hashes the TYPE, so this string is the whole contract.

    `jobId` is keccak256(owner ‖ c): the owner is already committed by the id,
    and the escrow derives under the one it reads from chain. A copy in the
    request could only ever disagree with the copy that counts — and the
    coordinator kept a narrow-signature function alive purely to keep the body's
    owner out of scope of the KDF, a defence with nothing left to defend once the
    field is gone.
    """
    members = ",".join(f"{f['type']} {f['name']}" for f in RELEASE_TYPES["Release"])
    assert f"Release({members})" == (
        "Release(bytes32 jobId,bytes seedWrap,bytes32 ctHash,bytes32 responsePubkey,uint64 issuedAt)"
    )
    assert not any(f["name"] == "owner" for f in RELEASE_TYPES["Release"])


async def test_issued_at_is_now():
    server = FakeEscrowServer()
    await a_client(server, clock=lambda: 1_234_567_890.9).release(
        a_job(), server.wrap(), ciphertext_hash(b"ct"))
    assert server.requests[0]["issued_at"] == 1_234_567_890


@pytest.mark.parametrize("code", RELEASE_REFUSALS)
async def test_each_frozen_code_arrives_as_itself(code):
    server = FakeEscrowServer(status=400, code=code)
    with pytest.raises(ReleaseRefused) as exc:
        await a_client(server).release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    assert exc.value.code == code and not exc.value.retryable


async def test_a_code_this_build_does_not_know_is_not_collapsed_into_a_neighbour():
    server = FakeEscrowServer(status=400, code="some_future_code")
    with pytest.raises(ReleaseRefused) as exc:
        await a_client(server).release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    assert exc.value.code == "some_future_code" and not exc.value.known


async def test_a_failure_carrying_no_code_is_a_transport_error_not_a_refusal():
    """The node's own failure to serve — a 503 with no chain behind it, a
    gateway — is not the escrow's verdict on this job. Dressing it up as a
    refusal would tell the daemon to give the claim back over an outage."""
    server = FakeEscrowServer(status=503, body={"error": {"message": "no chain",
                                                          "type": "chain_unreachable"}})
    with pytest.raises(httpx.HTTPStatusError):
        await a_client(server).release(a_job(), server.wrap(), ciphertext_hash(b"ct"))


async def test_an_answer_that_does_not_open_with_this_requests_key_is_unseal_failed():
    server = FakeEscrowServer()

    def stranger(req: httpx.Request) -> httpx.Response:
        sealed = bytes(SealedBox(PrivateKey.generate().public_key).encrypt(b"\x00" * 32))
        return httpx.Response(200, json={"dek_sealed": base64.b64encode(sealed).decode()})

    http = httpx.AsyncClient(transport=httpx.MockTransport(stranger))
    client = HttpEscrowRelease(http, ESCROW_URL, WalletSigner.generate(), CHAIN_ID)
    with pytest.raises(ReleaseRefused) as exc:
        await client.release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    assert exc.value.code == "unseal_failed"


async def test_a_wrap_of_the_wrong_width_never_reaches_the_escrow():
    server = FakeEscrowServer()
    with pytest.raises(ReleaseRefused) as exc:
        await a_client(server).release(a_job(), b"\x01" * 79, ciphertext_hash(b"ct"))
    assert exc.value.code == "bad_container"
    assert server.requests == []   # not one byte was sent


async def test_the_chain_id_may_arrive_late_and_is_read_once():
    """The chain id comes from `GET /evm/chain`, which the wiring cannot read
    synchronously — so it may be handed over as an awaitable, resolved once."""
    server = FakeEscrowServer()
    reads = []

    async def chain_id() -> int:
        reads.append(1)
        return CHAIN_ID

    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    client = HttpEscrowRelease(http, ESCROW_URL, WalletSigner.generate(), chain_id)
    await client.release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    await client.release(a_job(), server.wrap(), ciphertext_hash(b"ct"))
    assert len(reads) == 1
    assert await client.chain_id() == CHAIN_ID
