"""CoordinatorClient: the session handshake."""

from __future__ import annotations

import json

import httpx
import pytest

from vorqd._crypto import WalletSigner
from vorqd.coordinator import CoordinatorClient
from vorqd.errors import NotRegisteredError

API = "http://coord.test"


class Fake:
    """Routed handler standing in for /auth/*."""

    def __init__(self, now: float = 1_000.0, provider_id: int | None = 42, registered: bool = True):
        self.now = now
        self.nonce_calls = 0
        self.session_calls = 0
        self.session_ttl = 86_400
        self.provider_id = provider_id
        self.registered = registered

    def clock(self) -> float:
        return self.now

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/auth/nonce":
            self.nonce_calls += 1
            assert req.url.params["address"].startswith("0x")
            return httpx.Response(200, json={"nonce": f"n{self.nonce_calls}", "expires_at": self.now + 300, "chain_id": 84532})
        if path == "/auth/session":
            self.session_calls += 1
            body = json.loads(req.content)
            assert body["role"] == "provider"
            if not self.registered:
                return httpx.Response(
                    403,
                    json={"error": {"message": "Wallet is not registered as a provider.",
                                     "type": "authentication_error", "code": "not_registered"}},
                )
            resp = {"token": f"vorq_sess_{self.session_calls}", "expires_at": self.now + self.session_ttl}
            if self.provider_id is not None:
                resp["provider_id"] = self.provider_id
            return httpx.Response(200, json=resp)
        return httpx.Response(404, json={"error": {"message": "no route", "type": "not_found"}})


def make_client(fake: Fake) -> CoordinatorClient:
    ac = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return CoordinatorClient(ac, API, WalletSigner.generate(), clock=fake.clock, refresh_margin_s=60)


async def test_first_token_handshakes_then_caches():
    fake = Fake()
    cc = make_client(fake)
    t1 = await cc.token()
    assert t1 == "vorq_sess_1"
    assert (fake.nonce_calls, fake.session_calls) == (1, 1)
    t2 = await cc.token()  # within validity -> no new handshake
    assert t2 == "vorq_sess_1"
    assert (fake.nonce_calls, fake.session_calls) == (1, 1)


async def test_token_refreshes_after_expiry():
    fake = Fake()
    cc = make_client(fake)
    await cc.token()
    fake.now += fake.session_ttl  # push past expiry (minus margin)
    t = await cc.token()
    assert t == "vorq_sess_2"
    assert fake.session_calls == 2


async def test_job_bytes_are_not_a_coordinator_concern():
    # Task bytes are content-addressed and fetched from the public blob surface,
    # verified against the job's own commitment — the session plays no part. A
    # result, media included, is sealed to the client's key and rides the settle
    # call, so there is no file surface to upload to or read back from either.
    for gone in ("download_input", "download_file", "upload_result"):
        assert not hasattr(CoordinatorClient, gone)


async def test_handshake_sends_provider_role_and_captures_provider_id():
    fake = Fake(provider_id=7)
    cc = make_client(fake)
    assert cc.provider_id is None  # nothing before the first handshake
    await cc.token()
    assert cc.provider_id == 7


async def test_provider_id_none_when_server_omits_it():
    fake = Fake(provider_id=None)
    cc = make_client(fake)
    await cc.token()
    assert cc.provider_id is None


async def test_not_registered_raises_typed_error_and_caches_nothing():
    fake = Fake(registered=False)
    cc = make_client(fake)
    with pytest.raises(NotRegisteredError):
        await cc.token()
    assert fake.session_calls == 1
    # No token cached: a later, successful handshake still works.
    fake.registered = True
    t = await cc.token()
    assert t == "vorq_sess_2"


async def test_invalidate_forces_rehandshake_on_next_token_call():
    fake = Fake()
    cc = make_client(fake)
    await cc.token()
    assert (fake.nonce_calls, fake.session_calls) == (1, 1)
    cc.invalidate()
    t = await cc.token()
    assert t == "vorq_sess_2"
    assert (fake.nonce_calls, fake.session_calls) == (2, 2)


async def test_invalidate_is_idempotent_and_safe_with_nothing_cached():
    fake = Fake()
    cc = make_client(fake)
    cc.invalidate()
    cc.invalidate()
    t = await cc.token()
    assert t == "vorq_sess_1"
    assert (fake.nonce_calls, fake.session_calls) == (1, 1)
