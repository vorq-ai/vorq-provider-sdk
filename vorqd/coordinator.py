"""The coordinator seam: the session handshake, and nothing else.

No job bytes come through here in either direction. Task bytes are
content-addressed and fetched from the untrusted blob surface
(:mod:`vorqd.blob`), where the job's own commitment, not a session, is what makes
them trustworthy; results — text and media alike — are sealed to the client's key
and ride the settle call itself, so the daemon has no file traffic to carry.

Talks to the VORQ API's ``/auth/*`` surface. Mints a ``vorq_sess_…`` bearer on
first use and refreshes it as it nears expiry, and answers the provider id the
node put on the handshake. :class:`~vorqd.node.NodeClient` shares this one
session and attaches that bearer to the two doors the node gates.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from ._crypto import Signer
from .errors import NotRegisteredError


class CoordinatorClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        signer: Signer,
        *,
        clock=time.time,
        refresh_margin_s: float = 60,
    ) -> None:
        self._client = client
        self._api_url = api_url.rstrip("/")
        self._signer = signer
        self._clock = clock
        self._refresh_margin = refresh_margin_s
        self._token: str | None = None
        self._expires_at: float = 0
        self._provider_id: int | None = None
        self._lock = asyncio.Lock()

    @property
    def address(self) -> str:
        return self._signer.address

    @property
    def provider_id(self) -> int | None:
        """The provider id bound to the last successful handshake.

        ``None`` before the first successful handshake, or if the
        coordinator omitted it (old/fake servers). Reflects the LAST
        successful handshake regardless of :meth:`invalidate` — it does not
        get cleared when the cached token does, since it only changes when a
        new handshake actually succeeds.
        """
        return self._provider_id

    async def token(self) -> str:
        async with self._lock:
            now = self._clock()
            if self._token is not None and now < self._expires_at - self._refresh_margin:
                return self._token
            return await self._handshake()

    def invalidate(self) -> None:
        """Clear the cached token so the next :meth:`token` call re-handshakes.

        Call this on any 401 from an API call: a wallet rotation or
        coordinator restart can void a cached token mid-lifetime, well
        before its stated expiry. Idempotent — safe to call with nothing
        cached.
        """
        self._token = None
        self._expires_at = 0

    async def _handshake(self) -> str:
        nonce_resp = await self._client.get(
            f"{self._api_url}/auth/nonce", params={"address": self._signer.address}
        )
        nonce_resp.raise_for_status()
        minted = nonce_resp.json()
        nonce, chain_id = minted["nonce"], int(minted["chain_id"])

        signature = self._signer.sign_nonce(nonce, chain_id)
        sess_resp = await self._client.post(
            f"{self._api_url}/auth/session",
            json={
                "address": self._signer.address,
                "nonce": nonce,
                "signature": signature,
                "role": "provider",
            },
        )
        if sess_resp.status_code == 403:
            err = sess_resp.json().get("error", {})
            if err.get("code") == "not_registered":
                raise NotRegisteredError(err.get("message", ""))
        sess_resp.raise_for_status()
        data = sess_resp.json()
        self._token = data["token"]
        self._expires_at = float(data["expires_at"])
        self._provider_id = data.get("provider_id")
        return self._token
