---
title: Rotate keys
description: Replace the operator wallet or the box key without stranding jobs.
---

A provider holds two long-lived keys: the operator wallet (`wallet_key`) and the box key
(`box_key`). VORQ provider onboarding holds the record of which keys belong to your provider
id, so every rotation has the same shape: drain, have the record updated, deploy the new key,
restart.

Your provider id, reputation and granted capacity stay with the id, not the key, so a rotation
does not change them.

## Rotate the operator wallet

1. Drain: stop the daemon with `SIGTERM` and let in-flight jobs settle (see
   [Stop and restart safely](./stop-and-restart-safely.md)).
2. Ask onboarding to move your provider id to the new wallet address.
3. Set the new key in `VORQ_WALLET_KEY` (or wherever `wallet_key` points).
4. Start the daemon.

Do not rotate under load. Once the record moves, every session signed by the old wallet is
invalid. A running daemon re-signs in once on a `401` with the key it still has, fails, and can
no longer claim or settle. The jobs it holds sit claimed until you deploy the new key and
restart, and any whose window closes in the meantime count as missed SLAs.

## Rotate the box key

1. Drain the daemon as above.
2. Generate a new key (see the [Quickstart](../quickstart.md#2-create-and-register-your-keys))
   and ask onboarding to record the new public key.
3. Set the new private key in `VORQ_BOX_KEY`.
4. Start the daemon.

The daemon refuses to start while its local box key does not match the key on record, so steps
2 and 3 must both be done before the restart.

Payloads sealed to the old key cannot be opened with the new one. That includes jobs still
waiting on the book that were sealed to you, not only jobs in flight. Draining first keeps that
window short.

## Related

- [Encryption](../concepts/encryption.md)
- [Run a confidential provider](./run-a-confidential-provider.md), whose payload key rotates on
  every restart
