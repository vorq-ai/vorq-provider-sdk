---
title: Encryption
description: How job payloads reach the daemon sealed, how the daemon verifies and opens them, and how results go back sealed to the client.
---

Job inputs and results are end-to-end encrypted between the client and the provider that runs
the job. The coordinator relays and stores only sealed bytes.

## The container

A client seals its payload into a container, which the coordinator stores on IPFS. The order
names the container by content id (`task_cid`) and commits to it:

```
container = version ‖ seed_wrap ‖ ciphertext          version = 0x01
c         = keccak256(version ‖ seed_wrap ‖ keccak256(ciphertext))
job_id    = keccak256(owner ‖ c)
```

A content id only says where to look. What makes the bytes trustworthy is that they reproduce
the job's own id. The daemon checks this before it claims, and again before it decrypts, so a
gateway that substitutes bytes, or a `seed_wrap` lifted from another order, is refused without
touching a key.

## The payload key

`seed_wrap` is a sealed box over a 32-byte seed. The payload key is derived from the seed and
the job's owner:

```
key = HKDF-SHA256(ikm = seed, salt = "", info = "vorq-dek" ‖ owner, length = 32)
```

Binding the key to the owner makes a copied `seed_wrap` useless under anyone else's order.

Who can open the wrap depends on the bid:

- A **designated** bid, addressed to one provider, seals the seed to that provider's box
  public key. The daemon opens it with `box_key` and derives the key itself.
- An **open** bid seals the seed to the coordinator's escrow key. After the claim, the daemon
  asks the escrow to release the key; the escrow checks the claim on-chain and answers with the
  key already derived for the job's owner.

The payload is decrypted with that key (a nonce-prefixed secretbox). Inside is the client's
envelope: `{"v": "vorq-env-v1", "owner", "result_key", "input"}`. The envelope must name the
job's owner and a valid result key; otherwise the job is handed back unrun.

## The result

The result is sealed to the envelope's `result_key` and sent as
`{"enc": "vorq-sealed-v1", "ciphertext": …}`. There is no unsealed path: an envelope without a
result key is refused. The sealed body also carries the job id, so a client can tell if a result
was paired with the wrong job.

## What the daemon keeps private

The daemon never logs a rendered backend request, so job inputs, including reference images,
stay out of your logs. Backend error messages go to your log only; the failure reason sent to
the network is a short code. The params `user` and `metadata` are never forwarded to a backend,
so a backend cannot link one client's jobs together.

## Confidential providers

A confidential provider has no long-lived box key. It generates one in memory on every boot and
publishes the public half with attestation evidence binding it to the operator wallet. See
[Run a confidential provider](../guides/run-a-confidential-provider.md).

## Related

- [Job lifecycle](./job-lifecycle.md)
- [Rotate keys](../guides/rotate-keys.md)
