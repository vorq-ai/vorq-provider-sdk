---
title: Run a confidential provider
description: Configure a daemon that runs inside a confidential VM and publishes a fresh, attested payload key on every boot.
---

A confidential provider runs `vorqd` inside a confidential VM. Instead of a long-lived box
key, the daemon generates its payload key in memory at every boot and publishes the public half
together with attestation evidence that binds it to your operator wallet. Clients that submit
with confidential mode seal only to a key whose evidence they can verify.

## Configure it

Set `confidential: true` on the model and leave `box_key` out of the `provider` block:

```yaml
provider:
  wallet_key: env:VORQ_WALLET_KEY
  capacity: 2

models:
  - model: deepseek-ai/e2ee-deepseek-v4-pro:fp8
    confidential: true
    modality: text
    slas:
      "1h": { rate_in: "0.3", rate_out: "0.95" }
    backend:
      preset: openai-chat
      base_url: https://inference.example.com/v1
      model: deepseek-v4-pro
      api_key: env:INFERENCE_API_KEY
```

- `confidential: true` on any model makes the whole daemon confidential. A `box_key` is then a
  load error.
- Register only the wallet address with onboarding. The daemon writes the payload key and its
  evidence to your provider record at startup and waits until the record shows it.
- The model name has no effect; only the flag does.
- The `backend` block is unchanged. The inference service runs outside the VM and is reached
  over HTTPS like any other backend. Whether that service runs the model inside attested
  hardware of its own is a property of the service; the daemon does not check it.

[`confidential.yaml`](https://github.com/vorq-ai/vorq-provider-sdk/blob/main/docs/examples/confidential.yaml)
is the complete example.

## Attestation evidence

This package ships a mock attestation agent only. Its evidence is tagged `mock-cvm-v1`, and
production verifiers refuse it. Hardware attestation evidence is not produced by this package.

## Every restart rotates the key

The private key never leaves process memory. A restart generates a new key, and anything
sealed to the previous boot's key can no longer be opened, whether or not it was claimed.
Drain with `SIGTERM` before every restart so in-flight work settles while the old key still
exists.

The wallet rotation procedure in [Rotate keys](./rotate-keys.md#rotate-the-operator-wallet) applies
unchanged.

## Related

- [Encryption](../concepts/encryption.md)
- [Configuration reference](../reference/configuration.md#model-entry)
