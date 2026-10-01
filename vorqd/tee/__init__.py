"""Attestation for a confidential provider: this boot's payload key and the
evidence that binds it.

Everything TEE lives here, behind one import seam: core modules (scheduler,
backend, chain, coordinator, config) never import this package; ``cli`` imports
it lazily, only when some model in the config sets ``confidential: true``.
"""

from .agent import BootIdentity, MockAttestationAgent, boot_identity, report_data

__all__ = [
    "BootIdentity",
    "MockAttestationAgent",
    "boot_identity",
    "report_data",
]
