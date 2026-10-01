"""The attestation-agent seam: produce evidence binding a key to this boot.

``hardware`` and ``mock`` implementations sit behind one interface; daemon code
above it is identical. The hardware agent exists only inside the measured CVM
image (which bakes in its platform deps); this package ships the mock, whose
evidence is structurally identical but carries a distinct type tag — the client
verifier accepts it only in mock mode (mock honesty rule).
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .._crypto import BoxCipher

MOCK_CVM_MEASUREMENT = hashlib.sha256(b"vorq-mock-cvm-image-v1").hexdigest()


def report_data(box_public_key: str, wallet_address: str) -> str:
    """The binding that makes evidence sound: ``sha256(box_pub ‖ wallet)`` hex.

    Raw bytes on both sides — the 32-byte Curve25519 key and the 20-byte payee
    address — so every party (agent, client verifier, e2e helpers) recomputes
    the identical digest from the registry record's fields.
    """
    return hashlib.sha256(
        bytes.fromhex(box_public_key) + bytes.fromhex(wallet_address.lower().removeprefix("0x"))
    ).hexdigest()


@runtime_checkable
class AttestationAgent(Protocol):
    evidence_type: str

    def evidence(self, report_data_hex: str) -> dict: ...


class MockAttestationAgent:
    """Structurally valid evidence, honestly tagged ``mock-cvm-v1``.

    ``debug`` / ``tcb_svn`` knobs exist so tests can exercise every client
    refusal path (debug flag, TCB floor) against otherwise-valid evidence.
    """

    evidence_type = "mock-cvm-v1"

    def __init__(self, *, measurement: str = MOCK_CVM_MEASUREMENT, debug: bool = False, tcb_svn: int = 1) -> None:
        self._measurement = measurement
        self._debug = debug
        self._tcb_svn = tcb_svn

    def evidence(self, report_data_hex: str) -> dict:
        return {
            "type": self.evidence_type,
            "measurement": self._measurement,
            "report_data": report_data_hex,
            "debug": self._debug,
            "tcb": {"svn": self._tcb_svn},
            "quote": base64.b64encode(b"mock-quote:" + bytes.fromhex(report_data_hex)).decode(),
        }


@dataclass(frozen=True)
class BootIdentity:
    """This boot's payload identity: an in-memory keypair plus evidence binding it."""

    cipher: BoxCipher
    evidence: dict


def boot_identity(wallet_address: str, agent: AttestationAgent) -> BootIdentity:
    """Generate the per-boot box keypair and its evidence. Never persisted:
    the key lives in this process's memory only — restart = rotation."""
    cipher = BoxCipher.generate()
    return BootIdentity(cipher=cipher, evidence=agent.evidence(report_data(cipher.public_key, wallet_address)))
