"""The attestation-agent seam: mock evidence with the real report-data binding."""

import hashlib

from vorqd.tee.agent import (
    MOCK_CVM_MEASUREMENT,
    BootIdentity,
    MockAttestationAgent,
    boot_identity,
    report_data,
)

WALLET = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"


def test_report_data_binds_key_and_payee():
    box_pub = "ab" * 32
    expected = hashlib.sha256(bytes.fromhex(box_pub) + bytes.fromhex(WALLET[2:].lower())).hexdigest()
    assert report_data(box_pub, WALLET) == expected


def test_mock_evidence_is_structurally_valid_and_honestly_tagged():
    ev = MockAttestationAgent().evidence("cd" * 32)
    assert ev["type"] == "mock-cvm-v1"          # distinct tag: never masquerades as real
    assert ev["measurement"] == MOCK_CVM_MEASUREMENT
    assert ev["report_data"] == "cd" * 32
    assert ev["debug"] is False
    assert ev["tcb"] == {"svn": 1}
    assert isinstance(ev["quote"], str) and ev["quote"]


def test_boot_identity_generates_fresh_key_with_bound_evidence():
    a = boot_identity(WALLET, MockAttestationAgent())
    b = boot_identity(WALLET, MockAttestationAgent())
    assert isinstance(a, BootIdentity)
    assert a.cipher.public_key != b.cipher.public_key          # restart = rotation, by construction
    assert a.evidence["report_data"] == report_data(a.cipher.public_key, WALLET)
