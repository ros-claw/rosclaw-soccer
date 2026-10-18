"""Ownership routing only; payload reader and report load are explicit mocks."""

from pathlib import Path

import pytest

from rosclaw_soccer.rsi import failure_curriculum_evidence, fixed_physical_payload_reader
from scripts.rsi_mesh_contact_native_pilot import load_historical_report


def receipt(
    report="/old/attempt-0000/report.json.gz", payload="/old/.shared-models/payload.json.gz"
):
    return dict(
        witnesses=dict(
            report=dict(path=report), proof_payload=dict(path=payload, sha256="sha256:" + "a" * 64)
        ),
        physical_report_hash="physical",
    )


def test_historical_report_uses_historical_payload_not_new_copy(monkeypatch):
    calls = []
    sentinel = object()

    def reader(path, **kwargs):
        calls.append(path)
        return sentinel

    def sealed(path, *, physical_payload_reader):
        assert path == Path("/old/attempt-0000/report.json.gz")
        assert physical_payload_reader is sentinel
        return dict(report_hash="physical", executed_motor_policy={})

    monkeypatch.setattr(fixed_physical_payload_reader, "FixedPhysicalPayloadReader", reader)
    monkeypatch.setattr(failure_curriculum_evidence, "_sealed", sealed)
    assert load_historical_report(receipt())["report_hash"] == "physical"
    assert calls == [Path("/old/.shared-models/payload.json.gz")]


def test_new_copy_cannot_stand_in_for_historical_owner(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid owner must fail before payload allocation")

    monkeypatch.setattr(fixed_physical_payload_reader, "FixedPhysicalPayloadReader", forbidden)
    with pytest.raises(ValueError, match="same owner"):
        load_historical_report(receipt(payload="/new/.shared-models/payload.json.gz"))


def test_original_physical_report_identity_is_still_enforced(monkeypatch):
    monkeypatch.setattr(
        fixed_physical_payload_reader, "FixedPhysicalPayloadReader", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        failure_curriculum_evidence, "_sealed", lambda *a, **kw: dict(report_hash="wrong")
    )
    with pytest.raises(ValueError, match="physical evidence"):
        load_historical_report(receipt())
