"""Mock allocation routing, not policy execution or physical qualification."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi import (
    fixed_physical_payload_reader,
    foundation_torch_capture_audit,
    json_artifact_io,
    recurrent_sampling_episode_factory,
    sampling_reference_audit,
)
from scripts import rsi_mesh_contact_native_pilot as historical
from scripts.rsi_native_first_touch_bundle import allocate_native_first_touch_bundle


def setup(tmp_path, monkeypatch, *, historical_error=False, reader_error=False):
    original = tmp_path / "original.json.gz"
    original.write_bytes(b"mock payload, not a numerical model")
    root = tmp_path / "new-experiment"
    root.mkdir()
    reference = dict(witnesses=dict(proof_payload=dict(path=str(original), sha256="mock-file-pin")))
    phases, calls = [], []

    def old(receipt):
        assert receipt == reference
        calls.append("historical")
        if historical_error:
            raise ValueError("historical source refused")
        return dict(executed_motor_policy=dict(synthetic=True))

    monkeypatch.setattr(historical, "load_historical_report", old)

    def compiler(policy):
        assert policy == dict(synthetic=True)
        calls.append("compiler")
        return "compiler"

    monkeypatch.setattr(sampling_reference_audit, "SamplingReferenceAuditCompiler", compiler)
    monkeypatch.setattr(json_artifact_io, "load_json_artifact", lambda path: dict(mock=True))
    monkeypatch.setattr(
        recurrent_sampling_episode_factory,
        "RecurrentSamplingEpisodeFactory",
        lambda model: "factory",
    )

    def reader(path, *, expected_file_hash):
        assert path == root / ".shared-models" / original.name
        assert path.read_bytes() == original.read_bytes()
        assert expected_file_hash == "mock-file-pin"
        if reader_error:
            raise ValueError("new copied payload refused")
        return "reader"

    monkeypatch.setattr(fixed_physical_payload_reader, "FixedPhysicalPayloadReader", reader)
    monkeypatch.setattr(foundation_torch_capture_audit, "TorchCallAuditor", lambda path: "auditor")

    def session(path, options, *, providers):
        assert providers == ["CPUExecutionProvider"]
        assert options.intra_op_num_threads == options.inter_op_num_threads == 1
        return "session"

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda value: None))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=session),
    )
    assets = SimpleNamespace(EXPORT=Path("mock-export"), SONIC=Path("mock-sonic"))
    return assets, reference, root, phases, calls, original


def test_bundle_owns_new_payload_but_loads_old_policy_from_original_owner(tmp_path, monkeypatch):
    assets, reference, root, phases, calls, original = setup(tmp_path, monkeypatch)
    bundle = allocate_native_first_touch_bundle(assets, reference, root, record_phase=phases.append)
    assert calls == ["historical", "compiler"]
    assert bundle.factory == "factory" and bundle.reader == "reader"
    assert bundle.compiler == "compiler" and bundle.auditor == "auditor"
    assert bundle.encoder_session == "session"
    assert phases[0] == "VERIFYING_OWNED_HISTORICAL_REPORT"
    assert phases[-1] == "FIXED_NATIVE_REFERENCE_BUNDLE_READY"
    assert original.read_bytes() == b"mock payload, not a numerical model"


def test_bad_historical_source_fails_before_factory_or_new_store(tmp_path, monkeypatch):
    assets, reference, root, phases, calls, _ = setup(tmp_path, monkeypatch, historical_error=True)
    with pytest.raises(ValueError, match="historical source"):
        allocate_native_first_touch_bundle(assets, reference, root, record_phase=phases.append)
    assert calls == ["historical"]
    assert not (root / ".shared-models").exists()


def test_new_payload_failure_is_propagated_with_evidence_preserved(tmp_path, monkeypatch):
    assets, reference, root, phases, _, original = setup(tmp_path, monkeypatch, reader_error=True)
    with pytest.raises(ValueError, match="copied payload"):
        allocate_native_first_touch_bundle(assets, reference, root, record_phase=phases.append)
    assert (root / ".shared-models" / original.name).read_bytes() == original.read_bytes()
    assert "FIXED_NATIVE_REFERENCE_BUNDLE_READY" not in phases
