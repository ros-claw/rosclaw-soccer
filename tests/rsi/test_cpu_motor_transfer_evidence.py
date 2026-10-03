import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


@pytest.mark.parametrize("mutation", ["authority", "trace", "source"])
@pytest.mark.parametrize("compressed", [False, True])
def test_cpu_replay_rejects_untrusted_evidence_before_parsing_model(tmp_path, mutation, compressed):
    mujoco = pytest.importorskip("mujoco")
    source = Path(__file__)
    snapshot = tmp_path / "compiled_model.mjb"
    trace = tmp_path / "physical_trace.npz"
    snapshot.write_bytes(b"never parse this untrusted model")
    trace.write_bytes(b"untrusted trace")
    report = dict(
        source_hash=hash_bytes(source.read_bytes()),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        compiled_model_hash=hash_bytes(snapshot.read_bytes()),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        physics=dict(engine="MuJoCo", device="cpu", version=mujoco.__version__),
    )
    if mutation == "authority":
        report["hardware_authorized"] = True
    elif mutation == "trace":
        report["physical_trace_hash"] = hash_bytes(b"other")
    else:
        report["source_hash"] = hash_bytes(b"other")
    report["report_hash"] = hash_json(report)
    write_once(tmp_path / ("report.json.gz" if compressed else "report.json"), report)
    with pytest.raises(ValueError, match="authority contract"):
        audit_cpu_transfer(tmp_path, source)


def test_cpu_replay_rejects_ambiguous_plain_and_gzip_report(tmp_path):
    pytest.importorskip("mujoco")
    report = {"report_hash": hash_json({})}
    (tmp_path / "report.json").write_text(json.dumps(report))
    write_once(tmp_path / "report.json.gz", report)
    with pytest.raises(ValueError, match="exactly one complete physical report"):
        audit_cpu_transfer(tmp_path, Path(__file__))


@pytest.mark.parametrize("factory", [object(), lambda _: None])
def test_cpu_audit_does_not_accept_arbitrary_decoder_callback(tmp_path, monkeypatch, factory):
    pytest.importorskip("mujoco")
    from rosclaw_soccer.rsi import cpu_motor_transfer_evidence as module

    monkeypatch.setattr(module, "_sealed", lambda _: {"executed_motor_policy": {}})
    with pytest.raises(ValueError, match="verified smooth sampling factory"):
        audit_cpu_transfer(tmp_path, Path(__file__), sampling_decoder_factory=factory)
