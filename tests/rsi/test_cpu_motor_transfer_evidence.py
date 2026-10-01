import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@pytest.mark.parametrize("mutation", ["authority", "trace", "source"])
def test_cpu_replay_rejects_untrusted_evidence_before_parsing_model(tmp_path, mutation):
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
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="authority contract"):
        audit_cpu_transfer(tmp_path, source)
