import json

import numpy as np
import pytest
from rsi_r1_local_fresh_audit_v87 import _read_arm

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def test_local_exam_requires_bound_foot_shot_and_rejects_tampering(tmp_path):
    run = tmp_path / "rsi-r1-strike-fresh-v87-s00-candidate"
    run.mkdir()
    protocol = {
        "partition": "CONSUMED_DEV",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "scenario": {"scenario_id": "s199.rsi.r1.receiver-bridge.consumed"},
    }
    (run / "protocol.json").write_text(json.dumps(protocol))
    trace_path = run / "trace.npz"
    np.savez_compressed(
        trace_path,
        time=np.array([0.2, 0.4, 0.8, 1.0]),
        ball_contact_agent_code=np.array([6, 4, 4, 0]),
        ball_contact_foot_code=np.array([2, 2, 2, 0]),
        ball_velocity=np.array([[1.0, 0, 0], [0.4, 0, 0], [3.0, 0, 0], [2.5, 0, 0]]),
        ball_nonfoot_contact_agent_code=np.zeros(4, dtype=int),
        red_finisher_joint_safety_margin_rad=np.ones((4, 29)) * 0.05,
    )
    report = {
        "protocol_hash": hash_bytes((run / "protocol.json").read_bytes()),
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "chain": {"receiver_contact_sec": 0.4, "clean_transfer_observed": True},
        "crossing": {"frame": 3, "inside_geometry": True},
        "result": {"safe": True},
        "chain_success": True,
    }
    report["report_hash"] = hash_json(report)
    (run / "report.json").write_text(json.dumps(report))
    row = _read_arm(tmp_path, "s00", "candidate")
    assert row["chain_pass"] is True
    assert row["shot_frame"] == 2
    (run / "trace.npz").write_bytes(trace_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="invalid immutable run evidence"):
        _read_arm(tmp_path, "s00", "candidate")
