import json
from pathlib import Path

import pytest
from rsi_r1_bridge_audit_v72 import audit

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _frozen_motor_hash(chosen):
    action = {
        "entry_frame": 30,
        "forward_cap_m": 0.16,
        "lateral_cap_m": 0.10,
        "vertical_offset_m": 0.04,
        "swing_foot_acquisition_gap_m": 0.55,
        "swing_acquisition_max_lateral_gap_m": 0.22,
        "revalidate_swing_side": True,
        "joint_boundary_recovery_cap_rad": 0.04,
        "rolling_cap_rad": chosen["rolling_cap_rad"],
        "rolling_radius_m": chosen["rolling_radius_m"],
        "rolling_vertical_m": chosen["rolling_vertical_m"],
        "rolling_requires_commitment": chosen["rolling_requires_commitment"],
    }
    return hash_json(
        {
            "schema": "rsi_team_swing_motor_v13",
            "agent": "red.finisher",
            "enabled": True,
            "action": action,
            "activation_selector_hash": None,
            "activation_ceiling": "SIM_ONLY",
        }
    )


def test_frozen_audit_checks_all_pairs_and_rejects_tampered_trace(tmp_path):
    source = (
        Path(__file__).resolve().parents[1] / "docs/rsi/protocols/r1-receiver-bridge-fresh-v72.json"
    )
    protocol = json.loads(source.read_text())
    protocol_path = tmp_path / "frozen.json"
    protocol_path.write_text(json.dumps(protocol))
    for index, scene in enumerate(protocol["scenes"]):
        for arm in ("control", "candidate"):
            directory = tmp_path / f"rsi-r1-bridge-fresh-v72-{scene['id']}-{arm}"
            directory.mkdir()
            candidate = arm == "candidate"
            run_protocol = {
                "scenario": {
                    "seed": scene["seed"],
                    "ball_initial_position_m": [scene["ball_x_m"], scene["ball_y_m"], 0.115],
                },
                "activation_ceiling": "SIM_ONLY",
                "promotion_authorized": False,
                "source_hashes": {"code": "sha256:" + "a" * 64},
                "enabled": candidate,
                "motor_present": candidate,
                "grounded": candidate,
                "retired_ankle_braking": 12.0 if candidate else None,
                "motor_contract_hash": _frozen_motor_hash(protocol["chosen_from_consumed"]),
            }
            run_path = directory / "protocol.json"
            run_path.write_text(json.dumps(run_protocol))
            trace_path = directory / "trace.npz"
            trace_path.write_bytes(b"synthetic-test-trace")
            report = {
                "protocol_hash": hash_bytes(run_path.read_bytes()),
                "trace_hash": hash_bytes(trace_path.read_bytes()),
                "result": {"safe": True},
                "chain": {"clean_transfer_observed": candidate and index < 6, "reasons": []},
                "chain_success": candidate and index < 2,
                "motor_active_frames": 7 if candidate else 0,
            }
            report["report_hash"] = hash_json(report)
            (directory / "report.json").write_text(json.dumps(report))
    result = audit(protocol_path, tmp_path)
    assert result["frozen_gate_passed"]
    assert result["counts"]["candidate"] == {
        "safe": 8,
        "clean_transfer": 6,
        "chain_success": 2,
    }
    corrupted = tmp_path / "rsi-r1-bridge-fresh-v72-s00-candidate/trace.npz"
    corrupted.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="broken evidence integrity"):
        audit(protocol_path, tmp_path)
