"""SIM_ONLY CPU center exam of a training-selected exact-physics bias actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_cpu_body_bias_recalibration import WEIGHTS, _summary
from rsi_mjx_clean_touch_body_coord_es import FULL_JOINTS, _validate_body_joints
from rsi_mjx_clean_touch_control_es import _capture
from rsi_receiving_contact_dynamics_audit import _run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def exam(
    *, asset_root: Path, captured: Path, fidelity: Path, candidate: Path, output_dir: Path
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY center exam directory required")
    payload: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    commitment = payload.pop("report_hash")
    protocol: dict[str, Any] = json.loads(
        (candidate.parent / "protocol.json").read_text(encoding="utf-8")
    )
    protocol_hash = protocol.pop("protocol_hash")
    weights = np.asarray(payload["full_weights"], dtype=np.float32)
    if (
        commitment != hash_json(payload)
        or protocol_hash != hash_json(protocol)
        or payload["protocol_hash"] != protocol_hash
        or payload["promotion_authorized"] is not False
        or payload["fresh8_opened"] is not False
        or weights.shape != (WEIGHTS,)
        or not np.isfinite(weights).all()
        or payload["full_weights_hash"] != hash_bytes(weights.tobytes())
        or payload["best"]["clean_foot_count"] != 8
        or payload["best"]["safe_count"] != 8
        or payload["best"]["nonfoot_count"] != 0
        or payload["best"]["mean_ball_speed_mps"] > 0.85 * payload["parent"]["mean_ball_speed_mps"]
        or payload["best"]["mean_ball_pelvis_distance_m"]
        > 0.90 * payload["parent"]["mean_ball_pelvis_distance_m"]
    ):
        raise ValueError("exact CPU training hard gate required")
    arrays = _capture(captured, fidelity)
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    parent = _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64), FULL_JOINTS)
    child = _run(model, arrays, weights.astype(np.float64), FULL_JOINTS)
    summary = _summary([child])
    passed = bool(
        summary["clean_foot_count"] == 1
        and summary["safe_count"] == 1
        and summary["nonfoot_count"] == 0
        and child["exam_ball_speed_mps"] <= 0.85 * parent["exam_ball_speed_mps"]
        and child["exam_ball_pelvis_distance_m"] <= 0.90 * parent["exam_ball_pelvis_distance_m"]
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("center exam source changed during CPU physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.cpu_body_bias_center_exam.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "candidate_hash": commitment,
        "protocol_hash": protocol_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "parent": parent,
        "candidate": child,
        "center_gate_passed": passed,
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = exam(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("report_hash", "parent", "candidate", "center_gate_passed")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
