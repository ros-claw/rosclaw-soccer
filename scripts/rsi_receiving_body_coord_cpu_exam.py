"""SIM_ONLY CPU center gate for training-selected body-coordinated first touch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_mjx_clean_touch_body_coord_es import (
    BODY_WEIGHTS,
    FULL_FEATURES,
    FULL_JOINTS,
    _validate_body_joints,
)
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
        raise ValueError("new external SIM_ONLY CPU center directory required")
    payload: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_hash = payload.pop("report_hash")
    protocol: dict[str, Any] = json.loads(
        (candidate.parent / "protocol.json").read_text(encoding="utf-8")
    )
    protocol_hash = protocol.pop("protocol_hash")
    weights32 = np.asarray(payload["full_weights"], dtype=np.float32)
    expected_weights = len(FULL_JOINTS) * (FULL_FEATURES + 1)
    if (
        candidate_hash != hash_json(payload)
        or protocol_hash != hash_json(protocol)
        or payload["protocol_hash"] != protocol_hash
        or protocol["physical_gpu_index"] != 0
        or payload["best_clean_foot_count"] != 8
        or payload["best_nonfoot_count"] != 0
        or payload["promotion_authorized"] is not False
        or weights32.shape != (expected_weights,)
        or BODY_WEIGHTS != 130
        or not np.isfinite(weights32).all()
        or payload["full_weights_hash"] != hash_bytes(weights32.tobytes())
    ):
        raise ValueError("training-selected GPU0 body candidate required")
    parent_speed = float(payload["parent_mean_half_second_ball_speed_mps"])
    parent_distance = float(payload["parent_mean_half_second_distance_m"])
    best_courses = np.asarray(payload["best_courses"], dtype=np.float64)
    if (
        best_courses.shape != (8, 7)
        or any(row[3] < 0.65 or row[4] >= 0.30 for row in best_courses)
        or payload["best_mean_half_second_ball_speed_mps"] > 0.85 * parent_speed
        or payload["best_mean_half_second_distance_m"] > 0.90 * parent_distance
    ):
        raise ValueError("training hard gate did not pass")
    arrays = _capture(captured, fidelity)
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    parent = _run(model, arrays, np.zeros(expected_weights, dtype=np.float64), FULL_JOINTS)
    child = _run(model, arrays, weights32.astype(np.float64), FULL_JOINTS)
    clean = (
        child["first_contact"] is not None
        and child["first_contact"]["kind"] == "foot"
        and child["first_nonfoot_contact"] is None
        and child["foot_normal_impulse_ns"] > 0
    )
    passed = bool(
        clean
        and child["minimum_pelvis_height_m"] >= 0.65
        and child["maximum_tilt_rad"] < 0.30
        and child["exam_ball_speed_mps"] <= 0.85 * parent["exam_ball_speed_mps"]
        and child["exam_ball_pelvis_distance_m"] <= 0.90 * parent["exam_ball_pelvis_distance_m"]
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("CPU center exam source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_body_coord_cpu_center.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "candidate_hash": candidate_hash,
        "protocol_hash": protocol_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "parent": parent,
        "candidate": child,
        "cpu_center_gate_passed": passed,
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
    result = exam(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("report_hash", "parent", "candidate", "cpu_center_gate_passed")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
