"""One-shot Fresh8 SIM_ONLY physics exam for a frozen teacher-free torque student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_cpu_body_bias_recalibration import WEIGHTS, _summary
from rsi_mjx_clean_touch_body_coord_es import FULL_JOINTS, _validate_body_joints
from rsi_mjx_clean_touch_control_es import COURSES, FRESH8, _capture
from rsi_receiving_contact_dynamics_audit import _run

from rosclaw_soccer.providers.g1.receiving_torque_student import FrozenReceivingTorqueStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def examine(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    warm_start: Path,
    training: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_receiving_contact_dynamics_audit.py")
    trainer = source.with_name("rsi_cpu_receiving_torque_distill.py")
    student_source = (
        source.parents[1] / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py"
    )
    source_hash = hash_bytes(source.read_bytes())
    helper_hash = hash_bytes(helper.read_bytes())
    trainer_hash = hash_bytes(trainer.read_bytes())
    student_source_hash = hash_bytes(student_source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("one new external SIM_ONLY Fresh8 directory required")
    warm: dict[str, Any] = json.loads(warm_start.read_text(encoding="utf-8"))
    warm_hash = warm.pop("report_hash")
    report: dict[str, Any] = json.loads(training.read_text(encoding="utf-8"))
    report_hash = report.pop("report_hash")
    weights = np.asarray(warm["full_weights"], dtype=np.float32)
    model_file = training.parent / "student.npz"
    if (
        warm_hash != hash_json(warm)
        or report_hash != hash_json(report)
        or report["schema"] != "rosclaw_soccer.rsi.cpu_receiving_torque_distill.v1"
        or report["activation_ceiling"] != "SIM_ONLY"
        or report["training_gate_passed"] is not True
        or report["teacher_used_during_student_exam"] is not False
        or report["fresh8_opened"] is not False
        or report["shared_world_qualified"] is not False
        or report["promotion_authorized"] is not False
        or report["warm_start_hash"] != warm_hash
        or report["helper_hash"] != helper_hash
        or report["source_hash"] != trainer_hash
        or report["student_source_hash"] != student_source_hash
        or report["train_courses"] != [list(row) for row in COURSES]
        or report["reserved_fresh8"] != [list(row) for row in FRESH8]
        or warm["fresh8_opened"] is not False
        or warm["promotion_authorized"] is not False
        or weights.shape != (WEIGHTS,)
        or not np.isfinite(weights).all()
        or warm["full_weights_hash"] != hash_bytes(weights.tobytes())
        or set(COURSES) & set(FRESH8)
    ):
        raise ValueError("frozen nine-course successful student and sealed Fresh8 required")
    student = FrozenReceivingTorqueStudent.load_npz(
        model_file, expected_hash=str(report["model_hash"])
    )
    arrays = _capture(captured, fidelity)
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    model_hash = compiled_model_hash(model)
    if model_hash != report["compiled_model_hash"]:
        raise ValueError("Fresh8 compiled physics must match training exactly")
    parent_rows = [
        _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64), FULL_JOINTS, course)
        for course in FRESH8
    ]
    child_rows = [
        _run(
            model,
            arrays,
            weights.astype(np.float64),
            FULL_JOINTS,
            course,
            actor_torque_fn=student.predict,
        )
        for course in FRESH8
    ]
    parent, child = _summary(parent_rows), _summary(child_rows)
    ratios = {
        "mean_speed": child["mean_ball_speed_mps"] / parent["mean_ball_speed_mps"],
        "mean_distance": child["mean_ball_pelvis_distance_m"]
        / parent["mean_ball_pelvis_distance_m"],
    }
    passed = bool(
        child["safe_count"] == 8
        and child["clean_foot_count"] == 8
        and child["nonfoot_count"] == 0
        and ratios["mean_speed"] <= 0.90
        and ratios["mean_distance"] <= 0.90
    )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
        or hash_bytes(trainer.read_bytes()) != trainer_hash
        or hash_bytes(student_source.read_bytes()) != student_source_hash
        or hash_bytes(model_file.read_bytes()) != report["model_hash"]
    ):
        raise RuntimeError("frozen Fresh8 source or student artifact changed during physics")
    outcome: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.cpu_receiving_torque_fresh8.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "trainer_hash": trainer_hash,
        "student_source_hash": student_source_hash,
        "training_report_hash": report_hash,
        "model_hash": report["model_hash"],
        "compiled_model_hash": model_hash,
        "fresh_courses": [list(row) for row in FRESH8],
        "parent": parent,
        "student": child,
        "ratios": ratios,
        "fresh8_gate_passed": passed,
        "teacher_used": False,
        "shared_world_qualified": False,
        "promotion_authorized": False,
    }
    outcome["report_hash"] = hash_json(outcome)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(outcome, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = examine(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("report_hash", "parent", "student", "ratios", "fresh8_gate_passed")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
