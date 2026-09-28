"""SIM_ONLY scale sweep for an unsafe but effective privileged receiving teacher."""

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

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCALES = (0.25, 0.50, 0.75, 1.0)


def probe(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    warm_start: Path,
    prior: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_receiving_contact_dynamics_audit.py")
    source_hash, helper_hash = hash_bytes(source.read_bytes()), hash_bytes(helper.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY teacher-envelope directory required")
    payload: dict[str, Any] = json.loads(warm_start.read_text(encoding="utf-8"))
    commitment = payload.pop("report_hash")
    prior_report: dict[str, Any] = json.loads(prior.read_text(encoding="utf-8"))
    prior_hash = prior_report.pop("report_hash")
    weights = np.asarray(payload["full_weights"], dtype=np.float32)
    if (
        commitment != hash_json(payload)
        or prior_hash != hash_json(prior_report)
        or prior_report["schema"] != "rosclaw_soccer.rsi.cpu_receiving_privileged_teacher_probe.v1"
        or prior_report["warm_start_hash"] != commitment
        or prior_report["helper_hash"] != helper_hash
        or prior_report["selected_sign"] is not None
        or prior_report["results"][1]["sign"] != 1.0
        or prior_report["results"][1]["summary"]["safe_count"] != 0
        or payload["training_gate_passed"] is not False
        or payload["fresh8_opened"] is not False
        or payload["promotion_authorized"] is not False
        or weights.shape != (WEIGHTS,)
        or not np.isfinite(weights).all()
        or payload["full_weights_hash"] != hash_bytes(weights.tobytes())
        or tuple(tuple(row) for row in payload["train_courses"][:8]) != COURSES
        or len(payload["train_courses"]) != 9
        or tuple(tuple(row) for row in payload["reserved_fresh8"]) != FRESH8
    ):
        raise ValueError("sealed unsafe effective teacher and nine-course seed required")
    arrays = _capture(captured, fidelity)
    center = (
        float(arrays["sonic_recorded_qpos"][45, 36]),
        float(arrays["sonic_recorded_qpos"][45, 37]),
        float(arrays["sonic_recorded_qvel"][45, 35]),
    )
    courses = COURSES + (center,)
    if tuple(tuple(row) for row in payload["train_courses"]) != courses:
        raise ValueError("consumed nine-course identity mismatch")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    full = weights.astype(np.float64)
    parent_rows = [
        _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64), FULL_JOINTS, course)
        for course in courses
    ]
    parent = _summary(parent_rows)
    results = []
    selected_scale = None
    for scale in SCALES:
        rows = [
            _run(
                model,
                arrays,
                full,
                FULL_JOINTS,
                course,
                privileged_teacher_lateral_sign=1.0,
                privileged_teacher_torque_scale=scale,
            )
            for course in courses
        ]
        summary = _summary(rows)
        ratios = {
            "mean_speed": summary["mean_ball_speed_mps"] / parent["mean_ball_speed_mps"],
            "mean_distance": summary["mean_ball_pelvis_distance_m"]
            / parent["mean_ball_pelvis_distance_m"],
            "center_speed": rows[-1]["exam_ball_speed_mps"]
            / parent_rows[-1]["exam_ball_speed_mps"],
            "center_distance": rows[-1]["exam_ball_pelvis_distance_m"]
            / parent_rows[-1]["exam_ball_pelvis_distance_m"],
        }
        passed = bool(
            summary["clean_foot_count"] == 9
            and summary["safe_count"] == 9
            and summary["nonfoot_count"] == 0
            and ratios["mean_speed"] <= 0.85
            and ratios["mean_distance"] <= 0.90
            and ratios["center_speed"] <= 0.85
            and ratios["center_distance"] <= 0.90
        )
        if passed and selected_scale is None:
            selected_scale = scale
        results.append(
            {
                "scale": scale,
                "summary": summary,
                "ratios": ratios,
                "center_teacher_active_substeps": rows[-1]["privileged_teacher_active_substeps"],
                "center_teacher_peak_torque_nm": rows[-1]["privileged_teacher_peak_torque_nm"],
                "course_diagnostics": [
                    {
                        "course": list(course),
                        "minimum_pelvis_height_m": row["minimum_pelvis_height_m"],
                        "maximum_tilt_rad": row["maximum_tilt_rad"],
                        "exam_ball_speed_mps": row["exam_ball_speed_mps"],
                        "exam_ball_pelvis_distance_m": row["exam_ball_pelvis_distance_m"],
                    }
                    for course, row in zip(courses, rows, strict=True)
                ],
                "training_gate_passed": passed,
            }
        )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
    ):
        raise RuntimeError("teacher-envelope source changed during CPU physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.cpu_receiving_teacher_envelope.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "warm_start_hash": commitment,
        "prior_report_hash": prior_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "train_courses": [list(row) for row in courses],
        "reserved_fresh8": [list(row) for row in FRESH8],
        "scales": list(SCALES),
        "parent": parent,
        "parent_course_diagnostics": [
            {
                "course": list(course),
                "minimum_pelvis_height_m": row["minimum_pelvis_height_m"],
                "maximum_tilt_rad": row["maximum_tilt_rad"],
                "exam_ball_speed_mps": row["exam_ball_speed_mps"],
                "exam_ball_pelvis_distance_m": row["exam_ball_pelvis_distance_m"],
            }
            for course, row in zip(courses, parent_rows, strict=True)
        ],
        "results": results,
        "selected_scale": selected_scale,
        "teacher_is_privileged": True,
        "teacher_deployable": False,
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
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = probe(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "selected_scale": report["selected_scale"],
                "results": [
                    {
                        "scale": row["scale"],
                        "ratios": row["ratios"],
                        "summary": {
                            key: row["summary"][key]
                            for key in ("clean_foot_count", "nonfoot_count", "safe_count")
                        },
                        "training_gate_passed": row["training_gate_passed"],
                    }
                    for row in report["results"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
