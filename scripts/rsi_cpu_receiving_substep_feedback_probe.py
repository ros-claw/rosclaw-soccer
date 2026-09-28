"""SIM_ONLY paired 50 Hz versus 500 Hz receiving-feedback mechanism probe."""

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


def probe(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    trust: Path,
    near: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_receiving_contact_dynamics_audit.py")
    source_hash, helper_hash = hash_bytes(source.read_bytes()), hash_bytes(helper.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY probe directory required")
    trust_payload: dict[str, Any] = json.loads(trust.read_text(encoding="utf-8"))
    near_payload: dict[str, Any] = json.loads(near.read_text(encoding="utf-8"))
    trust_hash = trust_payload.pop("report_hash")
    near_hash = near_payload.pop("report_hash")
    weights = np.asarray(trust_payload["full_weights"], dtype=np.float32)
    near_bias = np.asarray(near_payload["near_touch_bias"], dtype=np.float32)
    if (
        trust_hash != hash_json(trust_payload)
        or near_hash != hash_json(near_payload)
        or trust_payload["training_gate_passed"] is not False
        or near_payload["training_gate_passed"] is not False
        or trust_payload["fresh8_opened"] is not False
        or near_payload["fresh8_opened"] is not False
        or trust_payload["promotion_authorized"] is not False
        or near_payload["promotion_authorized"] is not False
        or near_payload["warm_start_hash"] != trust_hash
        or near_payload["full_weights_hash"] != trust_payload["full_weights_hash"]
        or weights.shape != (WEIGHTS,)
        or near_bias.shape != (len(FULL_JOINTS),)
        or not np.isfinite(weights).all()
        or not np.isfinite(near_bias).all()
        or trust_payload["full_weights_hash"] != hash_bytes(weights.tobytes())
        or near_payload["near_touch_bias_hash"] != hash_bytes(near_bias.tobytes())
        or tuple(tuple(row) for row in trust_payload["train_courses"][:8]) != COURSES
        or tuple(tuple(row) for row in near_payload["reserved_fresh8"]) != FRESH8
    ):
        raise ValueError("sealed unpromoted receiving candidates required")
    arrays = _capture(captured, fidelity)
    center = (
        float(arrays["sonic_recorded_qpos"][45, 36]),
        float(arrays["sonic_recorded_qpos"][45, 37]),
        float(arrays["sonic_recorded_qvel"][45, 35]),
    )
    courses = COURSES + (center,)
    if (
        tuple(tuple(row) for row in trust_payload["train_courses"]) != courses
        or tuple(tuple(row) for row in near_payload["train_courses"]) != courses
    ):
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
    modes: dict[str, list[dict[str, Any]]] = {
        "fifty_hz": [_run(model, arrays, full, FULL_JOINTS, course) for course in courses],
        "five_hundred_hz": [
            _run(model, arrays, full, FULL_JOINTS, course, substep_feedback=True)
            for course in courses
        ],
        "five_hundred_hz_near": [
            _run(
                model,
                arrays,
                full,
                FULL_JOINTS,
                course,
                near_touch_bias=near_bias.astype(np.float64),
                substep_feedback=True,
            )
            for course in courses
        ],
    }
    parent = _summary(parent_rows)
    results = {}
    for name, rows in modes.items():
        summary = _summary(rows)
        results[name] = {
            "summary": summary,
            "center": rows[-1],
            "mean_speed_ratio": summary["mean_ball_speed_mps"] / parent["mean_ball_speed_mps"],
            "mean_distance_ratio": summary["mean_ball_pelvis_distance_m"]
            / parent["mean_ball_pelvis_distance_m"],
            "center_speed_ratio": rows[-1]["exam_ball_speed_mps"]
            / parent_rows[-1]["exam_ball_speed_mps"],
            "center_distance_ratio": rows[-1]["exam_ball_pelvis_distance_m"]
            / parent_rows[-1]["exam_ball_pelvis_distance_m"],
            "gate_passed": bool(
                summary["clean_foot_count"] == 9
                and summary["safe_count"] == 9
                and summary["nonfoot_count"] == 0
                and summary["mean_ball_speed_mps"] <= 0.85 * parent["mean_ball_speed_mps"]
                and summary["mean_ball_pelvis_distance_m"]
                <= 0.90 * parent["mean_ball_pelvis_distance_m"]
                and rows[-1]["exam_ball_speed_mps"] <= 0.85 * parent_rows[-1]["exam_ball_speed_mps"]
                and rows[-1]["exam_ball_pelvis_distance_m"]
                <= 0.90 * parent_rows[-1]["exam_ball_pelvis_distance_m"]
            ),
        }
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
    ):
        raise RuntimeError("probe source changed during exact physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.cpu_receiving_substep_feedback_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "trust_hash": trust_hash,
        "near_hash": near_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "train_courses": [list(row) for row in courses],
        "reserved_fresh8": [list(row) for row in FRESH8],
        "parent": parent,
        "results": results,
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
    parser.add_argument("--trust", required=True, type=Path)
    parser.add_argument("--near", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = probe(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "results": {
                    key: {
                        metric: row[metric]
                        for metric in (
                            "mean_speed_ratio",
                            "mean_distance_ratio",
                            "center_speed_ratio",
                            "center_distance_ratio",
                            "gate_passed",
                        )
                    }
                    for key, row in report["results"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
