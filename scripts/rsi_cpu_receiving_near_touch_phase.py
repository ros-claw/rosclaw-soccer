"""SIM_ONLY exact CPU near-contact phase learning; Fresh8 stays sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsi_cpu_body_bias_recalibration import BIAS_START, WEIGHTS, _summary
from rsi_mjx_clean_touch_body_coord_es import FULL_JOINTS, _validate_body_joints
from rsi_mjx_clean_touch_control_es import COURSES, FRESH8, _capture
from rsi_receiving_contact_dynamics_audit import _run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

STEPS = (0.12, 0.06, 0.03)


def train(
    *, asset_root: Path, captured: Path, fidelity: Path, warm_start: Path, output_dir: Path
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_receiving_contact_dynamics_audit.py")
    source_hash, helper_hash = hash_bytes(source.read_bytes()), hash_bytes(helper.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY phase-training directory required")
    payload: dict[str, Any] = json.loads(warm_start.read_text(encoding="utf-8"))
    commitment = payload.pop("report_hash")
    weights = np.asarray(payload["full_weights"], dtype=np.float32)
    if (
        commitment != hash_json(payload)
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
        raise ValueError("sealed nine-course trust-region failure required")
    arrays = _capture(captured, fidelity)
    center = (
        float(arrays["sonic_recorded_qpos"][45, 36]),
        float(arrays["sonic_recorded_qpos"][45, 37]),
        float(arrays["sonic_recorded_qvel"][45, 35]),
    )
    if tuple(payload["train_courses"][-1]) != center:
        raise ValueError("consumed center identity mismatch")
    courses = COURSES + (center,)
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    full = weights.astype(np.float64)

    def evaluate(near_bias: NDArray[np.float32]) -> list[dict[str, Any]]:
        return [
            _run(
                model,
                arrays,
                full,
                FULL_JOINTS,
                course,
                near_touch_bias=near_bias.astype(np.float64),
            )
            for course in courses
        ]

    parent_rows = [
        _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64), FULL_JOINTS, course)
        for course in courses
    ]
    parent = _summary(parent_rows)

    def ratios(rows: list[dict[str, Any]]) -> dict[str, float]:
        summary = _summary(rows)
        return {
            "mean_speed": summary["mean_ball_speed_mps"] / parent["mean_ball_speed_mps"],
            "mean_distance": summary["mean_ball_pelvis_distance_m"]
            / parent["mean_ball_pelvis_distance_m"],
            "center_speed": rows[-1]["exam_ball_speed_mps"]
            / parent_rows[-1]["exam_ball_speed_mps"],
            "center_distance": rows[-1]["exam_ball_pelvis_distance_m"]
            / parent_rows[-1]["exam_ball_pelvis_distance_m"],
        }

    def merit(rows: list[dict[str, Any]]) -> float:
        summary = _summary(rows)
        if (
            summary["clean_foot_count"] != 9
            or summary["safe_count"] != 9
            or summary["nonfoot_count"] != 0
        ):
            return -1e6
        ratio = ratios(rows)
        return float(
            -200 * ratio["center_speed"]
            - 25 * ratio["mean_speed"]
            - 5 * ratio["mean_distance"]
            - 80 * max(0.0, ratio["center_distance"] / 0.90 - 1)
            - 80 * max(0.0, ratio["mean_distance"] / 0.90 - 1)
            - 80 * max(0.0, ratio["mean_speed"] / 0.85 - 1)
        )

    current = weights[BIAS_START:].copy()
    start_rows = evaluate(current)
    current_rows = start_rows
    current_merit = merit(current_rows)
    history: list[dict[str, Any]] = []
    trials = 0
    for step in STEPS:
        for coordinate in range(len(FULL_JOINTS)):
            options = []
            for sign in (-1, 1):
                proposal = current.copy()
                proposal[coordinate] = np.float32(
                    np.clip(float(proposal[coordinate]) + sign * step, -2.0, 2.0)
                )
                if proposal[coordinate] == current[coordinate]:
                    continue
                rows = evaluate(proposal)
                trials += 1
                options.append((merit(rows), proposal, rows, sign))
            options.sort(key=lambda row: row[0], reverse=True)
            if options and options[0][0] > current_merit + 1e-9:
                current_merit, current, current_rows, sign = options[0]
            else:
                sign = 0
            history.append(
                {
                    "step": step,
                    "joint_index": coordinate,
                    "accepted_direction": sign,
                    "merit": current_merit,
                    "ratios": ratios(current_rows),
                }
            )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
    ):
        raise RuntimeError("phase-training source changed during exact physics")
    result = ratios(current_rows)
    summary = _summary(current_rows)
    passed = bool(
        summary["clean_foot_count"] == 9
        and summary["safe_count"] == 9
        and summary["nonfoot_count"] == 0
        and result["mean_speed"] <= 0.85
        and result["mean_distance"] <= 0.90
        and result["center_speed"] <= 0.85
        and result["center_distance"] <= 0.90
    )
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.cpu_receiving_near_touch_phase.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "warm_start_hash": commitment,
        "compiled_model_hash": compiled_model_hash(model),
        "train_courses": [list(row) for row in courses],
        "reserved_fresh8": [list(row) for row in FRESH8],
        "steps": list(STEPS),
        "trials": trials,
        "parent": parent,
        "start": _summary(start_rows),
        "best": summary,
        "start_ratios": ratios(start_rows),
        "best_ratios": result,
        "best_courses": current_rows,
        "full_weights": weights.tolist(),
        "full_weights_hash": hash_bytes(weights.tobytes()),
        "near_touch_bias": current.tolist(),
        "near_touch_bias_hash": hash_bytes(current.tobytes()),
        "history_hash": hash_json(history),
        "training_gate_passed": passed,
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("report_hash", "start_ratios", "best_ratios", "training_gate_passed")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
