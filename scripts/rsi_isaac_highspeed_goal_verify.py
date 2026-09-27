"""Independently audit the four frozen SIM_ONLY Isaac high-speed goal courses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.ball_contact_evidence import classify_ball_body_contacts
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.virtual_goal_evidence import first_virtual_goal_crossing

COURSES = ((2.43, 0.08), (2.47, 0.12), (2.53, 0.08), (2.57, 0.12))


def _read_case(root: Path, *, x: float, y: float, speed: float) -> dict[str, Any]:
    report_path = root / "report.json"
    trajectory_path = root / "trajectory.npz"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    stored_hash = report.pop("report_hash")
    if (
        stored_hash != hash_json(report)
        or hash_bytes(trajectory_path.read_bytes()) != report["trajectory_hash"]
    ):
        raise ValueError(f"Isaac evidence hash mismatch: {root}")
    report["report_hash"] = stored_hash
    if (
        report["activation_ceiling"] != "SIM_ONLY"
        or report["ball_x_m"] != x
        or report["ball_y_m"] != y
        or report["forward_command_m_s"] != speed
        or report["frames"] != 300
        or report["agent_count"] != 1
        or not report["track_ball_contacts"]
        or bool(report["experimental_high_speed"]) != (speed > 0.5)
        or report["right_knee_contact_residual_rad"] != 0.0
        or report["right_foot_ik_forward_m"] != 0.0
        or report["lateral_command_m_s"] != 0.0
        or report["reactive_lateral_command_m_s"] != 0.0
    ):
        raise ValueError(f"Isaac course or action drift: {root}")
    with np.load(trajectory_path, allow_pickle=False) as trace:
        ball = np.asarray(trace["ball_position_m"], dtype=np.float64)
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        force = np.asarray(trace["ball_body_contact_force_micro_n"], dtype=np.float64)
    if ball.shape != (300, 3) or qpos.shape != (300, 1, 36) or force.shape != (300, 10, 6):
        raise ValueError(f"Isaac trajectory shape mismatch: {root}")
    if not (np.isfinite(ball).all() and np.isfinite(qpos).all()):
        raise ValueError(f"nonfinite Isaac trajectory: {root}")
    contact = classify_ball_body_contacts(force)
    min_pelvis_m = float(np.min(qpos[:, 0, 2]))
    displacement_m = float(np.linalg.norm(ball[-1, :2] - ball[0, :2]))
    crossing = first_virtual_goal_crossing(ball)
    if (
        report["clean_foot_only_contact_verified"] != contact.clean_foot_only
        or report["first_foot_ball_contact_microstep"] != contact.first_foot_microstep
        or report["first_nonfoot_lower_leg_ball_contact_microstep"]
        != contact.first_nonfoot_microstep
        or not math.isclose(report["min_pelvis_height_m"], min_pelvis_m, abs_tol=1e-5)
        or not math.isclose(report["ball_horizontal_displacement_m"], displacement_m, abs_tol=1e-5)
    ):
        raise ValueError(f"reported Isaac physics differs from trajectory: {root}")
    success = bool(
        min_pelvis_m >= 0.55
        and contact.clean_foot_only
        and displacement_m >= 3.0
        and report["individual"]["blue.playmaker"]["joint_projection_count"] == 0
        and crossing is not None
        and crossing.whole_ball_inside
    )
    return {
        "success": success,
        "clean_foot_only": contact.clean_foot_only,
        "first_foot_microstep": contact.first_foot_microstep,
        "first_nonfoot_microstep": contact.first_nonfoot_microstep,
        "minimum_pelvis_m": min_pelvis_m,
        "ball_displacement_m": displacement_m,
        "crossing": vars(crossing) if crossing is not None else None,
        "trajectory_hash": report["trajectory_hash"],
        "source_hash": report["source_hash"],
        "model_hash": report["model_hash"],
        "asset_hash": report["asset_hash"],
        "report_hash": report["report_hash"],
    }


def verify(root: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(COURSES):
        prefix = f"rsi-highspeed-goal-holdout-{index}"
        candidate = _read_case(root / f"{prefix}-candidate-20260928-v1", x=x, y=y, speed=1.2)
        replay = _read_case(root / f"{prefix}-candidate-replay-20260928-v1", x=x, y=y, speed=1.2)
        parent = _read_case(root / f"{prefix}-parent-20260928-v1", x=x, y=y, speed=0.5)
        if any(
            candidate[key] != replay[key]
            for key in ("trajectory_hash", "source_hash", "model_hash", "asset_hash")
        ):
            raise ValueError(f"candidate physical replay mismatch: course {index}")
        if any(
            candidate[key] != parent[key] for key in ("source_hash", "model_hash", "asset_hash")
        ):
            raise ValueError(f"paired physical identity mismatch: course {index}")
        results.append({"ball_xy_m": [x, y], "candidate": candidate, "parent": parent})
    candidate_successes = sum(row["candidate"]["success"] for row in results)
    parent_successes = sum(row["parent"]["success"] for row in results)
    passed = bool(
        candidate_successes >= 3
        and candidate_successes - parent_successes >= 2
        and all(row["candidate"]["minimum_pelvis_m"] >= 0.55 for row in results)
    )
    result = {
        "schema": "rosclaw_soccer.rsi.isaac_highspeed_goal_holdout.v1",
        "activation_ceiling": "SIM_ONLY",
        "candidate_successes": candidate_successes,
        "parent_successes": parent_successes,
        "local_gate_passed": passed,
        "promotion_authorized": False,
        "results": results,
    }
    result["verification_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence_root), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
