"""Independently check the paired FRESH right-leg residual experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.verify_sonic_ball_goal import _first_physics_contact, _read
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

COURSES = ((0.055, 0.0475), (0.095, 0.09166666666666667), (0.135, 0.1), (0.175, 0.1))
SELECTOR = "sha256:23126ae9179e8493d54d07410e5e32be72e4e75581cec7f8c97a52a44d80901e"


def verify(root: Path, *, stadium_assets: Path, probe_source: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    source_hash = hash_bytes(probe_source.read_bytes())
    rows: list[dict[str, Any]] = []
    hashes = []
    for y, lateral in COURSES:
        pair = []
        for candidate in (False, True):
            mode = "candidate" if candidate else "parent"
            path = root / f"rsi-sonic-right-swing-fresh-x218-y{round(y * 1000)}-{mode}-20260924"
            report, arrays = _read(path)
            q, v = arrays["qpos"], arrays["qvel"]
            contact = _first_physics_contact(
                arrays["physics_qpos"], stadium_assets, report["physics_hash"]
            )
            goal = np.flatnonzero(
                (q[:, 36] - report["ball_radius_m"] >= 5.0)
                & (np.abs(q[:, 37]) + report["ball_radius_m"] <= 1.2)
                & (q[:, 38] + report["ball_radius_m"] <= 1.6)
            )
            peak_speed = float(np.linalg.norm(v[:, 35:38], axis=1).max())
            pelvis_min = float(q[:, 2].min())
            physics_quat = arrays["physics_qpos"][:, 3:7]
            peak_tilt = float(
                np.arccos(
                    np.clip(
                        1.0 - 2.0 * (physics_quat[:, 1] ** 2 + physics_quat[:, 2] ** 2),
                        -1.0,
                        1.0,
                    )
                ).max()
            )
            expected_goal_frame = int(goal[0]) if len(goal) else None
            claimed_contact = report["first_robot_ball_contact"]
            if (
                report["partition"] != "FRESH"
                or report["activation_ceiling"] != "SIM_ONLY"
                or report["promotion_authorized"] is not False
                or report["source_hash"] != source_hash
                or report["selector_hash"] != SELECTOR
                or report["ball_initial_xy_m"] != [2.18, y]
                or report["run_lateral_mps"] != lateral
                or report["run_speed_mps"] != 1.4
                or report["stop_frame"] != 120
                or report["left_contact_residual_rad"] != [-0.15, 0.15, 0.0]
                or report["right_contact_residual_rad"]
                != ([-0.25, -0.25, 0.0] if candidate else [0.0, 0.0, 0.0])
                or not np.array_equal(arrays["physics_qpos"][9::10], q)
                or contact is None
                or claimed_contact is None
                or contact["geom"] != claimed_contact["geom"]
                or abs(contact["time_sec"] - claimed_contact["time_sec"]) > 0.006
                or contact["is_foot"] is not report["first_robot_ball_contact_is_foot"]
                or expected_goal_frame != report["goal_frame"]
                or (expected_goal_frame is not None) is not report["whole_ball_goal_crossed"]
                or not math.isclose(peak_speed, report["peak_ball_speed_mps"], abs_tol=1e-12)
                or not math.isclose(pelvis_min, report["minimum_pelvis_height_m"], abs_tol=1e-12)
                or not math.isclose(peak_tilt, report["peak_pelvis_tilt_rad"], abs_tol=1e-12)
                or pelvis_min < 0.68
                or peak_tilt > 0.35
                or report["actuator_saturation_fraction"] != 0.0
                or report["peak_torque_demand_ratio"] >= 1.0
                or not contact["is_foot"]
                or expected_goal_frame is None
            ):
                raise ValueError(f"FRESH swing physical/safety contract failed: {path}")
            row = {
                "ball_y_m": y,
                "candidate": candidate,
                "first_contact_geom": contact["geom"],
                "goal_frame": expected_goal_frame,
                "peak_ball_speed_mps": peak_speed,
                "minimum_pelvis_height_m": pelvis_min,
                "peak_pelvis_tilt_rad": peak_tilt,
                "reported_peak_torque_demand_ratio": report["peak_torque_demand_ratio"],
            }
            rows.append(row)
            pair.append(report)
            hashes.append(hash_json(report))
        if pair[0]["initial_state_hash"] != pair[1]["initial_state_hash"]:
            raise ValueError("paired swing exam initial physical states differ")
        if pair[1]["peak_ball_speed_mps"] <= pair[0]["peak_ball_speed_mps"]:
            raise ValueError("candidate failed to improve physical ball speed")
    parent_speeds = [row["peak_ball_speed_mps"] for row in rows if not row["candidate"]]
    candidate_speeds = [row["peak_ball_speed_mps"] for row in rows if row["candidate"]]
    result: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.sonic_right_swing_holdout_verification.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "partition": "FRESH",
        "selector_hash": SELECTOR,
        "source_hash": source_hash,
        "physical_execution_count": len(rows),
        "physical_contact_independently_reconstructed": True,
        "parent_foot_first_goals": 4,
        "candidate_foot_first_goals": 4,
        "parent_mean_peak_ball_speed_mps": sum(parent_speeds) / 4,
        "candidate_mean_peak_ball_speed_mps": sum(candidate_speeds) / 4,
        "report_hashes": sorted(hashes),
        "results": rows,
        "torque_telemetry_independently_reconstructed": False,
    }
    result["verification_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--probe-source", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(**vars(args)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
