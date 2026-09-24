"""Verify a paired frozen-SONIC football goal from raw MuJoCo trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _read(path: Path) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    report: dict[str, Any] = json.loads((path / "report.json").read_text(encoding="utf-8"))
    claimed = report.pop("report_hash", None)
    if claimed != hash_json(report):
        raise ValueError("SONIC ball goal report hash mismatch")
    trace = path / "trajectory.npz"
    if hash_bytes(trace.read_bytes()) != report.get("trajectory_hash"):
        raise ValueError("SONIC ball goal trajectory hash mismatch")
    with np.load(trace, allow_pickle=False) as stream:
        arrays = {key: stream[key].copy() for key in stream.files}
    if (
        set(arrays) != {"qpos", "qvel", "target"}
        or arrays["qpos"].shape != (300, 43)
        or arrays["qvel"].shape != (300, 41)
        or arrays["target"].shape != (300, 29)
        or any(not np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError("SONIC ball goal raw tensor layout is invalid")
    return report, arrays


def verify_sonic_ball_goal(primary: Path, replay: Path) -> dict[str, Any]:
    first, a = _read(primary.expanduser().resolve())
    second, b = _read(replay.expanduser().resolve())
    if first != second or any(not np.array_equal(a[name], b[name]) for name in a):
        raise ValueError("SONIC ball goal is not an exact physical replay")
    if (
        first["schema"] != "rosclaw_soccer.rsi.sonic_ball_contact_probe.v1"
        or first["partition"] != "DISCOVERY"
        or first["activation_ceiling"] != "SIM_ONLY"
        or first["promotion_authorized"]
        or first["trained_actor"]
        or first["frames"] != 300
        or first["ball_radius_m"] != 0.11
        or first["ball_mass_kg"] != 0.43
        or first["goal_plane_x_m"] != 5.0
        or first["foot_ball_contact_substeps"] < 1
        or first["first_foot_ball_contact"]["foot_geom"] != "left_foot3_collision"
    ):
        raise ValueError("SONIC ball goal is outside the declared discovery contract")
    q = a["qpos"]
    v = a["qvel"]
    radius = float(first["ball_radius_m"])
    whole_ball_across = q[:, 36] - radius >= 5.0
    inside_posts = np.abs(q[:, 37]) + radius <= 1.2
    below_crossbar = q[:, 38] + radius <= 1.6
    frames = np.flatnonzero(whole_ball_across & inside_posts & below_crossbar)
    goal_frame = int(frames[0]) if len(frames) else None
    initial = np.asarray((*first["ball_initial_xy_m"], radius), dtype=np.float64)
    checks = {
        "minimum_pelvis_height_m": float(q[:, 2].min()),
        "peak_ball_speed_mps": float(np.linalg.norm(v[:, 35:38], axis=1).max()),
        "final_ball_displacement_xyz_m": (q[-1, 36:39] - initial).tolist(),
        "goal_frame": goal_frame,
        "goal_crossing_ball_center_xyz_m": (
            q[goal_frame, 36:39].tolist() if goal_frame is not None else None
        ),
    }
    for name in ("minimum_pelvis_height_m", "peak_ball_speed_mps"):
        if not math.isclose(first[name], checks[name], rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"SONIC ball goal {name} differs from raw physics")
    if (
        first["whole_ball_goal_crossed"] is not (goal_frame is not None)
        or first["goal_frame"] != goal_frame
        or not np.allclose(
            first["final_ball_displacement_xyz_m"],
            checks["final_ball_displacement_xyz_m"],
            rtol=0.0,
            atol=1.0e-12,
        )
        or first["goal_crossing_ball_center_xyz_m"] != checks["goal_crossing_ball_center_xyz_m"]
        or checks["minimum_pelvis_height_m"] < 0.55
    ):
        raise ValueError("SONIC ball goal claim differs from raw physical outcome")
    return {
        "schema": "rosclaw_soccer.rsi.sonic_ball_goal_verification.v1",
        "strict_replay": True,
        "whole_ball_goal_crossed": goal_frame is not None,
        "goal_frame": goal_frame,
        "minimum_pelvis_height_m": checks["minimum_pelvis_height_m"],
        "foot_ball_contact_reported": True,
        "foot_ball_contact_independently_reconstructed": False,
        "physical_execution_count": 2,
        "partition": "DISCOVERY",
        "promotion_authorized": False,
        "verification_hash": hash_json(
            {
                "report_hash": hash_bytes((primary / "report.json").read_bytes()),
                "trajectory_hash": first["trajectory_hash"],
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_sonic_ball_goal(args.primary, args.replay), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
