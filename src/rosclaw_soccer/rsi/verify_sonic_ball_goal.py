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
    expected = {"qpos", "qvel", "target"}
    if report.get("schema") == "rosclaw_soccer.rsi.sonic_ball_contact_probe.v2":
        expected.add("physics_qpos")
    if (
        set(arrays) != expected
        or arrays["qpos"].shape != (300, 43)
        or arrays["qvel"].shape != (300, 41)
        or arrays["target"].shape != (300, 29)
        or "physics_qpos" in arrays
        and arrays["physics_qpos"].shape != (3000, 43)
        or any(not np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError("SONIC ball goal raw tensor layout is invalid")
    return report, arrays


def _first_physics_contact(
    positions: NDArray[np.float64], stadium_assets: Path, expected_physics_hash: str
) -> dict[str, Any] | None:
    import mujoco

    from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
    from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

    model = build_g1_stadium_model(
        stadium_assets, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    if compiled_model_hash(model) != expected_physics_hash:
        raise ValueError("contact audit model differs from physical execution")
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    ball_geoms = {
        index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == ball_body
    }
    data = mujoco.MjData(model)
    for index, q in enumerate(positions):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in ball_geoms and second not in ball_geoms:
                continue
            other = second if first in ball_geoms else first
            body = int(model.geom_bodyid[other])
            ancestor = body
            while ancestor > 0 and ancestor != pelvis_body:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor != pelvis_body:
                continue
            geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            return {
                "time_sec": (index + 1) * 0.002,
                "geom": geom,
                "is_foot": "foot" in geom or "ankle" in geom,
            }
    return None


def verify_sonic_ball_goal(
    primary: Path, replay: Path, *, stadium_assets: Path | None = None
) -> dict[str, Any]:
    if primary.expanduser().resolve() == replay.expanduser().resolve():
        raise ValueError("strict replay requires two distinct execution bundles")
    first, a = _read(primary.expanduser().resolve())
    second, b = _read(replay.expanduser().resolve())
    first_comparable = dict(first)
    second_comparable = dict(second)
    first_execution = first_comparable.pop("execution_id", None)
    second_execution = second_comparable.pop("execution_id", None)
    if first["schema"].endswith(".v2") and (
        not isinstance(first_execution, str)
        or not isinstance(second_execution, str)
        or len(first_execution) != 32
        or len(second_execution) != 32
        or first_execution == second_execution
    ):
        raise ValueError("paired physical executions need distinct runner identities")
    if first_comparable != second_comparable or any(
        not np.array_equal(a[name], b[name]) for name in a
    ):
        raise ValueError("SONIC ball goal is not an exact physical replay")
    if (
        first["schema"]
        not in (
            "rosclaw_soccer.rsi.sonic_ball_contact_probe.v1",
            "rosclaw_soccer.rsi.sonic_ball_contact_probe.v2",
        )
        or first["partition"] not in ("DISCOVERY", "FRESH")
        or first["activation_ceiling"] != "SIM_ONLY"
        or first["promotion_authorized"]
        or first["trained_actor"]
        or first["frames"] != 300
        or first["ball_radius_m"] != 0.11
        or first["ball_mass_kg"] != 0.43
        or first["goal_plane_x_m"] != 5.0
        or first["foot_ball_contact_substeps"] < 1
        or not isinstance(first["first_foot_ball_contact"], dict)
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
    independently_reconstructed = False
    if first["schema"].endswith(".v2"):
        if stadium_assets is None:
            raise ValueError("v2 physical contact audit requires stadium assets")
        if not np.array_equal(a["physics_qpos"][9::10], q):
            raise ValueError("500 Hz and 50 Hz physical trajectories differ")
        contact = _first_physics_contact(a["physics_qpos"], stadium_assets, first["physics_hash"])
        claimed_contact = first["first_robot_ball_contact"]
        if (
            contact is None
            or not contact["is_foot"]
            or first["first_robot_ball_contact_is_foot"] is not True
            or contact["geom"] != claimed_contact["geom"]
            or abs(contact["time_sec"] - claimed_contact["time_sec"]) > 0.006
            or first["first_foot_ball_contact"]["foot_geom"] != claimed_contact["geom"]
            or abs(first["first_foot_ball_contact"]["time_sec"] - claimed_contact["time_sec"])
            > 0.002
        ):
            raise ValueError("foot-first contact claim differs from reconstructed physics")
        independently_reconstructed = True
    return {
        "schema": "rosclaw_soccer.rsi.sonic_ball_goal_verification.v1",
        "strict_replay": True,
        "whole_ball_goal_crossed": goal_frame is not None,
        "goal_frame": goal_frame,
        "minimum_pelvis_height_m": checks["minimum_pelvis_height_m"],
        "foot_ball_contact_reported": True,
        "foot_ball_contact_independently_reconstructed": independently_reconstructed,
        "physical_execution_count": 2,
        "partition": first["partition"],
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
    parser.add_argument("--stadium-assets", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_sonic_ball_goal(args.primary, args.replay, stadium_assets=args.stadium_assets),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
