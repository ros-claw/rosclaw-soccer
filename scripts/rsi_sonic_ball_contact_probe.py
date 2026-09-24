"""SIM_ONLY frozen SONIC run-through against one real MuJoCo football.

Discovery-only contact diagnostic. It records every physics-step ball/foot
contact and ball velocity; no kick, goal, or learning success is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation
from rosclaw_soccer.skills.team.navigation_envelope import SimulationNavigationEnvelope
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def _observation(
    data: mujoco.MjData,
    frame: int,
    envelope: SimulationNavigationEnvelope | None,
    run_speed_mps: float,
    run_lateral_mps: float,
    stop_frame: int,
) -> TeamMotorObservation:
    forward_mps = math.sqrt(run_speed_mps**2 - run_lateral_mps**2)
    return TeamMotorObservation(
        agent_id="blue.playmaker",
        frame=frame,
        time_sec=frame * 0.02,
        intent="other",
        prospective_owner=False,
        qpos=tuple(float(value) for value in data.qpos),
        qvel=tuple(float(value) for value in data.qvel),
        target_position_m=(0.0, 0.0, 0.0),
        navigation_command=(
            forward_mps if frame < stop_frame else 0.0,
            run_lateral_mps if frame < stop_frame else 0.0,
            0.0,
        ),
        navigation_envelope=envelope,
    )


def run(
    *,
    model_root: Path,
    stadium_assets: Path,
    output_dir: Path,
    ball_x_m: float,
    ball_y_m: float,
    frames: int,
    run_speed_mps: float = 1.2,
    run_lateral_mps: float = 0.0,
    stop_frame: int = 115,
) -> dict[str, Any]:
    if (
        output_dir.exists()
        or not math.isfinite(ball_x_m)
        or not math.isfinite(ball_y_m)
        or not 1.0 <= ball_x_m <= 2.2
        or abs(ball_y_m) > 0.25
        or not 180 <= frames <= 350
        or not math.isfinite(run_speed_mps)
        or not 0.7 < run_speed_mps <= 1.5
        or not math.isfinite(run_lateral_mps)
        or abs(run_lateral_mps) > 0.30
        or abs(run_lateral_mps) >= run_speed_mps
        or not 70 <= stop_frame <= min(170, frames - 30)
    ):
        raise ValueError("new output and bounded ball position required")
    source_hash = hash_bytes(Path(__file__).read_bytes())
    model = build_g1_stadium_model(
        stadium_assets, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise ValueError("expected one 29-DoF G1 and one free football")
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_DDS_JOINT_NAMES]
    )
    if (
        np.any(joint_ids < 0)
        or not np.array_equal(model.jnt_qposadr[joint_ids], np.arange(7, 36))
        or not np.array_equal(model.actuator_trnid[:, 0], joint_ids)
    ):
        raise ValueError("SONIC-to-physics joint order changed")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    if ball_geom < 0:
        raise ValueError("compiled football geom missing")
    ball_dimensions = inspect_native_ball_dimensions(model, geom_id=ball_geom)
    if not ball_dimensions.size_and_mass_in_ifab_range:
        raise ValueError("compiled football dimensions outside adult regulation envelope")
    ball_geoms = {
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == ball_dimensions.body_id
    }
    data = mujoco.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0)
    data.qpos[7:36] = G1SonicRunupController.default_angles
    data.qpos[36:43] = (ball_x_m, ball_y_m, ball_dimensions.radius_m, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    initial_hash = hash_json({"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()})
    nav = G1SonicNavigation(
        model_root,
        "blue.playmaker",
        SonicNavigationConfig(
            maximum_frames=frames,
            model_variant="low_latency",
            experimental_maximum_speed_mps=run_speed_mps,
            planner_seed=920101,
        ),
    )
    nav.backend.qualification.require_eligible()
    torque_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    qpos = []
    qvel = []
    target_rows = []
    foot_contacts: list[dict[str, Any]] = []
    initial_ball_position = data.qpos[36:39].copy()
    for frame in range(frames):
        observation = _observation(
            data, frame, nav.navigation_envelope, run_speed_mps, run_lateral_mps, stop_frame
        )
        if frame == 0:
            nav.start_from_observation(observation)
        action = nav.propose(observation)
        target = np.asarray(action.target_rad, dtype=np.float64)
        for substep in range(10):
            torque = np.clip(
                (target - data.qpos[7:36]) * nav.backend.kp - data.qvel[6:35] * nav.backend.kd,
                -torque_limits,
                torque_limits,
            )
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if not (set((int(contact.geom1), int(contact.geom2))) & ball_geoms):
                    continue
                other = contact.geom2 if int(contact.geom1) in ball_geoms else contact.geom1
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or ""
                if "foot" in name or "ankle" in name:
                    foot_contacts.append(
                        {
                            "time_sec": frame * 0.02 + (substep + 1) * 0.002,
                            "foot_geom": name,
                            "ball_position_m": data.qpos[36:39].tolist(),
                            "ball_velocity_mps": data.qvel[35:38].tolist(),
                        }
                    )
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        target_rows.append(target.copy())
    arrays = {"qpos": np.asarray(qpos), "qvel": np.asarray(qvel), "target": np.asarray(target_rows)}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite ball-contact physics")
    output_dir.mkdir(parents=True)
    path = output_dir / "trajectory.npz"
    np.savez_compressed(path, **arrays)
    ball_speed = np.linalg.norm(arrays["qvel"][:, 35:38], axis=1)
    ball_delta = arrays["qpos"][:, 36:39] - initial_ball_position
    goal_spec = G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    whole_ball_across = arrays["qpos"][:, 36] - ball_dimensions.radius_m >= goal_spec.plane_x_m
    inside_posts = (
        np.abs(arrays["qpos"][:, 37]) + ball_dimensions.radius_m <= goal_spec.width_m / 2.0
    )
    below_crossbar = arrays["qpos"][:, 38] + ball_dimensions.radius_m <= goal_spec.height_m
    goal_frames = np.flatnonzero(whole_ball_across & inside_posts & below_crossbar)
    goal_frame = int(goal_frames[0]) if len(goal_frames) else None
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.sonic_ball_contact_probe.v1",
        "partition": "DISCOVERY",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "physics_hash": compiled_model_hash(model),
        "model_hash": nav.backend.qualification.qualification_hash,
        "initial_state_hash": initial_hash,
        "ball_radius_m": ball_dimensions.radius_m,
        "ball_mass_kg": ball_dimensions.body_mass_kg,
        "ball_initial_xy_m": [ball_x_m, ball_y_m],
        "frames": frames,
        "run_speed_mps": run_speed_mps,
        "run_lateral_mps": run_lateral_mps,
        "stop_frame": stop_frame,
        "goal_plane_x_m": goal_spec.plane_x_m,
        "whole_ball_goal_crossed": goal_frame is not None,
        "goal_frame": goal_frame,
        "goal_crossing_ball_center_xyz_m": (
            arrays["qpos"][goal_frame, 36:39].tolist() if goal_frame is not None else None
        ),
        "foot_ball_contact_substeps": len(foot_contacts),
        "first_foot_ball_contact": foot_contacts[0] if foot_contacts else None,
        "peak_ball_speed_mps": float(ball_speed.max()),
        "final_ball_displacement_xyz_m": ball_delta[-1].tolist(),
        "minimum_pelvis_height_m": float(arrays["qpos"][:, 2].min()),
        "trajectory_hash": hash_bytes(path.read_bytes()),
        "trained_actor": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if hash_bytes(Path(__file__).read_bytes()) != source_hash:
        raise RuntimeError("ball probe source changed during execution")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ball-x-m", type=float, required=True)
    parser.add_argument("--ball-y-m", type=float, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--run-speed-mps", type=float, default=1.2)
    parser.add_argument("--run-lateral-mps", type=float, default=0.0)
    parser.add_argument("--stop-frame", type=int, default=115)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                model_root=args.model_root,
                stadium_assets=args.stadium_assets,
                output_dir=args.output_dir,
                ball_x_m=args.ball_x_m,
                ball_y_m=args.ball_y_m,
                frames=args.frames,
                run_speed_mps=args.run_speed_mps,
                run_lateral_mps=args.run_lateral_mps,
                stop_frame=args.stop_frame,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
