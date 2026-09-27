"""SIM_ONLY frozen SONIC run-through against one real MuJoCo football.

Contact diagnostic with an optional sealed SIM_ONLY feedback actor. It records
every physics-step ball/foot contact and ball velocity; no promotion is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicModelVariant, G1SonicRunupController
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
    left_hip_residual_rad: float = 0.0,
    left_knee_residual_rad: float = 0.0,
    left_ankle_residual_rad: float = 0.0,
    right_hip_residual_rad: float = 0.0,
    right_knee_residual_rad: float = 0.0,
    right_ankle_residual_rad: float = 0.0,
    contact_envelope_center_m: float = 0.48,
    contact_envelope_sigma_m: float = 0.18,
    lateral_feedback_gain_s_inv: float = 0.0,
    partition: str = "DISCOVERY",
    selector_hash: str | None = None,
    sonic_variant: str = "low_latency",
    feedback_actor: Path | None = None,
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
        or any(
            not math.isfinite(value) or abs(value) > 0.25
            for value in (
                left_hip_residual_rad,
                left_knee_residual_rad,
                left_ankle_residual_rad,
                right_hip_residual_rad,
                right_knee_residual_rad,
                right_ankle_residual_rad,
            )
        )
        or not math.isfinite(contact_envelope_center_m)
        or not 0.35 <= contact_envelope_center_m <= 0.65
        or not math.isfinite(contact_envelope_sigma_m)
        or not 0.08 <= contact_envelope_sigma_m <= 0.25
        or not math.isfinite(lateral_feedback_gain_s_inv)
        or not 0.0 <= lateral_feedback_gain_s_inv <= 1.5
        or partition not in ("DISCOVERY", "FRESH")
        or (
            selector_hash is not None
            and (not selector_hash.startswith("sha256:") or len(selector_hash) != 71)
        )
        or sonic_variant not in ("low_latency", "sonic_v1_1")
        or (
            feedback_actor is not None
            and any(
                value != 0.0
                for value in (
                    left_hip_residual_rad,
                    left_knee_residual_rad,
                    left_ankle_residual_rad,
                    right_hip_residual_rad,
                    right_knee_residual_rad,
                    right_ankle_residual_rad,
                )
            )
        )
    ):
        raise ValueError("new output and bounded ball position required")
    source_hash = hash_bytes(Path(__file__).read_bytes())
    actor_commitment: str | None = None
    actor_matrix: NDArray[np.float64] | None = None
    actor_bias: NDArray[np.float64] | None = None
    if feedback_actor is not None:
        actor_payload = json.loads(feedback_actor.read_text(encoding="utf-8"))
        actor_commitment = actor_payload.pop("result_hash", None)
        if (
            actor_commitment != hash_json(actor_payload)
            or actor_payload.get("schema") != "rosclaw_soccer.rsi.mjx_contact_residual_es.v1"
            or actor_payload.get("promotion_authorized") is not False
        ):
            raise ValueError("sealed unpromoted SIM_ONLY feedback actor required")
        actor_parameters = np.asarray(actor_payload["parameters"], dtype=np.float64)
        if actor_parameters.shape != (64,) or not np.isfinite(actor_parameters).all():
            raise ValueError("bounded 64-parameter feedback actor required")
        actor_matrix = actor_parameters[:60].reshape(4, 15)
        actor_bias = actor_parameters[60:]
    execution_id = uuid.uuid4().hex
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
            model_variant=cast(G1SonicModelVariant, sonic_variant),
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
    robot_contacts: list[dict[str, Any]] = []
    physics_qpos = []
    saturated_substeps = 0
    peak_torque_demand_ratio = 0.0
    peak_pelvis_tilt_rad = 0.0
    actor_projection_count = 0
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body < 0:
        raise ValueError("compiled G1 pelvis body missing")

    def belongs_to_robot(body_id: int) -> bool:
        while body_id > 0:
            if body_id == pelvis_body:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    initial_ball_position = data.qpos[36:39].copy()
    for frame in range(frames):
        lateral_error_m = float(data.qpos[37] - data.qpos[1] + 0.19)
        live_lateral_mps = float(
            np.clip(
                run_lateral_mps + lateral_feedback_gain_s_inv * lateral_error_m,
                -0.30,
                0.30,
            )
        )
        observation = _observation(
            data, frame, nav.navigation_envelope, run_speed_mps, live_lateral_mps, stop_frame
        )
        if frame == 0:
            nav.start_from_observation(observation)
        action = nav.propose(observation)
        target = np.asarray(action.target_rad, dtype=np.float64)
        relative_ball_x_m = float(data.qpos[36] - data.qpos[0])
        contact_envelope = math.exp(
            -0.5 * ((relative_ball_x_m - contact_envelope_center_m) / contact_envelope_sigma_m) ** 2
        )
        target[0] += left_hip_residual_rad * contact_envelope
        target[3] += left_knee_residual_rad * contact_envelope
        target[4] += left_ankle_residual_rad * contact_envelope
        target[6] += right_hip_residual_rad * contact_envelope
        target[9] += right_knee_residual_rad * contact_envelope
        target[10] += right_ankle_residual_rad * contact_envelope
        if actor_matrix is not None and actor_bias is not None:
            actor_indices = np.asarray((0, 3, 6, 9))
            actor_features = np.concatenate(
                (
                    np.asarray(
                        (
                            relative_ball_x_m,
                            data.qpos[37] - data.qpos[1],
                            data.qvel[35] - data.qvel[0],
                            data.qvel[0],
                            data.qpos[2] - 0.75,
                        )
                    ),
                    data.qpos[7 + actor_indices],
                    data.qvel[6 + actor_indices] / 5.0,
                    np.asarray(
                        (
                            bool(foot_contacts),
                            any(
                                "foot" not in row["geom"] and "ankle" not in row["geom"]
                                for row in robot_contacts
                            ),
                        ),
                        dtype=np.float64,
                    ),
                )
            )
            if actor_features.shape != (15,) or not np.isfinite(actor_features).all():
                raise ValueError("nonfinite deployable motor observation")
            gate = float((0.05 < relative_ball_x_m < 1.2) or bool(foot_contacts))
            residual = 0.12 * gate * np.tanh(actor_matrix @ actor_features + actor_bias)
            limits = model.jnt_range[model.actuator_trnid[actor_indices, 0]]
            modified = np.clip(target[actor_indices] + residual, limits[:, 0], limits[:, 1])
            actor_projection_count += int(
                np.count_nonzero(np.abs(modified - target[actor_indices] - residual) > 1e-8)
            )
            target[actor_indices] = modified
        for substep in range(10):
            demanded_torque = (target - data.qpos[7:36]) * nav.backend.kp - data.qvel[
                6:35
            ] * nav.backend.kd
            peak_torque_demand_ratio = max(
                peak_torque_demand_ratio,
                float(np.max(np.abs(demanded_torque) / torque_limits)),
            )
            saturated_substeps += int(np.any(np.abs(demanded_torque) > torque_limits))
            torque = np.clip(demanded_torque, -torque_limits, torque_limits)
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            quat = data.qpos[3:7]
            peak_pelvis_tilt_rad = max(
                peak_pelvis_tilt_rad,
                math.acos(float(np.clip(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))),
            )
            physics_qpos.append(data.qpos.copy())
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if not (set((int(contact.geom1), int(contact.geom2))) & ball_geoms):
                    continue
                other = contact.geom2 if int(contact.geom1) in ball_geoms else contact.geom1
                body = int(model.geom_bodyid[int(other)])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or ""
                if belongs_to_robot(body):
                    robot_contacts.append(
                        {
                            "time_sec": frame * 0.02 + (substep + 1) * 0.002,
                            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                            "geom": name,
                            "ball_position_m": data.qpos[36:39].tolist(),
                            "ball_velocity_mps": data.qvel[35:38].tolist(),
                        }
                    )
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
    arrays = {
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "target": np.asarray(target_rows),
        "physics_qpos": np.asarray(physics_qpos),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite ball-contact physics")
    output_dir.mkdir(parents=True)
    path = output_dir / "trajectory.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
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
        "schema": "rosclaw_soccer.rsi.sonic_ball_contact_probe.v2",
        "execution_id": execution_id,
        "partition": partition,
        "selector_hash": selector_hash,
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "physics_hash": compiled_model_hash(model),
        "model_hash": nav.backend.qualification.qualification_hash,
        "sonic_variant": sonic_variant,
        "initial_state_hash": initial_hash,
        "ball_radius_m": ball_dimensions.radius_m,
        "ball_mass_kg": ball_dimensions.body_mass_kg,
        "ball_initial_xy_m": [ball_x_m, ball_y_m],
        "frames": frames,
        "run_speed_mps": run_speed_mps,
        "run_lateral_mps": run_lateral_mps,
        "lateral_feedback_gain_s_inv": lateral_feedback_gain_s_inv,
        "desired_ball_relative_y_m": -0.19,
        "stop_frame": stop_frame,
        "left_contact_residual_rad": [
            left_hip_residual_rad,
            left_knee_residual_rad,
            left_ankle_residual_rad,
        ],
        "right_contact_residual_rad": [
            right_hip_residual_rad,
            right_knee_residual_rad,
            right_ankle_residual_rad,
        ],
        "peak_torque_demand_ratio": peak_torque_demand_ratio,
        "actuator_saturation_fraction": saturated_substeps / (frames * 10),
        "peak_pelvis_tilt_rad": peak_pelvis_tilt_rad,
        "contact_envelope_relative_x_center_m": contact_envelope_center_m,
        "contact_envelope_relative_x_sigma_m": contact_envelope_sigma_m,
        "goal_plane_x_m": goal_spec.plane_x_m,
        "whole_ball_goal_crossed": goal_frame is not None,
        "goal_frame": goal_frame,
        "goal_crossing_ball_center_xyz_m": (
            arrays["qpos"][goal_frame, 36:39].tolist() if goal_frame is not None else None
        ),
        "foot_ball_contact_substeps": len(foot_contacts),
        "first_foot_ball_contact": foot_contacts[0] if foot_contacts else None,
        "robot_ball_contact_substeps": len(robot_contacts),
        "first_robot_ball_contact": robot_contacts[0] if robot_contacts else None,
        "first_robot_ball_contact_is_foot": bool(
            robot_contacts
            and ("foot" in robot_contacts[0]["geom"] or "ankle" in robot_contacts[0]["geom"])
        ),
        "peak_ball_speed_mps": float(ball_speed.max()),
        "final_ball_displacement_xyz_m": ball_delta[-1].tolist(),
        "minimum_pelvis_height_m": float(arrays["qpos"][:, 2].min()),
        "trajectory_hash": hash_bytes(path.read_bytes()),
        "trained_actor": feedback_actor is not None,
        "feedback_actor_hash": actor_commitment,
        "feedback_actor_projection_count": actor_projection_count,
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
    parser.add_argument("--left-hip-residual-rad", type=float, default=0.0)
    parser.add_argument("--left-knee-residual-rad", type=float, default=0.0)
    parser.add_argument("--left-ankle-residual-rad", type=float, default=0.0)
    parser.add_argument("--right-hip-residual-rad", type=float, default=0.0)
    parser.add_argument("--right-knee-residual-rad", type=float, default=0.0)
    parser.add_argument("--right-ankle-residual-rad", type=float, default=0.0)
    parser.add_argument("--contact-envelope-center-m", type=float, default=0.48)
    parser.add_argument("--contact-envelope-sigma-m", type=float, default=0.18)
    parser.add_argument("--lateral-feedback-gain-s-inv", type=float, default=0.0)
    parser.add_argument("--partition", choices=("DISCOVERY", "FRESH"), default="DISCOVERY")
    parser.add_argument("--selector-hash")
    parser.add_argument("--feedback-actor", type=Path)
    parser.add_argument(
        "--sonic-variant", choices=("low_latency", "sonic_v1_1"), default="low_latency"
    )
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
                left_hip_residual_rad=args.left_hip_residual_rad,
                left_knee_residual_rad=args.left_knee_residual_rad,
                left_ankle_residual_rad=args.left_ankle_residual_rad,
                right_hip_residual_rad=args.right_hip_residual_rad,
                right_knee_residual_rad=args.right_knee_residual_rad,
                right_ankle_residual_rad=args.right_ankle_residual_rad,
                contact_envelope_center_m=args.contact_envelope_center_m,
                contact_envelope_sigma_m=args.contact_envelope_sigma_m,
                lateral_feedback_gain_s_inv=args.lateral_feedback_gain_s_inv,
                partition=args.partition,
                selector_hash=args.selector_hash,
                sonic_variant=args.sonic_variant,
                feedback_actor=args.feedback_actor,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
