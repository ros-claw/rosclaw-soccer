"""SIM_ONLY one-G1 live SONIC/ball-feedback proxy, benchmarked against eight-G1 tape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.providers.g1.receiving_foot_capture import ReceivingFootCaptureTeacher
from rosclaw_soccer.providers.g1.receiving_phase_adapter import ReceivingPhaseAdapter
from rosclaw_soccer.providers.g1.receiving_sonic import ReceivingSonicBallFollowOption
from rosclaw_soccer.sim.contracts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    hash_bytes,
    hash_json,
)
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.skills.team.independent_team_world import _project_joint_safe_torque
from rosclaw_soccer.skills.team.motor_option import (
    TeamBallContact,
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
)
from rosclaw_soccer.training.receiving_classroom import r0_receiving_configuration
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

FRAMES = 130
FOCAL = "red.finisher"


def evaluate(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
    left_hip_roll_offset_rad: float = 0.0,
    adapter_parameters: NDArray[np.float64] | None = None,
    foot_capture_gain: float = 0.0,
    precontact_foot_gain: float = 0.0,
    foot_offset_x_m: float = -0.18,
    foot_offset_y_m: float = 0.03,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "sonic": root / "src/rosclaw_soccer/providers/g1/receiving_sonic.py",
        "bridge": root / "src/rosclaw_soccer/providers/g1/qualified_receiving_student.py",
        "single_world": root / "src/rosclaw_soccer/world/field.py",
        "adapter": root / "src/rosclaw_soccer/providers/g1/receiving_phase_adapter.py",
        "foot_teacher": root / "src/rosclaw_soccer/providers/g1/receiving_foot_capture.py",
    }
    source_hashes = {key: hash_bytes(path.read_bytes()) for key, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY one-G1 live proxy directory required")
    if (
        type(left_hip_roll_offset_rad) is not float
        or left_hip_roll_offset_rad not in (0.0, -0.08)
        or adapter_parameters is not None
        and left_hip_roll_offset_rad != 0.0
        or type(foot_capture_gain) is not float
        or not np.isfinite(foot_capture_gain)
        or not 0.0 <= foot_capture_gain <= 1.0
        or foot_capture_gain != 0.0
        and (left_hip_roll_offset_rad != 0.0 or adapter_parameters is not None)
        or type(precontact_foot_gain) is not float
        or not np.isfinite(precontact_foot_gain)
        or not 0.0 <= precontact_foot_gain <= 1.0
        or precontact_foot_gain != 0.0
        and (left_hip_roll_offset_rad != 0.0 or adapter_parameters is not None)
    ):
        raise ValueError("predeclared zero or known minus-0.08 hip intervention required")
    adapter = (
        None
        if adapter_parameters is None
        else ReceivingPhaseAdapter(np.asarray(adapter_parameters, dtype=np.float64))
    )
    foot_teacher = (
        None
        if foot_capture_gain == precontact_foot_gain == 0.0
        else ReceivingFootCaptureTeacher(
            pre_gain=precontact_foot_gain,
            post_gain=foot_capture_gain,
            offset_x_m=foot_offset_x_m,
            offset_y_m=foot_offset_y_m,
        )
    )
    if foot_teacher is None and (foot_offset_x_m != -0.18 or foot_offset_y_m != 0.03):
        raise ValueError("foot offsets require an active SIM_ONLY teacher")
    commitment: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    tape_hash = commitment.pop("report_hash")
    tape_path = captured / "live-motor-tape.npz"
    if (
        tape_hash != hash_json(commitment)
        or commitment["schema"] != "rosclaw_soccer.rsi.receiving_live_motor_capture.v1"
        or commitment["physical_prefix_exact"] is not True
        or commitment["trace_hash"] != hash_bytes(tape_path.read_bytes())
        or commitment["promotion_authorized"] is not False
    ):
        raise ValueError("sealed exact eight-G1 live SONIC observation tape required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if student.model_hash != commitment["student_model_hash"]:
        raise ValueError("student differs from sealed eight-G1 tape")
    with np.load(tape_path, allow_pickle=False) as payload:
        tape = {key: np.asarray(payload[key]) for key in payload.files}
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise ValueError("exact one-G1 29-motor native MuJoCo model required")
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_DDS_JOINT_NAMES]
    )
    if np.any(joint_ids < 0):
        raise ValueError("complete local one-G1 motor joint identities required")
    qpos_ids = model.jnt_qposadr[joint_ids]
    qvel_ids = model.jnt_dofadr[joint_ids]
    data = mujoco.MjData(model)
    data.qpos[:] = tape["sonic_recorded_qpos"][0]
    data.qvel[:] = tape["sonic_recorded_qvel"][0]
    mujoco.mj_forward(model, data)
    world, _ = r0_receiving_configuration()
    option = ReceivingSonicBallFollowOption(
        sonic_model_root, FOCAL, start_frame=0, response_gain=0.75
    )
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    if ball_geom < 0:
        raise ValueError("physical soccer ball geom required")
    guarded = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    target_errors = []
    pre_qpos_errors = []
    pre_qvel_errors = []
    ball_pose = []
    ball_velocity = []
    pelvis_pose = []
    left_foot_position = []
    right_foot_position = []
    left_ankle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if left_ankle_body < 0 or right_ankle_body < 0:
        raise ValueError("measured left and right ankle bodies required")
    foot_frames: list[int] = []
    nonfoot_frames: list[int] = []
    minimum_pelvis = float("inf")
    maximum_tilt = 0.0
    first_foot_time: float | None = None
    for frame in range(FRAMES):
        local_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
        local_qvel = np.asarray(data.qvel, dtype=np.float64).copy()
        pre_qpos_errors.append(
            float(np.max(np.abs(local_qpos - tape["sonic_recorded_qpos"][frame])))
        )
        pre_qvel_errors.append(
            float(np.max(np.abs(local_qvel - tape["sonic_recorded_qvel"][frame])))
        )
        original_command = tuple(
            float(value) for value in tape["sonic_recorded_navigation_command"][frame]
        )
        observation = TeamMotorObservation(
            agent_id=FOCAL,
            frame=frame,
            time_sec=frame * 0.02,
            intent=str(tape["sonic_recorded_intent"][frame]),
            prospective_owner=bool(tape["sonic_recorded_prospective_owner"][frame]),
            qpos=tuple(float(value) for value in local_qpos),
            qvel=tuple(float(value) for value in local_qvel),
            target_position_m=tuple(
                float(value) for value in tape["sonic_recorded_target_position"][frame]
            ),
            navigation_command=original_command,
            committed_receiver=bool(tape["sonic_recorded_committed_receiver"][frame]),
        )
        proposal = option.propose(observation)
        if proposal is None:
            raise RuntimeError("frame-zero private SONIC policy must propose every frame")
        foundation = np.asarray(proposal.target_rad, dtype=np.float64)
        kp = np.asarray(proposal.kp, dtype=np.float64)
        kd = np.asarray(proposal.kd, dtype=np.float64)
        target_errors.append(
            float(np.max(np.abs(foundation - tape["sonic_recorded_target"][frame])))
        )
        frame_target = foundation
        for substep in range(10):
            q = np.asarray(data.qpos[qpos_ids], dtype=np.float64)
            dq = np.asarray(data.qvel[qvel_ids], dtype=np.float64)
            if 46 <= frame <= 100 and substep == 0:
                frame_target = student.motor_target(
                    qpos=np.asarray(data.qpos, dtype=np.float64),
                    qvel=np.asarray(data.qvel, dtype=np.float64),
                    foundation_target=foundation,
                    joint_ranges=np.asarray(model.jnt_range[joint_ids]),
                )
                if left_hip_roll_offset_rad != 0.0 and frame <= 61:
                    frame_target = frame_target.copy()
                    frame_target[1] = np.clip(
                        frame_target[1] + left_hip_roll_offset_rad,
                        model.jnt_range[joint_ids[1], 0],
                        model.jnt_range[joint_ids[1], 1],
                    )
                if adapter is not None:
                    frame_target = adapter.motor_target(
                        qpos=np.asarray(data.qpos, dtype=np.float64),
                        qvel=np.asarray(data.qvel, dtype=np.float64),
                        foundation_target=frame_target,
                        joint_ranges=np.asarray(model.jnt_range[joint_ids]),
                        has_foot_contact=first_foot_time is not None,
                        elapsed_sec=(
                            -1.0
                            if first_foot_time is None
                            else max(0.0, float(data.time) - first_foot_time)
                        ),
                    )
                if foot_teacher is not None and (
                    first_foot_time is not None or 0.12 < float(data.qpos[36] - data.qpos[0]) < 0.85
                ):
                    frame_target = foot_teacher.motor_target(
                        model=model,
                        data=data,
                        left_ankle_body=left_ankle_body,
                        left_joint_dofs=np.asarray(qvel_ids[:6], dtype=np.int64),
                        left_q=q[:6],
                        foundation_target=foundation,
                        current_target=frame_target,
                        left_joint_ranges=np.asarray(model.jnt_range[joint_ids[:6]]),
                        ball_position=np.asarray(data.qpos[36:39], dtype=np.float64),
                        has_foot_contact=first_foot_time is not None,
                        elapsed_sec=(
                            -1.0
                            if first_foot_time is None
                            else max(0.0, float(data.time) - first_foot_time)
                        ),
                    )
            raw = kp * (frame_target - q) - kd * dq
            if 46 <= frame <= 100:
                raw += student.torque(
                    qpos=np.asarray(data.qpos, dtype=np.float64),
                    qvel=np.asarray(data.qvel, dtype=np.float64),
                    has_foot_contact=first_foot_time is not None,
                    elapsed_sec=(
                        -1.0
                        if first_foot_time is None
                        else max(0.0, float(data.time) - first_foot_time)
                    ),
                )
            projected = _project_joint_safe_torque(
                joint_position=q,
                joint_velocity=dq,
                commanded_torque=raw,
                joint_ranges=np.asarray(model.jnt_range[joint_ids]),
                limited=model.jnt_limited[joint_ids].astype(bool),
                margin_rad=world.joint_guard_margin_rad,
            )
            data.ctrl[:] = np.clip(projected, -guarded, guarded)
            mujoco.mj_step(model, data)
            minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
            quat = np.asarray(data.qpos[3:7])
            tilt = float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
            maximum_tilt = max(maximum_tilt, tilt)
            contacts = []
            foot_force = 0.0
            other_force = 0.0
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if ball_geom not in (a, b):
                    continue
                other = b if a == ball_geom else a
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                if int(model.geom_bodyid[other]) == 0:
                    continue
                force: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_id, force)
                normal = max(0.0, float(force[0]))
                effector = (
                    "left_foot"
                    if "left_foot" in name
                    else "right_foot"
                    if "right_foot" in name
                    else "body"
                )
                contacts.append(TeamBallContact(other, FOCAL, effector, normal))
                if effector in ("left_foot", "right_foot"):
                    foot_force = max(foot_force, normal)
                    if frame not in foot_frames:
                        foot_frames.append(frame)
                    if first_foot_time is None:
                        first_foot_time = float(data.time)
                else:
                    other_force = max(other_force, normal)
                    if frame not in nonfoot_frames:
                        nonfoot_frames.append(frame)
            option.observe_physics(
                TeamMotorPhysicsObservation(
                    time_sec=float(data.time),
                    qpos=tuple(float(value) for value in data.qpos),
                    qvel=tuple(float(value) for value in data.qvel),
                    world_bodies_safe=minimum_pelvis >= 0.55 and maximum_tilt < 0.8,
                    foot_normal_force_n=foot_force,
                    other_non_ground_normal_force_n=other_force,
                    observer_agent_id=FOCAL,
                    contacts_complete=True,
                    ball_contacts=tuple(contacts),
                )
            )
        ball_pose.append(np.asarray(data.qpos[36:43]).copy())
        ball_velocity.append(np.asarray(data.qvel[35:41]).copy())
        pelvis_pose.append(np.asarray(data.qpos[:7]).copy())
        left_foot_position.append(np.asarray(data.xpos[left_ankle_body]).copy())
        right_foot_position.append(np.asarray(data.xpos[right_ankle_body]).copy())
    ball_pose_array = np.asarray(ball_pose)
    ball_velocity_array = np.asarray(ball_velocity)
    pelvis_pose_array = np.asarray(pelvis_pose)
    left_foot_array = np.asarray(left_foot_position)
    right_foot_array = np.asarray(right_foot_position)
    foot_distance = np.minimum(
        np.linalg.norm(left_foot_array - ball_pose_array[:, :3], axis=1),
        np.linalg.norm(right_foot_array - ball_pose_array[:, :3], axis=1),
    )
    ball_speed = np.linalg.norm(ball_velocity_array[:, :3], axis=1)
    full_speed = float(np.linalg.norm(tape["ball_velocity"][86, :2]))
    proxy_speed = float(np.linalg.norm(ball_velocity_array[86, :2]))
    full_distance = float(
        np.linalg.norm(tape["ball_pose"][86, :2] - tape["red_finisher_pelvis_pose"][86, :2])
    )
    proxy_distance = float(np.linalg.norm(ball_pose_array[86, :2] - pelvis_pose_array[86, :2]))
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("single live proxy source changed during physics")
    output_dir.mkdir(parents=True)
    trajectory_path = output_dir / "single-live.npz"
    np.savez_compressed(
        trajectory_path,
        ball_pose=ball_pose_array,
        ball_velocity=ball_velocity_array,
        pelvis_pose=pelvis_pose_array,
        left_foot_position=left_foot_array,
        right_foot_position=right_foot_array,
        foot_distance_m=foot_distance,
        ball_speed_mps=ball_speed,
        target_errors=np.asarray(target_errors),
        pre_qpos_errors=np.asarray(pre_qpos_errors),
        pre_qvel_errors=np.asarray(pre_qvel_errors),
    )
    report: dict[str, Any] = {
        "schema": (
            "rosclaw_soccer.rsi.receiving_single_live_precontact_foot.v1"
            if precontact_foot_gain != 0.0
            else "rosclaw_soccer.rsi.receiving_single_live_foot_capture.v1"
            if foot_capture_gain != 0.0
            else "rosclaw_soccer.rsi.receiving_single_live_sonic_adapter.v1"
            if adapter is not None
            else "rosclaw_soccer.rsi.receiving_single_live_sonic_proxy.v1"
        ),
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "capture_report_hash": tape_hash,
        "student_model_hash": student.model_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "left_hip_roll_offset_rad": left_hip_roll_offset_rad,
        "adapter_hash": None if adapter is None else adapter.artifact_hash,
        "adapter_parameters": None if adapter is None else adapter.parameters.tolist(),
        "foot_capture_gain": foot_capture_gain,
        "precontact_foot_gain": precontact_foot_gain,
        "foot_offset_x_m": foot_offset_x_m,
        "foot_offset_y_m": foot_offset_y_m,
        "foot_teacher_hash": None if foot_teacher is None else foot_teacher.contract_hash,
        "frame_count": FRAMES,
        "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        "maximum_foundation_target_error_rad": max(target_errors),
        "maximum_pre_qpos_error": max(pre_qpos_errors),
        "maximum_pre_qvel_error": max(pre_qvel_errors),
        "proxy_first_foot_frame": foot_frames[0] if foot_frames else None,
        "proxy_foot_frames": foot_frames,
        "proxy_nonfoot_frames": nonfoot_frames,
        "minimum_pelvis_height_m": minimum_pelvis,
        "maximum_tilt_rad": maximum_tilt,
        "full_frame86_ball_speed_mps": full_speed,
        "proxy_frame86_ball_speed_mps": proxy_speed,
        "frame86_ball_speed_error_mps": abs(proxy_speed - full_speed),
        "full_frame86_ball_pelvis_distance_m": full_distance,
        "proxy_frame86_ball_pelvis_distance_m": proxy_distance,
        "frame86_ball_pelvis_distance_error_m": abs(proxy_distance - full_distance),
        "tail_maximum_foot_distance_m": float(np.max(foot_distance[110:120])),
        "tail_maximum_ball_speed_mps": float(np.max(ball_speed[110:120])),
        "tail_mean_foot_distance_m": float(np.mean(foot_distance[110:120])),
        "tail_mean_ball_speed_mps": float(np.mean(ball_speed[110:120])),
        "training_proxy_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "sonic-model-root",
        "captured",
        "warm-start",
        "training",
        "fresh",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--left-hip-roll-offset-rad", type=float, default=0.0)
    parser.add_argument("--foot-capture-gain", type=float, default=0.0)
    parser.add_argument("--precontact-foot-gain", type=float, default=0.0)
    parser.add_argument("--foot-offset-x-m", type=float, default=-0.18)
    parser.add_argument("--foot-offset-y-m", type=float, default=0.03)
    report = evaluate(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "proxy_first_foot_frame",
                    "maximum_foundation_target_error_rad",
                    "frame86_ball_speed_error_mps",
                    "frame86_ball_pelvis_distance_error_m",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
