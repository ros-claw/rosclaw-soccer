"""SIM_ONLY 500 Hz contact-point dynamics audit of clean-touch parent and GPU candidate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_clean_touch_control_es import (
    ACTION_LIMIT_RAD,
    EXAM_FRAME,
    JOINTS,
    SNAPSHOT,
    WEIGHTS,
)
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    locomotion_contact_teacher_effect,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def _body_point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, point: NDArray[np.float64]
) -> NDArray[np.float64]:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point, body)
    return np.asarray(jacp @ data.qvel, dtype=np.float64)


def _run(
    model: mujoco.MjModel,
    arrays: dict[str, NDArray[np.float64]],
    weights: NDArray[np.float64],
    joint_indices: tuple[int, ...] = JOINTS,
    ball_course: tuple[float, float, float] | None = None,
    post_touch_bias: NDArray[np.float64] | None = None,
    near_touch_bias: NDArray[np.float64] | None = None,
    precontact_pulse_weights: NDArray[np.float64] | None = None,
    substep_feedback: bool = False,
    receiving_leg_kp_scale: float = 1.0,
    task_space_velocity_gain: float | None = None,
    privileged_teacher_lateral_sign: float | None = None,
    privileged_teacher_torque_scale: float = 1.0,
    support_posture_gain: float | None = None,
    sample_hook: Callable[
        [NDArray[np.float64], NDArray[np.float64], bool, float, NDArray[np.float64]], None
    ]
    | None = None,
    actor_torque_fn: Callable[
        [NDArray[np.float64], NDArray[np.float64], bool, float], NDArray[np.float64]
    ]
    | None = None,
) -> dict[str, Any]:
    if (
        type(receiving_leg_kp_scale) not in (float, int)
        or not np.isfinite(receiving_leg_kp_scale)
        or not 0.4 <= receiving_leg_kp_scale <= 1.0
    ):
        raise ValueError("bounded SIM_ONLY receiving-leg stiffness scale required")
    if task_space_velocity_gain is not None and (
        type(task_space_velocity_gain) not in (float, int)
        or not np.isfinite(task_space_velocity_gain)
        or not 0.0 <= task_space_velocity_gain <= 1.0
    ):
        raise ValueError("bounded causal task-space teacher gain required")
    if privileged_teacher_lateral_sign is not None and (
        type(privileged_teacher_lateral_sign) not in (float, int)
        or privileged_teacher_lateral_sign not in (-1.0, 1.0)
    ):
        raise ValueError("explicit privileged left-foot teacher sign required")
    if (
        type(privileged_teacher_torque_scale) not in (float, int)
        or not np.isfinite(privileged_teacher_torque_scale)
        or not 0.25 <= privileged_teacher_torque_scale <= 1.0
        or (privileged_teacher_lateral_sign is None and privileged_teacher_torque_scale != 1.0)
    ):
        raise ValueError("bounded explicit privileged teacher torque scale required")
    if support_posture_gain is not None and (
        type(support_posture_gain) not in (float, int)
        or not np.isfinite(support_posture_gain)
        or not -0.20 <= support_posture_gain <= 0.20
        or privileged_teacher_lateral_sign is None
    ):
        raise ValueError("bounded SIM_ONLY coupled support posture feedback required")
    if actor_torque_fn is not None and (
        privileged_teacher_lateral_sign is not None or support_posture_gain is not None
    ):
        raise ValueError("learned actor must replace, never stack on, privileged teachers")
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["sonic_recorded_qpos"][SNAPSHOT]
    data.qvel[:] = arrays["sonic_recorded_qvel"][SNAPSHOT]
    if ball_course is not None:
        x, y, vx = ball_course
        if (
            not all(np.isfinite(value) for value in ball_course)
            or not 2.4 <= x <= 2.7
            or not 1.2 <= y <= 1.5
            or not -0.95 <= vx <= -0.75
        ):
            raise ValueError("bounded declared CPU training ball course required")
        base_vx = float(data.qvel[35])
        data.qpos[36] = x
        data.qpos[37] = y
        data.qvel[35] = vx
        data.qvel[39] += (vx - base_vx) / 0.115
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    index = np.asarray(joint_indices)
    limits = model.jnt_range[model.actuator_trnid[list(joint_indices), 0]]
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS)
    features = 5 + 2 * len(joint_indices)
    matrix = weights[: len(joint_indices) * features].reshape(len(joint_indices), features)
    bias = weights[len(joint_indices) * features :]
    foot_geom = -1
    foot_body = -1
    teacher_config = None
    teacher_ankle_body = -1
    if privileged_teacher_lateral_sign is not None:
        teacher_config = G1LocomotionContactTeacherConfig()
        teacher_ankle_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        if teacher_ankle_body < 0:
            raise ValueError("qualified left ankle teacher geometry required")
    pelvis_body = -1
    support_limits = None
    if support_posture_gain is not None:
        pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        support_names = (
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        )
        actual_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
            for i in (6, 7, 10, 11)
        )
        if pelvis_body < 0 or actual_names != support_names:
            raise ValueError("qualified pelvis and right-support joints required")
        support_limits = model.jnt_range[model.actuator_trnid[[6, 7, 10, 11], 0]]
    if task_space_velocity_gain is not None:
        expected = (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        )
        actual = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
            for i in JOINTS
        )
        foot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot3_collision")
        if actual != expected or foot_geom < 0:
            raise ValueError("qualified left-foot task-space geometry required")
        foot_body = int(model.geom_bodyid[foot_geom])
    if post_touch_bias is not None and (
        post_touch_bias.shape != (len(joint_indices),) or not np.isfinite(post_touch_bias).all()
    ):
        raise ValueError("finite post-touch bias must match authorized joints")
    if near_touch_bias is not None and (
        near_touch_bias.shape != (len(joint_indices),) or not np.isfinite(near_touch_bias).all()
    ):
        raise ValueError("finite near-touch bias must match authorized joints")
    if precontact_pulse_weights is not None and (
        precontact_pulse_weights.shape != (len(joint_indices),)
        or not np.isfinite(precontact_pulse_weights).all()
    ):
        raise ValueError("finite precontact pulse must match authorized joints")
    first: dict[str, Any] | None = None
    first_nonfoot: dict[str, Any] | None = None
    foot_impulse = 0.0
    nonfoot_impulse = 0.0
    exam_ball_velocity = None
    exam_ball_distance = None
    minimum = float(data.qpos[2])
    maximum_tilt = 0.0
    impedance_substeps = 0
    task_space_active_frames = 0
    privileged_teacher_active_substeps = 0
    privileged_teacher_peak_torque_nm = 0.0
    for frame in range(SNAPSHOT + 1, 101):
        base_target = arrays["sonic_recorded_target"][frame]
        kp = arrays["sonic_recorded_kp"][frame]
        kd = arrays["sonic_recorded_kd"][frame]

        def target_at_current_state(
            base_target: NDArray[np.float64],
            first: dict[str, Any] | None,
            first_nonfoot: dict[str, Any] | None,
            *,
            with_support: bool = True,
        ) -> NDArray[np.float64]:
            target = base_target.copy()
            dx = data.qpos[36] - data.qpos[0]
            if 0.12 < dx < 0.85:
                feature = np.concatenate(
                    (
                        np.asarray(
                            (
                                dx,
                                data.qpos[37] - data.qpos[1],
                                data.qvel[35] - data.qvel[0],
                                data.qvel[36] - data.qvel[1],
                                data.qpos[2] - 0.75,
                            )
                        ),
                        data.qpos[7 + index],
                        data.qvel[6 + index] / 5.0,
                    )
                )
                phase_bias = bias
                if first is None and dx <= 0.50 and near_touch_bias is not None:
                    phase_bias = near_touch_bias
                elif (
                    post_touch_bias is not None
                    and first is not None
                    and first["kind"] == "foot"
                    and first_nonfoot is None
                ):
                    phase_bias = post_touch_bias
                pulse = 0.0
                if first is None and precontact_pulse_weights is not None:
                    pulse = float(np.exp(-0.5 * ((dx - 0.45) / 0.10) ** 2))
                extra = (
                    pulse * precontact_pulse_weights
                    if precontact_pulse_weights is not None
                    else 0.0
                )
                residual = ACTION_LIMIT_RAD * np.tanh(matrix @ feature + phase_bias + extra)
                target[index] = np.clip(target[index] + residual, limits[:, 0], limits[:, 1])
            if support_posture_gain is not None and with_support:
                rotation = np.asarray(data.xmat[pelvis_body], dtype=np.float64).reshape(3, 3)
                pitch = -float(rotation[2, 0]) + 0.05 * float(data.qvel[4])
                roll = float(rotation[2, 1]) + 0.05 * float(data.qvel[3])
                correction = np.clip(
                    support_posture_gain * np.asarray((-pitch, -roll, pitch, roll)),
                    -0.08,
                    0.08,
                )
                assert support_limits is not None
                target[[6, 7, 10, 11]] = np.clip(
                    target[[6, 7, 10, 11]] + correction,
                    support_limits[:, 0],
                    support_limits[:, 1],
                )
            return target

        target = target_at_current_state(base_target, first, first_nonfoot)
        unassisted_target = (
            target_at_current_state(base_target, first, first_nonfoot, with_support=False)
            if sample_hook is not None and support_posture_gain is not None
            else target
        )
        if task_space_velocity_gain is not None and first is None:
            foot_point = np.asarray(data.geom_xpos[foot_geom], dtype=np.float64)
            ball_point = np.asarray(data.qpos[36:39], dtype=np.float64)
            foot_distance = float(np.linalg.norm(foot_point - ball_point))
            dx = float(data.qpos[36] - data.qpos[0])
            if foot_distance <= 0.55 and 0.12 < dx < 0.85:
                jacp = np.zeros((3, model.nv), dtype=np.float64)
                jacr = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jac(model, data, jacp, jacr, foot_point, foot_body)
                foot_velocity = jacp @ data.qvel
                leg_jacobian = jacp[:, 6 + np.asarray(JOINTS)]
                desired_velocity = task_space_velocity_gain * data.qvel[35:38] - foot_velocity
                correction = (
                    0.08
                    * leg_jacobian.T
                    @ np.linalg.solve(
                        leg_jacobian @ leg_jacobian.T + 0.05**2 * np.eye(3),
                        desired_velocity,
                    )
                )
                leg_limits = model.jnt_range[model.actuator_trnid[list(JOINTS), 0]]
                existing_residual = target[list(JOINTS)] - base_target[list(JOINTS)]
                target[list(JOINTS)] = np.clip(
                    base_target[list(JOINTS)]
                    + np.clip(existing_residual + correction, -0.12, 0.12),
                    leg_limits[:, 0],
                    leg_limits[:, 1],
                )
                task_space_active_frames += 1
        for substep in range(10):
            if substep_feedback and substep > 0:
                target = target_at_current_state(base_target, first, first_nonfoot)
            kp_effective = kp
            dx_now = float(data.qpos[36] - data.qpos[0])
            impedance_active = (
                receiving_leg_kp_scale < 1.0
                and first_nonfoot is None
                and (
                    (first is None and 0.12 < dx_now <= 0.55)
                    or (
                        first is not None
                        and first["kind"] == "foot"
                        and frame - int(first["frame"]) <= 6
                    )
                )
            )
            if impedance_active:
                kp_effective = kp.copy()
                kp_effective[[0, 1, 3, 4, 5]] *= receiving_leg_kp_scale
                impedance_substeps += 1
            raw_torque = kp_effective * (target - data.qpos[7:36]) - kd * data.qvel[6:35]
            added_torque = kp_effective * (target - unassisted_target)
            contact_elapsed_sec = -1.0
            if first is not None and first["kind"] == "foot":
                contact_elapsed_sec = max(
                    0.0,
                    float(
                        ((frame - int(first["frame"])) * 10 + substep - int(first["substep"]))
                        * model.opt.timestep
                    ),
                )
            if teacher_config is not None and first_nonfoot is None:
                contact_progress = None
                teacher_window_open = first is None
                if first is not None and first["kind"] == "foot":
                    elapsed = (
                        (frame - int(first["frame"])) * 10 + substep - int(first["substep"])
                    ) * model.opt.timestep
                    teacher_window_open = 0.0 <= elapsed <= 0.12
                    if teacher_window_open:
                        contact_progress = float(np.clip(elapsed / 0.12, 0.0, 1.0))
                if teacher_window_open:
                    effect = locomotion_contact_teacher_effect(
                        model=model,
                        data=data,
                        ankle_body_id=teacher_ankle_body,
                        actuated_dof_indices=np.arange(6, 35, dtype=np.int64),
                        ball_position_m=np.asarray(data.qpos[36:39], dtype=np.float64),
                        ball_velocity_mps=np.asarray(data.qvel[35:38], dtype=np.float64),
                        desired_ball_direction_xy=np.asarray((1.0, 0.0), dtype=np.float64),
                        contact_mode="receive",
                        local_lateral_sign=privileged_teacher_lateral_sign,
                        contact_recent=first is not None,
                        config=teacher_config,
                        receive_capture_progress=contact_progress,
                    )
                    if effect.active:
                        effect_torque = privileged_teacher_torque_scale * effect.torque_nm
                        raw_torque += effect_torque
                        added_torque += effect_torque
                        privileged_teacher_active_substeps += 1
                        privileged_teacher_peak_torque_nm = max(
                            privileged_teacher_peak_torque_nm,
                            float(
                                privileged_teacher_torque_scale * np.max(np.abs(effect.torque_nm))
                            ),
                        )
            if sample_hook is not None:
                sample_hook(
                    data.qpos.copy(),
                    data.qvel.copy(),
                    first is not None and first["kind"] == "foot",
                    contact_elapsed_sec,
                    added_torque.copy(),
                )
            if actor_torque_fn is not None:
                actor_torque = actor_torque_fn(
                    data.qpos.copy(),
                    data.qvel.copy(),
                    first is not None and first["kind"] == "foot",
                    contact_elapsed_sec,
                )
                if (
                    not isinstance(actor_torque, np.ndarray)
                    or actor_torque.shape != (29,)
                    or not np.isfinite(actor_torque).all()
                    or np.max(np.abs(actor_torque)) > 14.0 + 1e-10
                ):
                    raise ValueError("bounded finite 29-joint learned torque residual required")
                raw_torque += actor_torque
            data.ctrl[:] = np.clip(raw_torque, -torque_limit, torque_limit)
            mujoco.mj_step(model, data)
            minimum = min(minimum, float(data.qpos[2]))
            quat = data.qpos[3:7]
            maximum_tilt = max(
                maximum_tilt,
                float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))),
            )
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if not robot_mask[other]:
                    continue
                kind = "foot" if foot_mask[other] else "nonfoot"
                point = np.asarray(contact.pos, dtype=np.float64)
                ball_velocity = _body_point_velocity(
                    model, data, int(model.geom_bodyid[ball_geom]), point
                )
                body_velocity = _body_point_velocity(
                    model, data, int(model.geom_bodyid[other]), point
                )
                relative = ball_velocity - body_velocity
                normal = np.asarray(contact.frame[:3], dtype=np.float64)
                force: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_id, force)
                impulse = max(0.0, float(force[0])) * model.opt.timestep
                if kind == "foot":
                    foot_impulse += impulse
                else:
                    nonfoot_impulse += impulse
                row = {
                    "frame": frame,
                    "substep": substep,
                    "kind": kind,
                    "ball_pelvis_dx_m": float(data.qpos[36] - data.qpos[0]),
                    "x_closing_speed_mps": float(data.qvel[0] - data.qvel[35]),
                    "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                    "contact_point_m": point.tolist(),
                    "ball_point_velocity_mps": ball_velocity.tolist(),
                    "body_point_velocity_mps": body_velocity.tolist(),
                    "relative_speed_mps": float(np.linalg.norm(relative)),
                    "absolute_normal_relative_speed_mps": abs(float(relative @ normal)),
                    "normal_force_n": float(force[0]),
                }
                if first is None:
                    first = row
                if kind == "nonfoot" and first_nonfoot is None:
                    first_nonfoot = row
        if frame == EXAM_FRAME:
            exam_ball_velocity = data.qvel[35:37].copy().tolist()
            exam_ball_distance = float(np.linalg.norm(data.qpos[36:38] - data.qpos[:2]))
    if exam_ball_velocity is None or exam_ball_distance is None:
        raise RuntimeError("missing declared half-second physics frame")
    return {
        "first_contact": first,
        "first_nonfoot_contact": first_nonfoot,
        "foot_normal_impulse_ns": foot_impulse,
        "nonfoot_normal_impulse_ns": nonfoot_impulse,
        "exam_ball_velocity_xy_mps": exam_ball_velocity,
        "exam_ball_speed_mps": float(np.linalg.norm(exam_ball_velocity)),
        "exam_ball_pelvis_distance_m": exam_ball_distance,
        "minimum_pelvis_height_m": minimum,
        "maximum_tilt_rad": maximum_tilt,
        "impedance_substeps": impedance_substeps,
        "task_space_active_frames": task_space_active_frames,
        "privileged_teacher_active_substeps": privileged_teacher_active_substeps,
        "privileged_teacher_peak_torque_nm": privileged_teacher_peak_torque_nm,
    }


def audit(*, asset_root: Path, captured: Path, candidate: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY audit directory required")
    trace_path = captured / "motor-trace.npz"
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    capture_hash = capture.pop("report_hash")
    candidate_data: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_hash = candidate_data.pop("report_hash")
    if (
        capture_hash != hash_json(capture)
        or capture["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or candidate_hash != hash_json(candidate_data)
        or candidate_data["promotion_authorized"] is not False
    ):
        raise ValueError("sealed unpromoted measured evidence required")
    weights32 = np.asarray(candidate_data["weights"], dtype=np.float32)
    if (
        weights32.shape != (WEIGHTS,)
        or not np.isfinite(weights32).all()
        or candidate_data["weights_hash"] != hash_bytes(weights32.tobytes())
    ):
        raise ValueError("finite bounded candidate weights required")
    weights = weights32.astype(np.float64)
    with np.load(trace_path, allow_pickle=False) as trace:
        arrays = {
            key: np.asarray(trace[key], dtype=np.float64)
            for key in (
                "sonic_recorded_qpos",
                "sonic_recorded_qvel",
                "sonic_recorded_target",
                "sonic_recorded_kp",
                "sonic_recorded_kd",
            )
        }
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    parent = _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64))
    child = _run(model, arrays, weights)
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("contact audit source changed during CPU physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_contact_dynamics_audit.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "capture_hash": capture_hash,
        "candidate_hash": candidate_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "parent": parent,
        "candidate": child,
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
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = audit(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: report[key] for key in ("report_hash", "parent", "candidate")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
