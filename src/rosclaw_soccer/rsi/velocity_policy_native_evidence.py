"""Typed native evidence for the HG29 velocity actor, not the SONIC action codec."""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_velocity_observation import (
    G1VelocityObservation,
    world_command_to_body,
)
from rosclaw_soccer.providers.g1.velocity_policy import (
    VelocityPolicyParameters,
    observation98,
    phase_delta,
    target29,
)
from rosclaw_soccer.rsi.locomotion_response_protocol import LocomotionResponseCourse
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def review_velocity_recording(
    raw: dict[str, np.ndarray],
    params: VelocityPolicyParameters,
    sensor: G1VelocityObservation,
    course: LocomotionResponseCourse,
) -> dict[str, Any]:
    shapes = dict(
        canonical_qpos=(300, 43),
        canonical_qvel=(300, 41),
        joint_target_rad=(300, 29),
        pd_requested_torque_nm=(300, 10, 29),
        torque_nm=(300, 10, 29),
        actual_actuator_force_nm=(300, 10, 29),
        pelvis_z_per_substep_m=(300, 10),
        force_n=(300, 6),
        ball_position_after_step_m=(300, 3),
        target_body_canonical_qpos_after_substep=(300, 10, 43),
        target_body_canonical_qvel_after_substep=(300, 10, 41),
        ball_contact_force_by_body_n=(300, 10, sensor.model.nbody),
        velocity_policy_observation=(300, 98),
        velocity_policy_action_hg=(300, 29),
        command_body_mps=(300, 3),
        command_world_mps=(300, 3),
        angular_velocity_body_radps=(300, 3),
        gait_phase=(300,),
    )
    if set(raw) != set(shapes):
        raise ValueError(
            "complete typed velocity-specific native record required; no SONIC codec aliases"
        )
    f32 = {
        "velocity_policy_observation",
        "velocity_policy_action_hg",
        "command_body_mps",
        "gait_phase",
    }
    for k, shape in shapes.items():
        a = raw[k]
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != (np.float32 if k in f32 else np.float64)
            or not np.isfinite(a).all()
        ):
            raise ValueError("finite exact velocity-native field required: " + k)
    if np.any(raw["ball_contact_force_by_body_n"] < 0) or np.any(raw["force_n"] < 0):
        raise ValueError("nonnegative solved contact force magnitudes required")
    q, v = raw["canonical_qpos"], raw["canonical_qvel"]
    nq, nv = (
        raw["target_body_canonical_qpos_after_substep"],
        raw["target_body_canonical_qvel_after_substep"],
    )
    if not np.array_equal(q[1:], nq[:-1, -1]) or not np.array_equal(v[1:], nv[:-1, -1]):
        raise ValueError("continuous actual native trajectory required")
    kp, kd = np.asarray(params.stiffness), np.asarray(params.damping)
    caps = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    phase = phase_delta()
    last = np.zeros(29, dtype=np.float32)
    for f in range(300):
        cmd = course.command(f)
        body = world_command_to_body(q[f], cmd)
        omega = sensor.angular_velocity_body(q[f], v[f])
        if (
            not np.array_equal(raw["command_world_mps"][f], cmd)
            or not np.array_equal(raw["command_body_mps"][f], body)
            or not np.array_equal(raw["angular_velocity_body_radps"][f], omega)
        ):
            raise ValueError(
                "current actual-state command/kinematics differ from independent derivation"
            )
        obs, phase = observation98(q[f], v[f], omega, body, last, phase, params)
        target = target29(raw["velocity_policy_action_hg"][f], params)
        if (
            not np.array_equal(raw["velocity_policy_observation"][f], obs)
            or raw["gait_phase"][f] != phase
            or not np.array_equal(raw["joint_target_rad"][f], target)
        ):
            raise ValueError("policy observation/phase/last action/HG29 target codec differs")
        last = raw["velocity_policy_action_hg"][f].copy()
        before_q = np.concatenate((q[f, None, :], nq[f, :-1]), axis=0)
        before_v = np.concatenate((v[f, None, :], nv[f, :-1]), axis=0)
        requested = (target - before_q[:, 7:36]) * kp - before_v[:, 6:35] * kd
        applied = np.clip(requested, -caps, caps)
        if (
            not np.array_equal(requested, raw["pd_requested_torque_nm"][f])
            or not np.array_equal(applied, raw["torque_nm"][f])
            or np.any(abs(raw["actual_actuator_force_nm"][f]) > caps + 1e-8)
        ):
            raise ValueError("actual native PD/torque limit evidence differs")
    end = course.onset_frame + course.pulse_frames
    mean = v[end - 50 : end, :2].mean(axis=0)
    desired = np.array(course.command_xy_mps)
    error = float(np.linalg.norm(mean - desired))
    terminal = float(np.linalg.norm(v[250:300, :2].mean(axis=0)))
    drift = float(np.linalg.norm(q[-1, :2] - q[end, :2]))
    upright = 1 - 2 * (nq[:, :, 4] ** 2 + nq[:, :, 5] ** 2)
    minimum_height = float(min(raw["pelvis_z_per_substep_m"].min(), nq[:, :, 2].min()))
    robot_ball = float(raw["ball_contact_force_by_body_n"][:, :, 1:].max())
    stable = bool(minimum_height >= 0.65 and upright.min() >= 0.9 and robot_ball <= 1)
    return dict(
        mean_world_velocity_last_pulse_second_mps=mean.tolist(),
        desired_world_velocity_mps=desired.tolist(),
        pulse_mean_velocity_error_mps=error,
        terminal_mean_speed_mps=terminal,
        after_pulse_root_drift_m=drift,
        minimum_pelvis_height_m=minimum_height,
        minimum_upright=float(upright.min()),
        maximum_robot_ball_force_n=robot_ball,
        physical_response_guard=stable,
        command_response_research_gate=bool(
            stable and error <= 0.08 and terminal <= 0.05 and drift <= 0.15
        ),
        joint_target_codec="HG29_float32_scale_plus_offset_not_SONIC_ISAAC",
        all_policy_features_and_native_pd_reconstructed=True,
        football_skill_proven=False,
        new_neural_calls=0,
        new_native_steps=0,
    )
