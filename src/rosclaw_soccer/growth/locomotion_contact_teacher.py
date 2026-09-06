"""Bounded task-space teacher for locomotion-to-football contact discovery.

The teacher is deliberately SIM-only and training-only.  It never writes a
robot root or football state.  A desired football direction is converted into
an ankle target, then through MuJoCo's measured Jacobian into a bounded joint
torque residual.  The resulting ball motion must still come from an actual
foot/ball collision in the shared physics solver.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import ShotParameters, hash_json


@dataclass(frozen=True)
class G1LocomotionContactTeacherConfig:
    maximum_foot_ball_distance_m: float = 0.52
    ankle_lateral_offset_m: float = 0.10
    receive_ankle_lateral_offset_m: float = 0.18
    committed_receive_ankle_lateral_offset_m: float = 0.18
    ankle_height_offset_m: float = -0.055
    precontact_depth_m: float = 0.075
    follow_through_depth_m: float = 0.14
    receive_cushion_depth_m: float = -0.06
    receive_minimum_forward_target_m: float = 0.06
    minimum_receive_ball_speed_mps: float = 0.10
    receive_follow_through_speed_mps: float = 0.0
    committed_receive_follow_through_speed_mps: float = 0.35
    committed_receive_velocity_damping_n_per_mps: float = 15.0
    committed_receive_maximum_task_force_n: float = 120.0
    committed_receive_maximum_joint_residual_nm: float = 20.0
    committed_receive_aim_yaw_bias_rad: float = -1.00
    strike_foot_speed_mps: float = 0.80
    pass_strike_foot_speed_mps: float = 0.80
    shot_strike_foot_speed_mps: float = 2.00
    one_touch_finish_enabled: bool = True
    one_touch_finish_aim_yaw_bias_rad: float = 1.00
    one_touch_support_damping_scale: float = 3.00
    contact_memory_sec: float = 0.30
    position_gain_n_per_m: float = 180.0
    velocity_damping_n_per_mps: float = 7.0
    maximum_task_force_n: float = 85.0
    maximum_joint_residual_nm: float = 14.0
    maximum_forward_foot_offset_m: float = 0.02
    aim_yaw_bias_rad: float = 0.0
    preferred_foot: str = "nearest"
    activation_ceiling: str = "SIM_ONLY"
    training_only: bool = True
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.g1_locomotion_contact_teacher_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.maximum_foot_ball_distance_m,
            self.ankle_lateral_offset_m,
            self.receive_ankle_lateral_offset_m,
            self.committed_receive_ankle_lateral_offset_m,
            self.ankle_height_offset_m,
            self.precontact_depth_m,
            self.follow_through_depth_m,
            self.receive_cushion_depth_m,
            self.receive_minimum_forward_target_m,
            self.minimum_receive_ball_speed_mps,
            self.receive_follow_through_speed_mps,
            self.committed_receive_follow_through_speed_mps,
            self.committed_receive_velocity_damping_n_per_mps,
            self.committed_receive_maximum_task_force_n,
            self.committed_receive_maximum_joint_residual_nm,
            self.committed_receive_aim_yaw_bias_rad,
            self.strike_foot_speed_mps,
            self.pass_strike_foot_speed_mps,
            self.shot_strike_foot_speed_mps,
            self.one_touch_finish_aim_yaw_bias_rad,
            self.one_touch_support_damping_scale,
            self.contact_memory_sec,
            self.position_gain_n_per_m,
            self.velocity_damping_n_per_mps,
            self.maximum_task_force_n,
            self.maximum_joint_residual_nm,
            self.maximum_forward_foot_offset_m,
            self.aim_yaw_bias_rad,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.25 <= self.maximum_foot_ball_distance_m <= 0.80
            or not 0.08 <= self.ankle_lateral_offset_m <= 0.24
            or not 0.12 <= self.receive_ankle_lateral_offset_m <= 0.24
            or not 0.10 <= self.committed_receive_ankle_lateral_offset_m <= 0.24
            or not -0.10 <= self.ankle_height_offset_m <= 0.02
            or not 0.04 <= self.precontact_depth_m <= 0.16
            or not 0.08 <= self.follow_through_depth_m <= 0.24
            or not -0.12 <= self.receive_cushion_depth_m <= 0.12
            or not 0.02 <= self.receive_minimum_forward_target_m <= 0.15
            or not 0.05 <= self.minimum_receive_ball_speed_mps <= 0.50
            or not 0.0 <= self.receive_follow_through_speed_mps <= 1.0
            or not 0.0 <= self.committed_receive_follow_through_speed_mps <= 1.0
            or not 5.0 <= self.committed_receive_velocity_damping_n_per_mps <= 25.0
            or not 40.0 <= self.committed_receive_maximum_task_force_n <= 160.0
            or not 8.0 <= self.committed_receive_maximum_joint_residual_nm <= 25.0
            or not -1.0 <= self.committed_receive_aim_yaw_bias_rad <= 1.0
            or not 0.30 <= self.strike_foot_speed_mps <= 2.50
            or not 0.50 <= self.pass_strike_foot_speed_mps <= 2.50
            or not 0.80 <= self.shot_strike_foot_speed_mps <= 2.50
            or not isinstance(self.one_touch_finish_enabled, bool)
            or not -1.0 <= self.one_touch_finish_aim_yaw_bias_rad <= 1.0
            or not 1.0 <= self.one_touch_support_damping_scale <= 5.0
            or not 0.04 <= self.contact_memory_sec <= 0.30
            or not 20.0 <= self.position_gain_n_per_m <= 300.0
            or not 0.0 <= self.velocity_damping_n_per_mps <= 15.0
            or not 10.0 <= self.maximum_task_force_n <= 120.0
            or not 2.0 <= self.maximum_joint_residual_nm <= 20.0
            or not -0.20 <= self.maximum_forward_foot_offset_m <= 0.08
            or not -1.0 <= self.aim_yaw_bias_rad <= 1.0
            or self.preferred_foot not in {"nearest", "left", "right"}
            or self.activation_ceiling != "SIM_ONLY"
            or not self.training_only
            or self.hardware_authorized
        ):
            raise ValueError("locomotion contact teacher violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class G1LocomotionContactTeacherEffect:
    torque_nm: NDArray[np.float64]
    task_force_n: NDArray[np.float64]
    ankle_target_m: NDArray[np.float64]
    foot_ball_distance_m: float
    longitudinal_foot_offset_m: float
    active: bool


@dataclass(frozen=True)
class G1RollingOptionBridgeConfig:
    """Training-only locomotion hand-off into PASS/SHOOT contact options."""

    entry_policy_frame: int = 205
    exit_policy_frame: int = 310
    blend_frames: int = 20
    minimum_strike_stance_depth_m: float = 0.45
    maximum_strike_stance_depth_m: float = 1.20
    maximum_strike_lateral_error_m: float = 0.50
    maximum_strike_yaw_error_rad: float = 0.35
    strike_lease_duration_sec: float = 5.0
    pass_enabled: bool = False
    pass_parameters: ShotParameters = ShotParameters(
        swing_amplitude=0.72,
        foot_yaw_offset=0.04,
        foot_pitch_offset=0.02,
        recovery_step_length=0.06,
    )
    shoot_parameters: ShotParameters = ShotParameters(
        swing_amplitude=1.0,
        foot_yaw_offset=0.06,
        foot_pitch_offset=0.01,
        recovery_step_length=0.04,
    )
    activation_ceiling: str = "SIM_ONLY"
    training_only: bool = True
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.g1_rolling_option_bridge_config.v1"

    def __post_init__(self) -> None:
        if (
            not 170 <= self.entry_policy_frame <= 235
            or not self.entry_policy_frame + 40 <= self.exit_policy_frame <= 340
            or not 5 <= self.blend_frames <= 50
            or not 0.30 <= self.minimum_strike_stance_depth_m <= 0.80
            or not 0.80 <= self.maximum_strike_stance_depth_m <= 1.50
            or self.minimum_strike_stance_depth_m >= self.maximum_strike_stance_depth_m
            or not 0.20 <= self.maximum_strike_lateral_error_m <= 0.60
            or not 0.15 <= self.maximum_strike_yaw_error_rad <= 0.60
            or not 0.8 <= self.strike_lease_duration_sec <= 6.0
            or not isinstance(self.pass_enabled, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or not self.training_only
            or self.hardware_authorized
        ):
            raise ValueError("rolling option bridge violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def locomotion_contact_teacher_effect(
    *,
    model: Any,
    data: Any,
    ankle_body_id: int,
    actuated_dof_indices: NDArray[np.int64],
    ball_position_m: NDArray[np.float64],
    ball_velocity_mps: NDArray[np.float64],
    desired_ball_direction_xy: NDArray[np.float64],
    contact_mode: str,
    local_lateral_sign: float,
    contact_recent: bool,
    config: G1LocomotionContactTeacherConfig,
) -> G1LocomotionContactTeacherEffect:
    """Return a bounded J^T residual that can only succeed through contact."""

    dofs = np.asarray(actuated_dof_indices, dtype=np.int64)
    ball = np.asarray(ball_position_m, dtype=np.float64)
    ball_velocity = np.asarray(ball_velocity_mps, dtype=np.float64)
    direction_xy = np.asarray(desired_ball_direction_xy, dtype=np.float64)
    zero_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    zero_xyz: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    if (
        dofs.shape != (29,)
        or len(set(int(value) for value in dofs)) != 29
        or np.any(dofs < 0)
        or np.any(dofs >= int(model.nv))
        or ball.shape != (3,)
        or ball_velocity.shape != (3,)
        or direction_xy.shape != (2,)
        or not np.all(np.isfinite(ball))
        or not np.all(np.isfinite(ball_velocity))
        or not np.all(np.isfinite(direction_xy))
        or contact_mode not in {"receive", "strike"}
        or local_lateral_sign not in {-1.0, 1.0}
    ):
        raise ValueError("locomotion contact teacher input contract is invalid")
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1.0e-9:
        return G1LocomotionContactTeacherEffect(
            zero_torque,
            zero_xyz,
            ball.copy(),
            math.inf,
            math.inf,
            False,
        )
    direction_xy /= norm
    cosine = math.cos(config.aim_yaw_bias_rad)
    sine = math.sin(config.aim_yaw_bias_rad)
    direction_xy = np.asarray(
        (
            cosine * direction_xy[0] - sine * direction_xy[1],
            sine * direction_xy[0] + cosine * direction_xy[1],
        ),
        dtype=np.float64,
    )
    lateral_xy = np.asarray((-direction_xy[1], direction_xy[0]), dtype=np.float64)
    foot = np.asarray(data.xpos[ankle_body_id], dtype=np.float64)
    distance = float(np.linalg.norm(foot - ball))
    longitudinal_offset = float(np.dot(foot[:2] - ball[:2], direction_xy))
    depth = (
        config.follow_through_depth_m
        if contact_recent
        else config.receive_cushion_depth_m
        if contact_mode == "receive"
        else -config.precontact_depth_m
    )
    target = ball.copy()
    target[:2] += depth * direction_xy
    lateral_offset = (
        config.receive_ankle_lateral_offset_m
        if contact_mode == "receive"
        else config.ankle_lateral_offset_m
    )
    target[:2] += local_lateral_sign * lateral_offset * lateral_xy
    target[2] += config.ankle_height_offset_m
    if contact_mode == "receive" and not contact_recent:
        forward_target = float(np.dot(target[:2] - foot[:2], direction_xy))
        if forward_target < config.receive_minimum_forward_target_m:
            target[:2] += (config.receive_minimum_forward_target_m - forward_target) * direction_xy
    if distance > config.maximum_foot_ball_distance_m or (
        contact_mode == "strike"
        and not contact_recent
        and longitudinal_offset > config.maximum_forward_foot_offset_m
    ):
        return G1LocomotionContactTeacherEffect(
            zero_torque,
            zero_xyz,
            target,
            distance,
            longitudinal_offset,
            False,
        )

    import mujoco

    jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotation: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(model, data, jacobian, rotation, foot, ankle_body_id)
    foot_velocity = jacobian @ np.asarray(data.qvel, dtype=np.float64)
    force = config.position_gain_n_per_m * (target - foot)
    desired_foot_velocity = (
        ball_velocity
        + np.asarray(
            (
                config.receive_follow_through_speed_mps * direction_xy[0],
                config.receive_follow_through_speed_mps * direction_xy[1],
                0.0,
            ),
            dtype=np.float64,
        )
        if contact_mode == "receive"
        else np.asarray(
            (
                config.strike_foot_speed_mps * direction_xy[0],
                config.strike_foot_speed_mps * direction_xy[1],
                0.0,
            ),
            dtype=np.float64,
        )
    )
    force -= config.velocity_damping_n_per_mps * (foot_velocity - desired_foot_velocity)
    force_norm = float(np.linalg.norm(force))
    if force_norm > config.maximum_task_force_n:
        force *= config.maximum_task_force_n / force_norm
    torque = jacobian[:, dofs].T @ force
    torque = np.clip(
        torque,
        -config.maximum_joint_residual_nm,
        config.maximum_joint_residual_nm,
    )
    return G1LocomotionContactTeacherEffect(
        torque,
        force,
        target,
        distance,
        longitudinal_offset,
        True,
    )


__all__ = [
    "G1LocomotionContactTeacherConfig",
    "G1LocomotionContactTeacherEffect",
    "G1RollingOptionBridgeConfig",
    "locomotion_contact_teacher_effect",
]
