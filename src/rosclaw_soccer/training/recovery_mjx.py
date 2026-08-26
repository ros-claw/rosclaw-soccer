"""Modern MuJoCo/MJX probe for contact-rich G1 recovery learning.

The module deliberately keeps JAX and MJX as optional, training-host-only
dependencies.  Importing :mod:`rosclaw_soccer` therefore never initializes a
CUDA runtime.  The probe binds the compiled G1 physics model, injects real
post-skill snapshots, compares one 50 Hz control step with CPU MuJoCo, and
measures sharded MJX throughput on every requested GPU.

MJX is a rollout and learning backend, not promotion truth.  A candidate must
still pass a separate CPU-MuJoCo exam, and snapshots originating in another
scene are explicitly marked as cross-scene transfer inputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus

_JOINT_COUNT = 29
_QPOS_WIDTH = 36
_QVEL_WIDTH = 35
_FAILURE_STATE_BASE_CONTEXT = [
    "qpos",
    "qvel",
    "trajectory_step",
    "trajectory_initial_step",
    "handoff_frozen",
]
_FAILURE_STATE_POLICY_CONTEXT = [
    *_FAILURE_STATE_BASE_CONTEXT,
    "last_motor_targets",
    "last_teacher_action",
    "last_residual",
    "proprioception_history",
    "phase_repeat",
]

# OpenTrack's public G1 v2 direct-PD contract.  Values are duplicated here so
# the external training checkout cannot silently change runtime authority.
_KPS = np.asarray(
    (
        100,
        100,
        100,
        200,
        80,
        20,
        100,
        100,
        100,
        200,
        80,
        20,
        300,
        300,
        300,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
    ),
    dtype=np.float32,
)
_KDS = np.asarray(
    (
        2,
        2,
        2,
        4,
        2,
        1,
        2,
        2,
        2,
        4,
        2,
        1,
        10,
        10,
        10,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
    ),
    dtype=np.float32,
)
_TORQUE_LIMIT = np.asarray(
    (
        88,
        139,
        88,
        139,
        50,
        50,
        88,
        139,
        88,
        139,
        50,
        50,
        88,
        50,
        50,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class RecoveryMJXProbeConfig:
    """Bounded configuration for an evidence-producing MJX rollout probe."""

    required_gpu_count: int = 4
    batch_size_per_gpu: int = 32
    benchmark_control_steps: int = 24
    parity_snapshot_count: int = 3
    control_dt_sec: float = 0.02
    simulation_dt_sec: float = 0.002
    maximum_qpos_rms_error_rad: float = 0.05
    maximum_qvel_rms_error_rad_s: float = 2.0
    minimum_parallel_control_steps_per_sec: float = 100.0
    snapshot_scene_equivalent: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_probe_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.control_dt_sec,
            self.simulation_dt_sec,
            self.maximum_qpos_rms_error_rad,
            self.maximum_qvel_rms_error_rad_s,
            self.minimum_parallel_control_steps_per_sec,
        )
        ratio = self.control_dt_sec / self.simulation_dt_sec
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in finite)
            or not 1 <= self.required_gpu_count <= 16
            or not 1 <= self.batch_size_per_gpu <= 1_024
            or not 2 <= self.benchmark_control_steps <= 2_000
            or not 1 <= self.parity_snapshot_count <= 64
            or abs(ratio - round(ratio)) > 1e-9
            or self.snapshot_scene_equivalent
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX probe config is invalid")

    @property
    def substeps(self) -> int:
        return int(round(self.control_dt_sec / self.simulation_dt_sec))

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryMJXPPOConfig:
    """Four-GPU residual PPO contract around frozen recovery skill memory."""

    total_timesteps: int = 2_097_152
    num_envs: int = 256
    episode_length: int = 1_800
    unroll_length: int = 32
    batch_size: int = 64
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    num_evals: int = 3
    num_eval_envs: int = 64
    learning_rate: float = 2.0e-4
    entropy_cost: float = 5.0e-4
    discounting: float = 0.997
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.20
    maximum_gradient_norm: float = 1.0
    residual_limit_lower_body_rad: float = 0.15
    residual_limit_waist_rad: float = 0.12
    residual_limit_arm_rad: float = 0.18
    maximum_target_step_rad: float = 0.05
    residual_penalty_scale: float = 0.08
    action_delta_penalty_scale: float = 0.04
    tracking_penalty_scale: float = 0.002
    torque_saturation_penalty_scale: float = 0.15
    joint_position_noise_rad: float = 0.03
    joint_velocity_noise_rad_s: float = 0.08
    root_linear_velocity_noise_mps: float = 0.05
    root_angular_velocity_noise_rad_s: float = 0.08
    success_stable_steps: int = 100
    random_seed: int = 5401
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_ppo_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.entropy_cost,
            self.discounting,
            self.gae_lambda,
            self.clipping_epsilon,
            self.maximum_gradient_norm,
            self.residual_limit_lower_body_rad,
            self.residual_limit_waist_rad,
            self.residual_limit_arm_rad,
            self.maximum_target_step_rad,
            self.residual_penalty_scale,
            self.action_delta_penalty_scale,
            self.tracking_penalty_scale,
            self.torque_saturation_penalty_scale,
            self.joint_position_noise_rad,
            self.joint_velocity_noise_rad_s,
            self.root_linear_velocity_noise_mps,
            self.root_angular_velocity_noise_rad_s,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 65_536 <= self.total_timesteps <= 1_000_000_000
            or not 32 <= self.num_envs <= 65_536
            or self.num_envs % 4
            or not 500 <= self.episode_length <= 3_000
            or not 8 <= self.unroll_length <= 256
            or not 8 <= self.batch_size <= 8_192
            or not 1 <= self.num_minibatches <= 64
            or (self.batch_size * self.num_minibatches) % self.num_envs
            or not 1 <= self.num_updates_per_batch <= 16
            or not 2 <= self.num_evals <= 100
            or not 4 <= self.num_eval_envs <= 1_024
            or self.num_eval_envs % 4
            or not 0.0 < self.learning_rate <= 1e-2
            or not 0.0 < self.entropy_cost <= 0.1
            or not 0.90 <= self.discounting < 1.0
            or not 0.80 <= self.gae_lambda <= 1.0
            or not 0.05 <= self.clipping_epsilon <= 0.40
            or not 0.0 < self.maximum_gradient_norm <= 10.0
            or not 0.01 <= self.residual_limit_lower_body_rad <= 0.20
            or not 0.01 <= self.residual_limit_waist_rad <= 0.15
            or not 0.01 <= self.residual_limit_arm_rad <= 0.25
            or not 0.005 <= self.maximum_target_step_rad <= 0.10
            or not 0.0 < self.residual_penalty_scale <= 1.0
            or not 0.0 < self.action_delta_penalty_scale <= 1.0
            or not 0.0 < self.tracking_penalty_scale <= 0.10
            or not 0.0 < self.torque_saturation_penalty_scale <= 1.0
            or not 0.0 <= self.joint_position_noise_rad <= 0.10
            or not 0.0 <= self.joint_velocity_noise_rad_s <= 0.50
            or not 0.0 <= self.root_linear_velocity_noise_mps <= 0.30
            or not 0.0 <= self.root_angular_velocity_noise_rad_s <= 0.50
            or not 25 <= self.success_stable_steps <= 250
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX PPO config is invalid")

    @property
    def residual_limits_rad(self) -> NDArray[np.float32]:
        return np.asarray(
            (self.residual_limit_lower_body_rad,) * 12
            + (self.residual_limit_waist_rad,) * 3
            + (self.residual_limit_arm_rad,) * 14,
            dtype=np.float32,
        )

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryMJXTeacherResidualPPOConfig:
    """MJX PPO contract for stabilizing a frozen closed-loop teacher.

    The external reference-conditioned teacher is immutable and available only
    during simulation training.  The residual actor receives deployable
    proprioception, and its bounded corrections are later distilled into an
    independent reference-free student before any promotion exam.
    """

    total_timesteps: int = 4_194_304
    num_envs: int = 256
    episode_length: int = 1_200
    unroll_length: int = 32
    batch_size: int = 64
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    num_evals: int = 4
    num_eval_envs: int = 64
    learning_rate: float = 1.0e-4
    entropy_cost: float = 5.0e-5
    discounting: float = 0.997
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.15
    maximum_gradient_norm: float = 0.75
    residual_limit_lower_body_rad: float = 0.08
    residual_limit_waist_rad: float = 0.05
    residual_limit_arm_rad: float = 0.08
    residual_penalty_scale: float = 0.25
    action_delta_penalty_scale: float = 0.12
    teacher_deviation_penalty_scale: float = 0.04
    ready_momentum_penalty_scale: float = 0.15
    directional_momentum_penalty_scale: float = 0.0
    failure_state_directional_penalty_scale: float = 0.0
    failure_state_backward_cost_weight: float = 3.0
    failure_state_lateral_cost_weight: float = 0.25
    failure_state_yaw_cost_weight: float = 0.10
    failure_state_directional_cost_mode: str = "ALWAYS_ON_PSEUDO_HUBER"
    failure_state_stable_streak_reward_scale: float = 0.0
    failure_state_target_horizon_steps: int = 400
    stable_streak_reward_scale: float = 1.0
    handoff_regression_penalty_scale: float = 1.0
    joint_position_noise_rad: float = 0.015
    joint_velocity_noise_rad_s: float = 0.04
    root_linear_velocity_noise_mps: float = 0.025
    root_angular_velocity_noise_rad_s: float = 0.04
    proprioception_history_frames: int = 4
    use_pelvis_imu_observation: bool = True
    pelvis_accelerometer_clip_mps2: float = 40.0
    use_base_velocity_estimate_observation: bool = False
    preserve_pelvis_accelerometer_observation: bool = False
    regularize_velocity_adapter_only: bool = False
    base_velocity_estimate_clip_mps: float = 2.0
    terminal_balance_reset_fraction: float = 0.0
    terminal_balance_root_linear_velocity_noise_mps: float = 0.20
    terminal_balance_root_angular_velocity_noise_rad_s: float = 0.50
    failure_state_reset_fraction: float = 0.0
    terminate_failure_state_episode_at_target_horizon: bool = False
    use_asymmetric_critic: bool = True
    failure_state_conditioned_critic: bool = False
    posture_gated_residual: bool = True
    ready_pelvis_height_m: float = 0.62
    ready_upright_projection: float = 0.75
    handoff_maximum_linear_speed_mps: float = 0.50
    handoff_maximum_angular_speed_rad_s: float = 1.50
    handoff_stable_steps: int = 10
    success_stable_steps: int = 100
    required_gpu_count: int = 4
    random_seed: int = 5411
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16"

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.entropy_cost,
            self.discounting,
            self.gae_lambda,
            self.clipping_epsilon,
            self.maximum_gradient_norm,
            self.residual_limit_lower_body_rad,
            self.residual_limit_waist_rad,
            self.residual_limit_arm_rad,
            self.residual_penalty_scale,
            self.action_delta_penalty_scale,
            self.teacher_deviation_penalty_scale,
            self.ready_momentum_penalty_scale,
            self.directional_momentum_penalty_scale,
            self.failure_state_directional_penalty_scale,
            self.failure_state_backward_cost_weight,
            self.failure_state_lateral_cost_weight,
            self.failure_state_yaw_cost_weight,
            self.failure_state_stable_streak_reward_scale,
            self.stable_streak_reward_scale,
            self.handoff_regression_penalty_scale,
            self.joint_position_noise_rad,
            self.joint_velocity_noise_rad_s,
            self.root_linear_velocity_noise_mps,
            self.root_angular_velocity_noise_rad_s,
            self.pelvis_accelerometer_clip_mps2,
            self.base_velocity_estimate_clip_mps,
            self.terminal_balance_reset_fraction,
            self.terminal_balance_root_linear_velocity_noise_mps,
            self.terminal_balance_root_angular_velocity_noise_rad_s,
            self.failure_state_reset_fraction,
            self.ready_pelvis_height_m,
            self.ready_upright_projection,
            self.handoff_maximum_linear_speed_mps,
            self.handoff_maximum_angular_speed_rad_s,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 65_536 <= self.total_timesteps <= 1_000_000_000
            or not 32 <= self.num_envs <= 65_536
            or not 1 <= self.required_gpu_count <= 16
            or self.num_envs % self.required_gpu_count
            or not 600 <= self.episode_length <= 3_000
            or not 8 <= self.unroll_length <= 256
            or not 8 <= self.batch_size <= 8_192
            or not 1 <= self.num_minibatches <= 64
            or (self.batch_size * self.num_minibatches) % self.num_envs
            or not 1 <= self.num_updates_per_batch <= 16
            or not 2 <= self.num_evals <= 100
            or not 4 <= self.num_eval_envs <= 1_024
            or self.num_eval_envs % self.required_gpu_count
            or not 0.0 < self.learning_rate <= 1e-2
            or not 0.0 <= self.entropy_cost <= 0.1
            or not 0.90 <= self.discounting < 1.0
            or not 0.80 <= self.gae_lambda <= 1.0
            or not 0.05 <= self.clipping_epsilon <= 0.40
            or not 0.0 < self.maximum_gradient_norm <= 10.0
            or not 0.01 <= self.residual_limit_lower_body_rad <= 0.20
            or not 0.01 <= self.residual_limit_waist_rad <= 0.15
            or not 0.01 <= self.residual_limit_arm_rad <= 0.25
            or not 0.0 < self.residual_penalty_scale <= 1.0
            or not 0.0 < self.action_delta_penalty_scale <= 1.0
            or not 0.0 < self.teacher_deviation_penalty_scale <= 1.0
            or not 0.0 < self.ready_momentum_penalty_scale <= 2.0
            or not 0.0 <= self.directional_momentum_penalty_scale <= 2.0
            or not 0.0 <= self.failure_state_directional_penalty_scale <= 2.0
            or not 0.1 <= self.failure_state_backward_cost_weight <= 10.0
            or not 0.0 <= self.failure_state_lateral_cost_weight <= 10.0
            or not 0.0 <= self.failure_state_yaw_cost_weight <= 10.0
            or self.failure_state_directional_cost_mode
            not in {"LEGACY_BALANCE_GATED_CLIPPED_SQUARE", "ALWAYS_ON_PSEUDO_HUBER"}
            or not 0.0 <= self.failure_state_stable_streak_reward_scale <= 5.0
            or not 0.0 < self.stable_streak_reward_scale <= 5.0
            or not 0.0 < self.handoff_regression_penalty_scale <= 5.0
            or not 0.0 <= self.joint_position_noise_rad <= 0.10
            or not 0.0 <= self.joint_velocity_noise_rad_s <= 0.50
            or not 0.0 <= self.root_linear_velocity_noise_mps <= 0.30
            or not 0.0 <= self.root_angular_velocity_noise_rad_s <= 0.50
            or isinstance(self.proprioception_history_frames, bool)
            or not 1 <= self.proprioception_history_frames <= 16
            or not isinstance(self.use_pelvis_imu_observation, bool)
            or not 10.0 <= self.pelvis_accelerometer_clip_mps2 <= 100.0
            or not isinstance(self.use_base_velocity_estimate_observation, bool)
            or not isinstance(self.preserve_pelvis_accelerometer_observation, bool)
            or not isinstance(self.regularize_velocity_adapter_only, bool)
            or not 0.5 <= self.base_velocity_estimate_clip_mps <= 5.0
            or (self.use_base_velocity_estimate_observation and not self.use_pelvis_imu_observation)
            or (
                self.preserve_pelvis_accelerometer_observation
                and not self.use_base_velocity_estimate_observation
            )
            or (
                self.regularize_velocity_adapter_only
                and not self.preserve_pelvis_accelerometer_observation
            )
            or not 0.0 <= self.terminal_balance_reset_fraction <= 0.90
            or not 0.05 <= self.terminal_balance_root_linear_velocity_noise_mps <= 0.50
            or not 0.10 <= self.terminal_balance_root_angular_velocity_noise_rad_s <= 1.50
            or not 0.0 <= self.failure_state_reset_fraction <= 0.90
            or not isinstance(self.terminate_failure_state_episode_at_target_horizon, bool)
            or (
                self.terminate_failure_state_episode_at_target_horizon
                and self.failure_state_reset_fraction <= 0.0
            )
            or (
                self.schema_version
                in {
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
                }
                and self.terminate_failure_state_episode_at_target_horizon
                and self.total_timesteps
                < 2 * self.num_envs * self.failure_state_target_horizon_steps
            )
            or isinstance(self.failure_state_target_horizon_steps, bool)
            or not 100 <= self.failure_state_target_horizon_steps <= self.episode_length
            or (
                self.terminal_balance_reset_fraction > 0.0
                and self.failure_state_reset_fraction > 0.0
            )
            or not isinstance(self.use_asymmetric_critic, bool)
            or (self.preserve_pelvis_accelerometer_observation and not self.use_asymmetric_critic)
            or not isinstance(self.failure_state_conditioned_critic, bool)
            or (
                (
                    self.failure_state_directional_penalty_scale > 0.0
                    or self.failure_state_stable_streak_reward_scale > 0.0
                )
                and not self.failure_state_conditioned_critic
            )
            or (
                self.failure_state_conditioned_critic
                and (self.failure_state_reset_fraction <= 0.0 or not self.use_asymmetric_critic)
            )
            or not isinstance(self.posture_gated_residual, bool)
            or not 0.50 <= self.ready_pelvis_height_m <= 0.90
            or not 0.50 <= self.ready_upright_projection <= 1.0
            or not 0.05 <= self.handoff_maximum_linear_speed_mps <= 1.0
            or not 0.10 <= self.handoff_maximum_angular_speed_rad_s <= 3.0
            or not 5 <= self.handoff_stable_steps <= 50
            or not 50 <= self.success_stable_steps <= 250
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX teacher-residual PPO config is invalid")

    @property
    def residual_limits_rad(self) -> NDArray[np.float32]:
        return np.asarray(
            (self.residual_limit_lower_body_rad,) * 12
            + (self.residual_limit_waist_rad,) * 3
            + (self.residual_limit_arm_rad,) * 14,
            dtype=np.float32,
        )

    @property
    def expected_failure_target_transition_fraction(self) -> float:
        """Expected targeted-transition share under Brax fixed-state auto-reset."""

        reset_fraction = self.failure_state_reset_fraction
        if reset_fraction <= 0.0:
            return 0.0
        horizon = float(self.failure_state_target_horizon_steps)
        episode = float(self.episode_length)
        if not self.terminate_failure_state_episode_at_target_horizon:
            return reset_fraction * horizon / episode
        if self.schema_version in {
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
        }:
            # Brax AutoReset restores each parallel environment's initial
            # state; it does not resample the reset source at episode borders.
            # A failure-bank environment terminated at the target horizon is
            # therefore targeted on every subsequent transition as well.
            return reset_fraction
        return (reset_fraction * horizon) / (
            reset_fraction * horizon + (1.0 - reset_fraction) * episode
        )

    @property
    def actor_proprioception_frame_dim(self) -> int:
        """Return the deployable frame width bound by the observation contract."""

        if not self.use_pelvis_imu_observation:
            return 93
        return 99 if self.preserve_pelvis_accelerometer_observation else 96

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryMJXFailureConstraintConfig:
    """Projected-dual update contract for exact failure-state retention.

    The update happens between immutable PPO generations.  It does not grant
    an exam authority to the GPU learner: every persisted checkpoint still
    needs a paired failure-state report, and a candidate is retainable only
    when the ordinary route gate and the local failure gate both pass.
    """

    dual_learning_rate: float = 0.5
    maximum_normalized_update: float = 2.0
    minimum_backward_multiplier: float = 0.1
    minimum_lateral_multiplier: float = 0.0
    minimum_yaw_multiplier: float = 0.0
    maximum_directional_multiplier: float = 10.0
    minimum_stable_streak_multiplier: float = 0.0
    maximum_stable_streak_multiplier: float = 5.0
    require_all_persisted_checkpoint_exams: bool = True
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_failure_constraint_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.dual_learning_rate,
            self.maximum_normalized_update,
            self.minimum_backward_multiplier,
            self.minimum_lateral_multiplier,
            self.minimum_yaw_multiplier,
            self.maximum_directional_multiplier,
            self.minimum_stable_streak_multiplier,
            self.maximum_stable_streak_multiplier,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 0.0 < self.dual_learning_rate <= 5.0
            or not 0.1 <= self.maximum_normalized_update <= 10.0
            or not 0.1 <= self.minimum_backward_multiplier <= 10.0
            or not 0.0 <= self.minimum_lateral_multiplier <= 10.0
            or not 0.0 <= self.minimum_yaw_multiplier <= 10.0
            or not max(
                self.minimum_backward_multiplier,
                self.minimum_lateral_multiplier,
                self.minimum_yaw_multiplier,
            )
            <= self.maximum_directional_multiplier
            <= 10.0
            or not 0.0
            <= self.minimum_stable_streak_multiplier
            <= self.maximum_stable_streak_multiplier
            <= 5.0
            or not isinstance(self.require_all_persisted_checkpoint_exams, bool)
            or not self.require_all_persisted_checkpoint_exams
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX failure constraint config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return str(hash_bytes(header + b"\0" + array.tobytes()))


def compiled_mujoco_model_contract(model: Any) -> dict[str, Any]:
    """Bind the numerical dynamics fields relevant to the direct-PD probe."""

    arrays = {
        name: _array_hash(getattr(model, name))
        for name in (
            "body_mass",
            "body_inertia",
            "dof_armature",
            "dof_damping",
            "geom_friction",
            "geom_size",
            "geom_type",
            "jnt_range",
            "jnt_type",
            "actuator_ctrlrange",
            "actuator_gear",
        )
    }
    payload: dict[str, Any] = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "timestep_sec": float(model.opt.timestep),
        "gravity_mps2": np.asarray(model.opt.gravity, dtype=np.float64).tolist(),
        "integrator": int(model.opt.integrator),
        "cone": int(model.opt.cone),
        "solver": int(model.opt.solver),
        "iterations": int(model.opt.iterations),
        "arrays": arrays,
        "pd_gain_hash": _array_hash(np.concatenate((_KPS, _KDS, _TORQUE_LIMIT))),
        "schema_version": "rosclaw_soccer.compiled_mujoco_model_contract.v1",
    }
    payload["model_hash"] = hash_json(payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_recovery_mjx_probe_report(path: Path) -> dict[str, Any]:
    """Load a probe report and fail closed on integrity or authority drift."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX probe report must be an object")
    declared = payload.pop("report_hash", None)
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_mjx_probe_report.v1"
        or declared != hash_json(payload)
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("physics_truth_backend") != "CPU_MUJOCO"
        or payload.get("rollout_backend") != "MUJOCO_MJX"
        or payload.get("snapshot_scene_equivalent") is not False
    ):
        raise ValueError("recovery MJX probe report is invalid")
    payload["report_hash"] = declared
    return payload


def validate_recovery_mjx_teacher_residual_report(path: Path) -> dict[str, Any]:
    """Fail closed when a teacher-residual training report exceeds its role."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX teacher-residual report must be an object")
    declared = payload.pop("report_hash", None)
    config = payload.get("config")
    devices = payload.get("devices")
    continued = payload.get("continued_from_parent", False)
    parent_hash = payload.get("parent_checkpoint_hash")
    parent_training_report_hash = payload.get("parent_training_report_hash")
    route_manifest_hash = payload.get("route_manifest_hash")
    route_group_hash = payload.get("route_group_hash")
    route_binding_enforced = payload.get("route_binding_enforced", False)
    actor_observation = payload.get("actor_observation")
    history_frames = (
        config.get("proprioception_history_frames", 1) if isinstance(config, dict) else 0
    )
    actor_observation_dim = payload.get("actor_observation_dim")
    pelvis_imu_enabled = (
        config.get("use_pelvis_imu_observation", False) if isinstance(config, dict) else None
    )
    pelvis_accelerometer_clip = (
        config.get("pelvis_accelerometer_clip_mps2") if isinstance(config, dict) else None
    )
    base_velocity_estimate_enabled = (
        config.get("use_base_velocity_estimate_observation", False)
        if isinstance(config, dict)
        else None
    )
    base_velocity_estimate_clip = (
        config.get("base_velocity_estimate_clip_mps", 2.0) if isinstance(config, dict) else None
    )
    preserve_accelerometer = (
        config.get("preserve_pelvis_accelerometer_observation", False)
        if isinstance(config, dict)
        else None
    )
    expected_frame_dim = (
        99
        if pelvis_imu_enabled is True and preserve_accelerometer is True
        else 96
        if pelvis_imu_enabled is True
        else 93
    )
    declared_frame_dim = payload.get("actor_proprioception_frame_dim")
    legacy_actor_observation_valid = (
        pelvis_imu_enabled is False
        and (
            (
                actor_observation == "DEPLOYABLE_PROPRIOCEPTION_ONLY"
                and history_frames == 1
                and actor_observation_dim in (None, 93)
            )
            or (
                actor_observation == "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY"
                and isinstance(history_frames, int)
                and 2 <= history_frames <= 16
                and actor_observation_dim == 93 * history_frames
            )
        )
        and declared_frame_dim in (None, 93)
    )
    pelvis_imu_contract = payload.get("actor_pelvis_imu_contract")
    pelvis_imu_actor_observation_valid = bool(
        pelvis_imu_enabled is True
        and base_velocity_estimate_enabled is False
        and isinstance(history_frames, int)
        and 1 <= history_frames <= 16
        and actor_observation
        == (
            "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_HISTORY_ONLY"
            if history_frames > 1
            else "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_ONLY"
        )
        and actor_observation_dim == expected_frame_dim * history_frames
        and declared_frame_dim == expected_frame_dim
        and pelvis_imu_contract
        == {
            "accelerometer_clip_mps2": pelvis_accelerometer_clip,
            "accelerometer_sensor": "accelerometer_pelvis",
            "accelerometer_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
            "gyroscope_scale": 0.05,
            "gyroscope_sensor": "gyro_pelvis",
        }
    )
    base_velocity_estimator_contract = payload.get("actor_base_velocity_estimator_contract")
    base_velocity_actor_observation_valid = bool(
        pelvis_imu_enabled is True
        and base_velocity_estimate_enabled is True
        and preserve_accelerometer is False
        and isinstance(base_velocity_estimate_clip, (int, float))
        and not isinstance(base_velocity_estimate_clip, bool)
        and 0.5 <= float(base_velocity_estimate_clip) <= 5.0
        and isinstance(history_frames, int)
        and 1 <= history_frames <= 16
        and actor_observation
        == (
            "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
            if history_frames > 1
            else "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_ONLY"
        )
        and actor_observation_dim == expected_frame_dim * history_frames
        and declared_frame_dim == expected_frame_dim
        and pelvis_imu_contract
        == {
            "linear_motion_feature": "BASE_VELOCITY_ESTIMATE",
            "linear_motion_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
            "gyroscope_scale": 0.05,
            "gyroscope_sensor": "gyro_pelvis",
        }
        and base_velocity_estimator_contract
        == {
            "clip_mps": base_velocity_estimate_clip,
            "deployment_source_required": "ONBOARD_STATE_ESTIMATOR",
            "ground_truth_hardware_velocity_authorized": False,
            "simulation_proxy": "MUJOCO_ROOT_QVEL_ROTATED_TO_PELVIS",
        }
    )
    appended_velocity_actor_observation_valid = bool(
        pelvis_imu_enabled is True
        and base_velocity_estimate_enabled is True
        and preserve_accelerometer is True
        and isinstance(base_velocity_estimate_clip, (int, float))
        and not isinstance(base_velocity_estimate_clip, bool)
        and 0.5 <= float(base_velocity_estimate_clip) <= 5.0
        and isinstance(history_frames, int)
        and 1 <= history_frames <= 16
        and actor_observation
        == (
            "DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
            if history_frames > 1
            else ("DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_ONLY")
        )
        and actor_observation_dim == expected_frame_dim * history_frames
        and declared_frame_dim == expected_frame_dim
        and pelvis_imu_contract
        == {
            "accelerometer_clip_mps2": pelvis_accelerometer_clip,
            "accelerometer_sensor": "accelerometer_pelvis",
            "accelerometer_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
            "linear_motion_feature": "BASE_VELOCITY_ESTIMATE",
            "linear_motion_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
            "gyroscope_scale": 0.05,
            "gyroscope_sensor": "gyro_pelvis",
        }
        and base_velocity_estimator_contract
        == {
            "clip_mps": base_velocity_estimate_clip,
            "deployment_source_required": "ONBOARD_STATE_ESTIMATOR",
            "ground_truth_hardware_velocity_authorized": False,
            "simulation_proxy": "MUJOCO_ROOT_QVEL_ROTATED_TO_PELVIS",
        }
    )
    actor_observation_valid = bool(
        (
            (legacy_actor_observation_valid or pelvis_imu_actor_observation_valid)
            and base_velocity_estimator_contract is None
        )
        or base_velocity_actor_observation_valid
        or appended_velocity_actor_observation_valid
    )
    config_schema = config.get("schema_version") if isinstance(config, dict) else None
    observation_migration = payload.get("actor_observation_migration")
    expected_source_dim = 96 * history_frames if isinstance(history_frames, int) else -1
    migration_error_fields_valid = bool(
        isinstance(observation_migration, dict)
        and all(
            isinstance(observation_migration.get(name), (int, float))
            and not isinstance(observation_migration.get(name), bool)
            and math.isfinite(float(observation_migration[name]))
            and 0.0 <= float(observation_migration[name]) <= 1.0e-6
            for name in ("policy_output_max_abs_error", "value_output_max_abs_error")
        )
    )
    migration_normalizer_fields_valid = bool(
        config_schema
        not in {
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
            "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
        }
        or (
            isinstance(observation_migration, dict)
            and observation_migration.get("parent_observation_normalizer_frozen") is True
            and isinstance(
                observation_migration.get("parent_normalizer_drift_invariance_max_abs_error"),
                (int, float),
            )
            and not isinstance(
                observation_migration.get("parent_normalizer_drift_invariance_max_abs_error"),
                bool,
            )
            and math.isfinite(
                float(observation_migration["parent_normalizer_drift_invariance_max_abs_error"])
            )
            and 0.0
            <= float(observation_migration["parent_normalizer_drift_invariance_max_abs_error"])
            <= 1.0e-6
        )
    )
    migration_gradient_fields_valid = bool(
        isinstance(observation_migration, dict)
        and isinstance(observation_migration.get("parent_policy_gradient_max_abs"), (int, float))
        and not isinstance(observation_migration.get("parent_policy_gradient_max_abs"), bool)
        and observation_migration.get("parent_policy_gradient_max_abs") == 0.0
        and isinstance(observation_migration.get("adapter_output_gradient_l2"), (int, float))
        and not isinstance(observation_migration.get("adapter_output_gradient_l2"), bool)
        and math.isfinite(float(observation_migration["adapter_output_gradient_l2"]))
        and float(observation_migration["adapter_output_gradient_l2"]) > 0.0
    )
    observation_migration_valid = bool(
        (preserve_accelerometer is not True and observation_migration is None)
        or (
            preserve_accelerometer is True
            and continued is True
            and migration_error_fields_valid
            and migration_gradient_fields_valid
            and migration_normalizer_fields_valid
            and isinstance(observation_migration, dict)
            and set(observation_migration)
            == (
                {
                    "strategy",
                    "source_actor_observation_dim",
                    "target_actor_observation_dim",
                    "source_frame_dim",
                    "target_frame_dim",
                    "appended_feature",
                    "legacy_accelerometer_slots_preserved",
                    "parent_policy_trunk_frozen",
                    "new_adapter_output_weights_initialized_to_zero",
                    "adapter_location_limit",
                    "parent_policy_gradient_max_abs",
                    "adapter_output_gradient_l2",
                    "new_value_input_weights_initialized_to_zero",
                    "policy_output_max_abs_error",
                    "value_output_max_abs_error",
                    "behavior_preserved",
                }
                | (
                    {
                        "parent_observation_normalizer_frozen",
                        "parent_normalizer_drift_invariance_max_abs_error",
                    }
                    if config_schema
                    in {
                        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
                        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
                        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
                        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
                    }
                    else set()
                )
            )
            and observation_migration.get("strategy")
            == "FROZEN_PARENT_TRUNK_ZERO_OUTPUT_VELOCITY_ADAPTER"
            and observation_migration.get("source_actor_observation_dim") == expected_source_dim
            and observation_migration.get("target_actor_observation_dim") == actor_observation_dim
            and observation_migration.get("source_frame_dim") == 96
            and observation_migration.get("target_frame_dim") == 99
            and observation_migration.get("appended_feature") == "ONBOARD_BASE_VELOCITY_ESTIMATE"
            and observation_migration.get("legacy_accelerometer_slots_preserved") is True
            and observation_migration.get("parent_policy_trunk_frozen") is True
            and observation_migration.get("new_adapter_output_weights_initialized_to_zero") is True
            and observation_migration.get("adapter_location_limit") == 0.25
            and observation_migration.get("new_value_input_weights_initialized_to_zero") is True
            and observation_migration.get("behavior_preserved") is True
        )
    )
    parent_actor_retention = payload.get("parent_actor_retention")
    retention_results = (
        parent_actor_retention.get("checkpoint_results")
        if isinstance(parent_actor_retention, dict)
        else None
    )
    checkpoint_file_steps = {
        int(str(row["path"]).split("/", maxsplit=1)[0])
        for row in payload.get("candidate_checkpoint_files", [])
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and str(row["path"]).split("/", maxsplit=1)[0].isdigit()
    }
    parent_actor_retention_valid = bool(
        (
            config_schema
            not in {
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
            }
            and parent_actor_retention is None
        )
        or (
            config_schema
            in {
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
            }
            and preserve_accelerometer is not True
            and parent_actor_retention is None
        )
        or (
            config_schema
            in {
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
            }
            and preserve_accelerometer is True
            and isinstance(parent_actor_retention, dict)
            and set(parent_actor_retention)
            == {
                "schema_version",
                "source_frozen_state_hash",
                "frozen_components",
                "checkpoint_results",
                "all_checkpoints_exact",
            }
            and parent_actor_retention.get("schema_version")
            == "rosclaw_soccer.frozen_parent_actor_retention.v1"
            and isinstance(parent_actor_retention.get("source_frozen_state_hash"), str)
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(parent_actor_retention["source_frozen_state_hash"]),
            )
            and parent_actor_retention.get("frozen_components")
            == [
                "DENSE_TRUNK",
                "ACTION_LOCATION_HEAD",
                "ACTION_SCALE_HEAD",
                "OBSERVATION_NORMALIZER_MEAN",
                "OBSERVATION_NORMALIZER_STD",
            ]
            and isinstance(retention_results, list)
            and bool(retention_results)
            and {row.get("step") for row in retention_results if isinstance(row, dict)}
            == checkpoint_file_steps
            and all(
                isinstance(row, dict)
                and set(row)
                == {
                    "step",
                    "frozen_state_hash",
                    "exact_equal",
                    "maximum_absolute_error",
                }
                and isinstance(row.get("step"), int)
                and not isinstance(row.get("step"), bool)
                and int(row["step"]) >= 0
                and isinstance(row.get("frozen_state_hash"), str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["frozen_state_hash"]))
                and row.get("frozen_state_hash")
                == parent_actor_retention.get("source_frozen_state_hash")
                and row.get("exact_equal") is True
                and row.get("maximum_absolute_error") == 0.0
                for row in retention_results
            )
            and parent_actor_retention.get("all_checkpoints_exact") is True
        )
    )
    regularize_adapter_only = (
        config.get("regularize_velocity_adapter_only", False) if isinstance(config, dict) else None
    )
    residual_regularization_target_valid = bool(
        (
            config_schema
            not in {
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
            }
            and payload.get("residual_regularization_target") is None
        )
        or (
            config_schema
            in {
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
            }
            and isinstance(regularize_adapter_only, bool)
            and (not regularize_adapter_only or preserve_accelerometer is True)
            and payload.get("residual_regularization_target")
            == (
                "VELOCITY_ADAPTER_INCREMENT_ONLY"
                if regularize_adapter_only
                else "TOTAL_TEACHER_RESIDUAL"
            )
        )
    )
    asymmetric_critic = (
        config.get("use_asymmetric_critic", False) if isinstance(config, dict) else None
    )
    failure_conditioned_critic = (
        config.get("failure_state_conditioned_critic", False) if isinstance(config, dict) else None
    )
    legacy_critic_features = [
        "root_linear_velocity",
        "root_angular_velocity",
        "pelvis_height",
        "upright",
    ]
    failure_conditioned_critic_features = [
        "root_body_linear_velocity",
        "pelvis_angular_velocity",
        "pelvis_height",
        "failure_state_reset_source",
    ]
    critic_features = payload.get("critic_privileged_features")
    critic_observation_valid = (
        asymmetric_critic is False and payload.get("critic_observation") in (None, "SAME_AS_ACTOR")
    ) or (
        asymmetric_critic is True
        and payload.get("critic_observation") == "SIMULATION_PRIVILEGED_VALUE_FUNCTION_ONLY"
        and payload.get("critic_privileged_auxiliary_dim") == 8
        and (
            (
                failure_conditioned_critic is False
                and critic_features in (None, legacy_critic_features)
            )
            or (
                failure_conditioned_critic is True
                and critic_features == failure_conditioned_critic_features
            )
        )
        and payload.get("critic_exported_with_actor") is False
    )
    targeted_directional_scale = (
        config.get("failure_state_directional_penalty_scale", 0.0)
        if isinstance(config, dict)
        else None
    )
    targeted_streak_scale = (
        config.get("failure_state_stable_streak_reward_scale", 0.0)
        if isinstance(config, dict)
        else None
    )
    targeted_horizon_steps = (
        config.get("failure_state_target_horizon_steps", 400) if isinstance(config, dict) else None
    )
    targeted_scales_valid = bool(
        isinstance(targeted_directional_scale, (int, float))
        and not isinstance(targeted_directional_scale, bool)
        and math.isfinite(float(targeted_directional_scale))
        and 0.0 <= float(targeted_directional_scale) <= 2.0
        and isinstance(targeted_streak_scale, (int, float))
        and not isinstance(targeted_streak_scale, bool)
        and math.isfinite(float(targeted_streak_scale))
        and 0.0 <= float(targeted_streak_scale) <= 5.0
        and isinstance(targeted_horizon_steps, int)
        and not isinstance(targeted_horizon_steps, bool)
        and 100
        <= targeted_horizon_steps
        <= int(cast(dict[str, Any], config).get("episode_length", 3_000))
        and isinstance(failure_conditioned_critic, bool)
    )
    targeted_cost_weights = (
        {
            "backward": config.get("failure_state_backward_cost_weight", 3.0),
            "lateral": config.get("failure_state_lateral_cost_weight", 0.25),
            "yaw": config.get("failure_state_yaw_cost_weight", 0.10),
        }
        if isinstance(config, dict)
        else {}
    )
    targeted_cost_weights_valid = bool(
        set(targeted_cost_weights) == {"backward", "lateral", "yaw"}
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in targeted_cost_weights.values()
        )
        and 0.1 <= float(targeted_cost_weights["backward"]) <= 10.0
        and 0.0 <= float(targeted_cost_weights["lateral"]) <= 10.0
        and 0.0 <= float(targeted_cost_weights["yaw"]) <= 10.0
    )
    directional_cost_mode = (
        config.get("failure_state_directional_cost_mode") if isinstance(config, dict) else None
    )
    directional_cost_mode_valid = bool(
        config_schema != "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16"
        or directional_cost_mode
        in {"LEGACY_BALANCE_GATED_CLIPPED_SQUARE", "ALWAYS_ON_PSEUDO_HUBER"}
    )
    targeted_reward = payload.get("failure_state_targeted_reward")
    expected_targeted_reward: dict[str, Any] = {
        "actor_observation_features": [],
        "critic_failure_source_indicator": failure_conditioned_critic,
        "directional_penalty_scale": targeted_directional_scale,
        "evaluation_active": False,
        "stable_streak_reward_scale": targeted_streak_scale,
        "training_scope": "FAILURE_STATE_RESET_EPISODES_ONLY",
    }
    if config_schema in {
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v10",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v11",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
    }:
        expected_targeted_reward.update(
            directional_cost_weights=targeted_cost_weights,
            target_horizon_steps=targeted_horizon_steps,
        )
    if config_schema == "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16":
        expected_targeted_reward.update(directional_cost_mode=directional_cost_mode)
    terminate_failure_horizon = (
        config.get("terminate_failure_state_episode_at_target_horizon", False)
        if isinstance(config, dict)
        else None
    )
    horizon_boundary_schema = config_schema in {
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
        "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
    }
    if horizon_boundary_schema:
        expected_targeted_reward.update(
            failure_episode_boundary=(
                "TARGET_HORIZON"
                if terminate_failure_horizon is True
                else "OUTER_EPISODE_OR_SUCCESS"
            )
        )
    targeted_reward_contract_valid = bool(
        targeted_scales_valid
        and targeted_cost_weights_valid
        and directional_cost_mode_valid
        and isinstance(terminate_failure_horizon, bool)
        and (
            targeted_reward == expected_targeted_reward
            or (
                targeted_reward is None
                and targeted_directional_scale == 0.0
                and targeted_streak_scale == 0.0
                and failure_conditioned_critic is False
            )
        )
    )
    signed_velocity_diagnostics = payload.get("signed_velocity_diagnostics")
    legacy_signed_velocity_diagnostics = {
        "actor_observation_features": [],
        "angular_frame": "PELVIS_IMU_SENSOR_FRAME",
        "linear_frame": "PELVIS_BODY_FRAME",
        "metrics": [
            "root_body_forward_velocity",
            "root_body_lateral_velocity",
            "root_body_vertical_velocity",
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_roll_rate",
            "pelvis_pitch_rate",
            "pelvis_yaw_rate",
        ],
        "use": "EVALUATION_AND_CURRICULUM_DIAGNOSTICS_ONLY",
    }
    temporal_signed_velocity_diagnostics = {
        "actor_observation_features": [],
        "angular_frame": "PELVIS_IMU_SENSOR_FRAME",
        "linear_frame": "PELVIS_BODY_FRAME",
        "metrics": [
            "root_body_forward_velocity",
            "root_body_lateral_velocity",
            "root_body_vertical_velocity",
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_roll_rate",
            "pelvis_pitch_rate",
            "pelvis_yaw_rate",
            "pelvis_yaw_speed",
        ],
        "temporal_bin_count": 6,
        "temporal_bin_semantics": "EQUAL_WIDTH_BY_WRAPPER_CONTROL_STEP",
        "temporal_metrics": [
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_yaw_speed",
        ],
        "use": "EVALUATION_AND_CURRICULUM_DIAGNOSTICS_ONLY",
    }
    signed_velocity_diagnostics_valid = signed_velocity_diagnostics in (
        None,
        legacy_signed_velocity_diagnostics,
        temporal_signed_velocity_diagnostics,
    )
    terminal_balance_fraction = (
        config.get("terminal_balance_reset_fraction", 0.0) if isinstance(config, dict) else None
    )
    terminal_balance_curriculum = payload.get("terminal_balance_curriculum")
    terminal_balance_curriculum_valid = terminal_balance_curriculum is None and (
        terminal_balance_fraction == 0.0
    )
    if isinstance(terminal_balance_curriculum, dict) and isinstance(config, dict):
        terminal_reference_frame = terminal_balance_curriculum.get("reference_frame")
        directional_manifest_hash = terminal_balance_curriculum.get(
            "directional_curriculum_manifest_hash"
        )
        directional_report_hash = terminal_balance_curriculum.get(
            "directional_curriculum_report_hash"
        )
        directional_linear_bias = terminal_balance_curriculum.get(
            "terminal_body_linear_velocity_bias_mps"
        )
        directional_yaw_bias = terminal_balance_curriculum.get(
            "terminal_pelvis_yaw_rate_bias_rad_s"
        )
        directional_fields_present = any(
            key in terminal_balance_curriculum
            for key in (
                "directional_curriculum_manifest_hash",
                "directional_curriculum_report_hash",
                "terminal_body_linear_velocity_bias_mps",
                "terminal_pelvis_yaw_rate_bias_rad_s",
            )
        )
        directional_curriculum_valid = not directional_fields_present or bool(
            (
                directional_manifest_hash is None
                and directional_report_hash is None
                and directional_linear_bias == [0.0, 0.0, 0.0]
                and directional_yaw_bias == 0.0
            )
            or (
                isinstance(directional_manifest_hash, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", directional_manifest_hash)
                and isinstance(directional_report_hash, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", directional_report_hash)
                and isinstance(directional_linear_bias, list)
                and len(directional_linear_bias) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in directional_linear_bias
                )
                and -0.30 <= float(directional_linear_bias[0]) <= 0.0
                and -0.20 <= float(directional_linear_bias[1]) <= 0.20
                and float(directional_linear_bias[2]) == 0.0
                and isinstance(directional_yaw_bias, (int, float))
                and not isinstance(directional_yaw_bias, bool)
                and math.isfinite(float(directional_yaw_bias))
                and -1.0 <= float(directional_yaw_bias) <= 1.0
                and isinstance(terminal_balance_fraction, (int, float))
                and not isinstance(terminal_balance_fraction, bool)
                and float(terminal_balance_fraction) > 0.0
            )
        )
        terminal_balance_curriculum_valid = bool(
            isinstance(terminal_balance_fraction, (int, float))
            and not isinstance(terminal_balance_fraction, bool)
            and 0.0 <= float(terminal_balance_fraction) <= 0.90
            and terminal_balance_curriculum.get("training_reset_fraction")
            == terminal_balance_fraction
            and terminal_balance_curriculum.get("evaluation_reset_fraction") == 0.0
            and terminal_balance_curriculum.get("root_linear_velocity_noise_mps")
            == config.get("terminal_balance_root_linear_velocity_noise_mps")
            and terminal_balance_curriculum.get("root_angular_velocity_noise_rad_s")
            == config.get("terminal_balance_root_angular_velocity_noise_rad_s")
            and directional_curriculum_valid
            and (
                float(terminal_balance_fraction) == 0.0
                or (
                    isinstance(terminal_reference_frame, int)
                    and not isinstance(terminal_reference_frame, bool)
                    and terminal_reference_frame >= 0
                )
            )
        )
    failure_state_fraction = (
        config.get("failure_state_reset_fraction", 0.0) if isinstance(config, dict) else None
    )
    expected_targeted_transition_fraction: float | None = None
    expected_completed_target_horizon_cycles: int | None = None
    if horizon_boundary_schema and isinstance(config, dict):
        reset_fraction_value = float(config.get("failure_state_reset_fraction", 0.0))
        horizon_value = float(config.get("failure_state_target_horizon_steps", 400))
        episode_value = float(config.get("episode_length", 1_200))
        if terminate_failure_horizon is True and reset_fraction_value > 0.0:
            expected_targeted_transition_fraction = (
                reset_fraction_value
                if config_schema
                in {
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
                }
                else (reset_fraction_value * horizon_value)
                / (
                    reset_fraction_value * horizon_value
                    + (1.0 - reset_fraction_value) * episode_value
                )
            )
        else:
            expected_targeted_transition_fraction = (
                reset_fraction_value * horizon_value / episode_value
            )
        total_timesteps_value = config.get("total_timesteps")
        num_envs_value = config.get("num_envs")
        horizon_steps_value = config.get("failure_state_target_horizon_steps")
        if (
            isinstance(total_timesteps_value, int)
            and not isinstance(total_timesteps_value, bool)
            and isinstance(num_envs_value, int)
            and not isinstance(num_envs_value, bool)
            and isinstance(horizon_steps_value, int)
            and not isinstance(horizon_steps_value, bool)
            and total_timesteps_value >= 0
            and num_envs_value > 0
            and horizon_steps_value > 0
        ):
            expected_completed_target_horizon_cycles = total_timesteps_value // (
                num_envs_value * horizon_steps_value
            )
    failure_state_curriculum = payload.get("failure_state_curriculum")
    failure_state_curriculum_valid = failure_state_curriculum is None and (
        failure_state_fraction == 0.0
    )
    if isinstance(failure_state_curriculum, dict):
        manifest_hash = failure_state_curriculum.get("failure_state_manifest_hash")
        manifest_file_hash = failure_state_curriculum.get("failure_state_manifest_file_hash")
        archive_hash = failure_state_curriculum.get("failure_state_archive_hash")
        failure_count = failure_state_curriculum.get("failure_state_count")
        inactive = bool(
            failure_state_fraction == 0.0
            and manifest_hash is None
            and manifest_file_hash is None
            and archive_hash is None
            and failure_count == 0
        )
        active_failure_bank = bool(
            isinstance(failure_state_fraction, (int, float))
            and not isinstance(failure_state_fraction, bool)
            and 0.0 < float(failure_state_fraction) <= (0.90 if horizon_boundary_schema else 0.50)
            and all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in (manifest_hash, manifest_file_hash, archive_hash)
            )
            and isinstance(failure_count, int)
            and not isinstance(failure_count, bool)
            and failure_count > 0
        )
        expected_context_adapter = (
            ("BASE_VELOCITY_ESTIMATE_APPENDED_AFTER_PRESERVED_ACCELEROMETER_CHANNELS_6_TO_8")
            if preserve_accelerometer is True
            else "BASE_VELOCITY_ESTIMATE_REPLACES_ACCELEROMETER_CHANNELS_6_TO_8"
            if base_velocity_estimate_enabled is True
            else None
        )
        failure_state_curriculum_valid = bool(
            failure_state_curriculum.get("evaluation_reset_fraction") == 0.0
            and failure_state_curriculum.get("training_reset_fraction") == failure_state_fraction
            and failure_state_curriculum.get("context_features_restored")
            in (_FAILURE_STATE_BASE_CONTEXT, _FAILURE_STATE_POLICY_CONTEXT)
            and failure_state_curriculum.get("observation_context_adapter")
            in (None, expected_context_adapter)
            and (
                base_velocity_estimate_enabled is not True
                or failure_state_curriculum.get("observation_context_adapter")
                == expected_context_adapter
            )
            and (inactive or active_failure_bank)
            and (
                not horizon_boundary_schema
                or failure_state_curriculum.get("training_episode_boundary")
                == (
                    "TARGET_HORIZON"
                    if terminate_failure_horizon is True
                    else "OUTER_EPISODE_OR_SUCCESS"
                )
            )
            and (
                not horizon_boundary_schema
                or (
                    isinstance(
                        failure_state_curriculum.get(
                            "expected_targeted_training_transition_fraction"
                        ),
                        (int, float),
                    )
                    and not isinstance(
                        failure_state_curriculum.get(
                            "expected_targeted_training_transition_fraction"
                        ),
                        bool,
                    )
                    and expected_targeted_transition_fraction is not None
                    and math.isclose(
                        float(
                            failure_state_curriculum[
                                "expected_targeted_training_transition_fraction"
                            ]
                        ),
                        expected_targeted_transition_fraction,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
            )
            and (
                config_schema
                not in {
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
                }
                or (
                    failure_state_curriculum.get("reset_source_resampling")
                    == "FIXED_PER_PARALLEL_ENVIRONMENT_AUTORESET"
                    and failure_state_curriculum.get("minimum_completed_target_horizon_cycles")
                    == expected_completed_target_horizon_cycles
                    and expected_completed_target_horizon_cycles is not None
                    and expected_completed_target_horizon_cycles >= 2
                )
            )
        )
    targeted_reward_enabled = bool(
        targeted_scales_valid
        and (
            float(cast(int | float, targeted_directional_scale)) > 0.0
            or float(cast(int | float, targeted_streak_scale)) > 0.0
        )
    )
    targeted_reward_binding_valid = bool(
        (not targeted_reward_enabled and failure_conditioned_critic is False)
        or (
            failure_conditioned_critic is True
            and asymmetric_critic is True
            and isinstance(failure_state_fraction, (int, float))
            and not isinstance(failure_state_fraction, bool)
            and float(failure_state_fraction) > 0.0
        )
    )
    route_hashes_valid = bool(
        isinstance(route_manifest_hash, str)
        and isinstance(route_group_hash, str)
        and len(route_manifest_hash) == 71
        and len(route_group_hash) == 71
        and route_manifest_hash.startswith("sha256:")
        and route_group_hash.startswith("sha256:")
    )
    if (
        payload.get("schema_version")
        != "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_report.v1"
        or declared != hash_json(payload)
        or not isinstance(config, dict)
        or not isinstance(devices, list)
        or len(devices) < int(config.get("required_gpu_count", 10**9))
        or payload.get("rollout_backend") != "MUJOCO_MJX"
        or payload.get("physics_truth_backend") != "CPU_MUJOCO"
        or payload.get("teacher_frozen") is not True
        or not actor_observation_valid
        or not observation_migration_valid
        or not parent_actor_retention_valid
        or not residual_regularization_target_valid
        or not critic_observation_valid
        or not targeted_reward_contract_valid
        or not targeted_reward_binding_valid
        or not signed_velocity_diagnostics_valid
        or not terminal_balance_curriculum_valid
        or not failure_state_curriculum_valid
        or payload.get("deployment_candidate") is not False
        or payload.get("requires_reference_free_distillation") is not True
        or payload.get("requires_independent_cpu_mujoco_exam") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
        or not isinstance(continued, bool)
        or (continued and not isinstance(parent_hash, str))
        or (not continued and parent_hash is not None)
        or (not continued and parent_training_report_hash is not None)
        or not isinstance(route_binding_enforced, bool)
        or (route_binding_enforced and not route_hashes_valid)
        or (
            route_binding_enforced
            and continued
            and (
                not isinstance(parent_training_report_hash, str)
                or len(parent_training_report_hash) != 71
                or not parent_training_report_hash.startswith("sha256:")
            )
        )
        or (
            not route_binding_enforced
            and (route_manifest_hash is not None or route_group_hash is not None)
        )
    ):
        raise ValueError("recovery MJX teacher-residual report is invalid")
    payload["report_hash"] = declared
    return payload


def build_recovery_mjx_directional_curriculum(
    *,
    training_report_path: Path,
    output_path: Path,
    evaluation_step: int = 0,
) -> dict[str, Any]:
    """Turn signed full-route failure telemetry into a bound replay curriculum."""

    if not isinstance(evaluation_step, int) or isinstance(evaluation_step, bool):
        raise ValueError("recovery MJX curriculum evaluation step is invalid")
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("recovery MJX directional curriculum refuses to overwrite evidence")
    source = validate_recovery_mjx_teacher_residual_report(training_report_path)
    diagnostic_contract = source.get("signed_velocity_diagnostics")
    if not isinstance(diagnostic_contract, dict):
        raise ValueError("recovery MJX source lacks signed velocity diagnostics")
    progress = source.get("progress")
    if not isinstance(progress, list):
        raise ValueError("recovery MJX source progress is invalid")
    matching_rows = [
        row for row in progress if isinstance(row, dict) and row.get("step") == evaluation_step
    ]
    if len(matching_rows) != 1 or not isinstance(matching_rows[0].get("metrics"), dict):
        raise ValueError("recovery MJX curriculum evaluation row is absent")
    metrics = matching_rows[0]["metrics"]
    required = (
        "eval/avg_episode_length",
        "eval/episode_root_body_forward_velocity",
        "eval/episode_root_body_lateral_velocity",
        "eval/episode_root_body_vertical_velocity",
        "eval/episode_pelvis_yaw_rate",
    )
    try:
        values = {name: float(metrics[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("recovery MJX directional metrics are incomplete") from exc
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("recovery MJX directional metrics must be finite")
    episode_length = values["eval/avg_episode_length"]
    if episode_length <= 0.0:
        raise ValueError("recovery MJX directional episode length is invalid")
    per_step = {
        "body_forward_velocity_mps": (
            values["eval/episode_root_body_forward_velocity"] / episode_length
        ),
        "body_lateral_velocity_mps": (
            values["eval/episode_root_body_lateral_velocity"] / episode_length
        ),
        "body_vertical_velocity_mps": (
            values["eval/episode_root_body_vertical_velocity"] / episode_length
        ),
        "pelvis_yaw_rate_rad_s": values["eval/episode_pelvis_yaw_rate"] / episode_length,
    }
    # Curriculum impulses intentionally reproduce the observed failure
    # direction, with conservative bounds below the training safety envelope.
    body_bias = [
        max(-0.30, min(0.0, per_step["body_forward_velocity_mps"])),
        max(-0.20, min(0.20, per_step["body_lateral_velocity_mps"])),
        0.0,
    ]
    yaw_bias = max(-1.0, min(1.0, per_step["pelvis_yaw_rate_rad_s"]))
    result: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_directional_curriculum.v1",
        "source_training_report_hash": source["report_hash"],
        "source_route_manifest_hash": source.get("route_manifest_hash"),
        "source_route_group_hash": source.get("route_group_hash"),
        "source_evaluation_step": evaluation_step,
        "source_evaluation_environment_count": source.get("config", {}).get("num_eval_envs"),
        "source_mean_signed_velocity": per_step,
        "terminal_body_linear_velocity_bias_mps": body_bias,
        "terminal_pelvis_yaw_rate_bias_rad_s": yaw_bias,
        "derivation": "CLIPPED_MEAN_SIGNED_FULL_ROUTE_FAILURE_DIRECTION",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    result["report_hash"] = hash_json(result)
    _atomic_json(target, result)
    return result


def validate_recovery_mjx_directional_curriculum(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX directional curriculum is invalid")
    declared = payload.pop("report_hash", None)
    linear_bias = payload.get("terminal_body_linear_velocity_bias_mps")
    yaw_bias = payload.get("terminal_pelvis_yaw_rate_bias_rad_s")
    source_mean = payload.get("source_mean_signed_velocity")
    hashes = (
        payload.get("source_training_report_hash"),
        payload.get("source_route_manifest_hash"),
        payload.get("source_route_group_hash"),
    )
    valid_bias = bool(
        isinstance(linear_bias, list)
        and len(linear_bias) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in linear_bias
        )
        and -0.30 <= float(linear_bias[0]) <= 0.0
        and -0.20 <= float(linear_bias[1]) <= 0.20
        and float(linear_bias[2]) == 0.0
        and isinstance(yaw_bias, (int, float))
        and not isinstance(yaw_bias, bool)
        and math.isfinite(float(yaw_bias))
        and -1.0 <= float(yaw_bias) <= 1.0
    )
    valid_source_mean = bool(
        isinstance(source_mean, dict)
        and set(source_mean)
        == {
            "body_forward_velocity_mps",
            "body_lateral_velocity_mps",
            "body_vertical_velocity_mps",
            "pelvis_yaw_rate_rad_s",
        }
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in source_mean.values()
        )
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_mjx_directional_curriculum.v1"
        or declared != hash_json(payload)
        or not all(
            isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            for value in hashes
        )
        or not valid_source_mean
        or not isinstance(payload.get("source_evaluation_step"), int)
        or isinstance(payload.get("source_evaluation_step"), bool)
        or int(payload["source_evaluation_step"]) < 0
        or not isinstance(payload.get("source_evaluation_environment_count"), int)
        or isinstance(payload.get("source_evaluation_environment_count"), bool)
        or int(payload["source_evaluation_environment_count"]) <= 0
        or not valid_bias
        or payload.get("derivation") != "CLIPPED_MEAN_SIGNED_FULL_ROUTE_FAILURE_DIRECTION"
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX directional curriculum is invalid")
    payload["report_hash"] = declared
    return payload


def build_recovery_mjx_failure_window_plan(
    *,
    training_report_path: Path,
    output_path: Path,
    evaluation_step: int = 0,
    selected_window_count: int = 2,
) -> dict[str, Any]:
    """Prioritize exact rollout windows for subsequent state-snapshot collection."""

    if evaluation_step != 0 or not 1 <= selected_window_count <= 3:
        raise ValueError("recovery MJX failure-window request is invalid")
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("recovery MJX failure-window plan refuses to overwrite evidence")
    source = validate_recovery_mjx_teacher_residual_report(training_report_path)
    diagnostics = source.get("signed_velocity_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("temporal_bin_count") != 6
        or diagnostics.get("temporal_bin_semantics") != "EQUAL_WIDTH_BY_WRAPPER_CONTROL_STEP"
        or source.get("continued_from_parent") is not True
        or not isinstance(source.get("parent_checkpoint_hash"), str)
    ):
        raise ValueError("recovery MJX source lacks temporal baseline evidence")
    progress = source.get("progress")
    rows = (
        [row for row in progress if isinstance(row, dict) and row.get("step") == evaluation_step]
        if isinstance(progress, list)
        else []
    )
    if len(rows) != 1 or not isinstance(rows[0].get("metrics"), dict):
        raise ValueError("recovery MJX failure-window evaluation row is absent")
    metrics = rows[0]["metrics"]
    try:
        episode_length = int(float(metrics["eval/avg_episode_length"]))
        window_rows: list[dict[str, Any]] = []
        previous_score = 0.0
        for index in range(6):
            start_step = index * episode_length // 6
            end_step = (index + 1) * episode_length // 6
            width = end_step - start_step
            backward = (
                float(metrics[f"eval/episode_root_body_backward_speed_phase_{index}"]) / width
            )
            lateral = float(metrics[f"eval/episode_root_body_lateral_speed_phase_{index}"]) / width
            yaw = float(metrics[f"eval/episode_pelvis_yaw_speed_phase_{index}"]) / width
            score = backward / 0.50 + 0.5 * lateral / 0.50 + 0.25 * yaw / 1.50
            onset_gain = max(0.0, score - previous_score)
            priority = score + 1.5 * onset_gain
            window_rows.append(
                {
                    "window_index": index,
                    "start_step_inclusive": start_step,
                    "end_step_exclusive": end_step,
                    "mean_backward_speed_mps": backward,
                    "mean_lateral_speed_mps": lateral,
                    "mean_pelvis_yaw_speed_rad_s": yaw,
                    "normalized_failure_score": score,
                    "onset_gain": onset_gain,
                    "snapshot_priority": priority,
                }
            )
            previous_score = score
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("recovery MJX temporal failure metrics are incomplete") from exc
    numeric_values = [
        float(value)
        for row in window_rows
        for key, value in row.items()
        if key not in {"window_index", "start_step_inclusive", "end_step_exclusive"}
    ]
    if episode_length <= 0 or any(
        not math.isfinite(value) or value < 0.0 for value in numeric_values
    ):
        raise ValueError("recovery MJX temporal failure metrics are invalid")
    selected_indices = sorted(
        int(row["window_index"])
        for row in sorted(
            window_rows,
            key=lambda row: (-float(row["snapshot_priority"]), int(row["window_index"])),
        )[:selected_window_count]
    )
    collection_steps = sorted(
        {
            step
            for row in window_rows
            if int(row["window_index"]) in selected_indices
            for step in (
                int(row["start_step_inclusive"]),
                (int(row["start_step_inclusive"]) + int(row["end_step_exclusive"])) // 2,
                int(row["end_step_exclusive"]) - 1,
            )
        }
    )
    result: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_window_plan.v1",
        "source_training_report_hash": source["report_hash"],
        "source_checkpoint_hash": source["parent_checkpoint_hash"],
        "source_checkpoint_role": "PARENT_BASELINE_AT_STEP_ZERO",
        "source_route_manifest_hash": source.get("route_manifest_hash"),
        "source_route_group_hash": source.get("route_group_hash"),
        "source_evaluation_step": evaluation_step,
        "source_evaluation_environment_count": source.get("config", {}).get("num_eval_envs"),
        "episode_length": episode_length,
        "temporal_bin_count": 6,
        "window_rows": window_rows,
        "selected_window_indices": selected_indices,
        "requested_collection_steps": collection_steps,
        "selection_semantics": "FAILURE_MAGNITUDE_PLUS_ONSET_GAIN",
        "requires_exact_qpos_qvel_snapshot_collection": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    result["report_hash"] = hash_json(result)
    _atomic_json(target, result)
    return result


def validate_recovery_mjx_failure_window_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX failure-window plan is invalid")
    declared = payload.pop("report_hash", None)
    hashes = (
        payload.get("source_training_report_hash"),
        payload.get("source_checkpoint_hash"),
        payload.get("source_route_manifest_hash"),
        payload.get("source_route_group_hash"),
    )
    windows = payload.get("window_rows")
    selected = payload.get("selected_window_indices")
    steps = payload.get("requested_collection_steps")
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_mjx_failure_window_plan.v1"
        or declared != hash_json(payload)
        or not all(
            isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            for value in hashes
        )
        or payload.get("source_checkpoint_role") != "PARENT_BASELINE_AT_STEP_ZERO"
        or payload.get("source_evaluation_step") != 0
        or not isinstance(windows, list)
        or len(windows) != 6
        or not isinstance(selected, list)
        or not 1 <= len(selected) <= 3
        or selected != sorted(set(selected))
        or any(not isinstance(index, int) or not 0 <= index < 6 for index in selected)
        or not isinstance(steps, list)
        or not steps
        or steps != sorted(set(steps))
        or any(not isinstance(step, int) or step < 0 for step in steps)
        or payload.get("selection_semantics") != "FAILURE_MAGNITUDE_PLUS_ONSET_GAIN"
        or payload.get("requires_exact_qpos_qvel_snapshot_collection") is not True
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX failure-window plan is invalid")
    payload["report_hash"] = declared
    return payload


def validate_recovery_mjx_failure_state_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX failure-state manifest is invalid")
    declared = payload.pop("report_hash", None)
    archive_name = payload.get("state_archive")
    archive_path = manifest_path.parent / str(archive_name)
    hashes = (
        payload.get("source_failure_window_plan_hash"),
        payload.get("source_failure_window_plan_file_hash"),
        payload.get("source_training_report_hash"),
        payload.get("source_actor_checkpoint_hash"),
        payload.get("source_actor_config_hash"),
        payload.get("source_route_manifest_hash"),
        payload.get("source_route_group_hash"),
        payload.get("teacher_checkpoint_hash"),
        payload.get("motion_archive_hash"),
        payload.get("snapshot_manifest_hash"),
        payload.get("state_archive_hash"),
    )
    config = payload.get("config")
    count = payload.get("collected_state_count")
    requested_steps = payload.get("requested_collection_steps")
    schema_version = payload.get("schema_version")
    exact_policy_context = schema_version == "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2"
    archive_valid = False
    if (
        archive_name == "failure-window-states.npz"
        and archive_path.is_file()
        and payload.get("state_archive_hash") == hash_bytes(archive_path.read_bytes())
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    ):
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                base_keys = {
                    "qpos",
                    "qvel",
                    "control_step",
                    "environment_index",
                    "handoff_frozen",
                    "trajectory_step",
                    "trajectory_initial_step",
                    "root_body_backward_speed_mps",
                    "root_body_lateral_speed_mps",
                    "pelvis_yaw_speed_rad_s",
                }
                policy_context_keys = {
                    "last_motor_targets",
                    "last_teacher_action",
                    "last_residual",
                    "proprioception_history",
                    "phase_repeat",
                }
                expected_keys = (
                    base_keys | policy_context_keys if exact_policy_context else base_keys
                )
                scalar_keys = expected_keys - {
                    "qpos",
                    "qvel",
                    "last_motor_targets",
                    "last_teacher_action",
                    "last_residual",
                    "proprioception_history",
                }
                policy_context_valid = not exact_policy_context
                if exact_policy_context:
                    history_shape = payload.get("proprioception_history_shape")
                    context_features = payload.get("context_features_collected")
                    policy_context_valid = bool(
                        archive["last_motor_targets"].shape == (count, _JOINT_COUNT)
                        and archive["last_teacher_action"].shape == (count, _JOINT_COUNT)
                        and archive["last_residual"].shape == (count, _JOINT_COUNT)
                        and isinstance(history_shape, list)
                        and len(history_shape) == 3
                        and history_shape[0] == count
                        and isinstance(history_shape[1], int)
                        and 1 <= history_shape[1] <= 16
                        and isinstance(history_shape[2], int)
                        and 90 <= history_shape[2] <= 128
                        and archive["proprioception_history"].shape == tuple(history_shape)
                        and context_features == _FAILURE_STATE_POLICY_CONTEXT
                        and np.min(archive["phase_repeat"]) >= 0
                        and np.max(archive["phase_repeat"]) <= 3
                    )
                archive_valid = bool(
                    set(archive.files) == expected_keys
                    and archive["qpos"].shape == (count, 36)
                    and archive["qvel"].shape == (count, 35)
                    and payload.get("qpos_shape") == [count, 36]
                    and payload.get("qvel_shape") == [count, 35]
                    and all(archive[name].shape == (count,) for name in scalar_keys)
                    and all(np.all(np.isfinite(archive[name])) for name in expected_keys)
                    and policy_context_valid
                    and isinstance(requested_steps, list)
                    and requested_steps == sorted(set(requested_steps))
                    and set(np.asarray(archive["control_step"], dtype=np.int64).tolist())
                    == set(requested_steps)
                    and np.min(archive["environment_index"]) >= 0
                    and isinstance(config, dict)
                    and np.max(archive["environment_index"])
                    < int(config.get("num_environments", 0))
                )
        except (KeyError, OSError, TypeError, ValueError):
            archive_valid = False
    if (
        schema_version
        not in {
            "rosclaw_soccer.recovery_mjx_failure_state_manifest.v1",
            "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2",
        }
        or declared != hash_json(payload)
        or not all(
            isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            for value in hashes
        )
        or not archive_valid
        or payload.get("rollout_backend") != "MUJOCO_MJX"
        or payload.get("physics_truth_backend") != "CPU_MUJOCO_REQUIRED_FOR_PROMOTION"
        or payload.get("deterministic_actor") is not True
        or payload.get("full_route_reset") is not True
        or payload.get("curriculum_use_only") is not True
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX failure-state manifest is invalid")
    payload["report_hash"] = declared
    return payload


def validate_recovery_mjx_failure_state_exam_report(path: Path) -> dict[str, Any]:
    """Validate a paired, diagnostic-only failure-state actor comparison."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX failure-state exam report is invalid")
    declared = payload.pop("report_hash", None)
    config = payload.get("config")
    coverage = payload.get("state_coverage")
    parent = payload.get("parent_metrics")
    candidate = payload.get("candidate_metrics")
    relative_changes = payload.get("relative_changes")
    per_window = payload.get("per_failure_window")
    gates = payload.get("retention_gates")
    hashes = (
        payload.get("failure_state_manifest_hash"),
        payload.get("failure_state_manifest_file_hash"),
        payload.get("failure_state_archive_hash"),
        payload.get("parent_training_report_hash"),
        payload.get("parent_checkpoint_hash"),
        payload.get("parent_training_checkpoint_tree_hash"),
        payload.get("candidate_training_report_hash"),
        payload.get("candidate_checkpoint_hash"),
        payload.get("candidate_training_checkpoint_tree_hash"),
        payload.get("route_manifest_hash"),
        payload.get("route_group_hash"),
        payload.get("teacher_checkpoint_hash"),
    )
    metric_keys = {
        "episode_count",
        "mean_episode_length",
        "success_rate",
        "stable_fraction",
        "ready_fraction",
        "mean_maximum_stable_streak",
        "root_body_backward_speed_mps",
        "root_body_lateral_speed_mps",
        "pelvis_yaw_speed_rad_s",
        "mean_reward_per_step",
        "non_success_termination_rate",
    }
    actor_observation_contracts = {
        "DEPLOYABLE_PROPRIOCEPTION_ONLY",
        "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY",
        "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_ONLY",
        "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_HISTORY_ONLY",
        "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_ONLY",
        "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY",
        "DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_ONLY",
        "DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY",
    }
    exam_schema = payload.get("schema_version")
    modern_exam_schema = exam_schema in {
        "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v3",
        "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4",
    }
    snapshot_binding_values = (
        payload.get("snapshot_manifest_hash"),
        payload.get("parent_snapshot_manifest_hash"),
        payload.get("candidate_snapshot_manifest_hash"),
        payload.get("failure_state_snapshot_manifest_hash"),
    )
    snapshot_binding_valid = bool(
        exam_schema
        in {
            "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v1",
            "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v2",
        }
        or (
            all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in snapshot_binding_values
            )
            and len(set(snapshot_binding_values)) == 1
        )
    )
    actor_contracts_present = any(
        name in payload
        for name in (
            "parent_actor_observation",
            "candidate_actor_observation",
            "parent_actor_observation_dim",
            "candidate_actor_observation_dim",
        )
    )
    exam_observation_migration = payload.get("actor_observation_migration")
    cross_dimension_migration_valid = bool(
        modern_exam_schema
        and isinstance(exam_observation_migration, dict)
        and exam_observation_migration.get("strategy")
        == "FROZEN_PARENT_TRUNK_ZERO_OUTPUT_VELOCITY_ADAPTER"
        and exam_observation_migration.get("source_actor_observation_dim")
        == payload.get("parent_actor_observation_dim")
        and exam_observation_migration.get("target_actor_observation_dim")
        == payload.get("candidate_actor_observation_dim")
        and exam_observation_migration.get("source_frame_dim") == 96
        and exam_observation_migration.get("target_frame_dim") == 99
        and exam_observation_migration.get("legacy_accelerometer_slots_preserved") is True
        and exam_observation_migration.get("parent_policy_trunk_frozen") is True
        and exam_observation_migration.get("new_adapter_output_weights_initialized_to_zero") is True
        and exam_observation_migration.get("adapter_location_limit") == 0.25
        and exam_observation_migration.get("parent_policy_gradient_max_abs") == 0.0
        and isinstance(exam_observation_migration.get("adapter_output_gradient_l2"), (int, float))
        and not isinstance(exam_observation_migration.get("adapter_output_gradient_l2"), bool)
        and math.isfinite(float(exam_observation_migration["adapter_output_gradient_l2"]))
        and float(exam_observation_migration["adapter_output_gradient_l2"]) > 0.0
        and exam_observation_migration.get("new_value_input_weights_initialized_to_zero") is True
        and exam_observation_migration.get("behavior_preserved") is True
        and all(
            isinstance(exam_observation_migration.get(name), (int, float))
            and not isinstance(exam_observation_migration.get(name), bool)
            and math.isfinite(float(exam_observation_migration[name]))
            and 0.0 <= float(exam_observation_migration[name]) <= 1.0e-6
            for name in ("policy_output_max_abs_error", "value_output_max_abs_error")
        )
    )
    actor_contracts_valid = bool(
        (
            payload.get("parent_actor_observation") in actor_observation_contracts
            and payload.get("candidate_actor_observation") in actor_observation_contracts
            and isinstance(payload.get("parent_actor_observation_dim"), int)
            and payload.get("parent_actor_observation_dim", 0) > 0
            and isinstance(payload.get("candidate_actor_observation_dim"), int)
            and payload.get("candidate_actor_observation_dim", 0) > 0
            and (
                payload.get("candidate_actor_observation_dim")
                == payload.get("parent_actor_observation_dim")
                or cross_dimension_migration_valid
            )
        )
        or (
            exam_schema == "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v1"
            and not actor_contracts_present
        )
    )

    def finite_number(value: Any) -> bool:
        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    action_authority = payload.get("candidate_action_authority")
    action_authority_valid = bool(
        isinstance(action_authority, dict)
        and set(action_authority)
        == {
            "baseline",
            "normalized_action_space",
            "motor_target_space",
            "frozen_parent_baseline",
            "mean_normalized_action_increment_rms",
            "maximum_normalized_action_increment_rms",
            "mean_motor_target_increment_rms_rad",
            "maximum_motor_target_increment_rms_rad",
            "residual_active_fraction",
        }
        and isinstance(action_authority.get("frozen_parent_baseline"), bool)
        and action_authority.get("baseline")
        == (
            "FROZEN_PARENT_RESIDUAL_POLICY"
            if action_authority.get("frozen_parent_baseline") is True
            else "ZERO_RESIDUAL_POLICY"
        )
        and action_authority.get("normalized_action_space") == "CANDIDATE_MINUS_BASELINE_RESIDUAL"
        and action_authority.get("motor_target_space")
        == "CLIPPED_CANDIDATE_MINUS_CLIPPED_BASELINE_RAD"
        and all(
            finite_number(action_authority.get(name))
            and 0.0 <= float(action_authority[name]) <= 2.0
            for name in (
                "mean_normalized_action_increment_rms",
                "maximum_normalized_action_increment_rms",
                "mean_motor_target_increment_rms_rad",
                "maximum_motor_target_increment_rms_rad",
            )
        )
        and float(action_authority.get("maximum_normalized_action_increment_rms", -1.0))
        >= float(action_authority.get("mean_normalized_action_increment_rms", float("inf")))
        and float(action_authority.get("maximum_motor_target_increment_rms_rad", -1.0))
        >= float(action_authority.get("mean_motor_target_increment_rms_rad", float("inf")))
        and finite_number(action_authority.get("residual_active_fraction"))
        and 0.0 <= float(action_authority["residual_active_fraction"]) <= 1.0
    )

    def valid_metrics(value: Any) -> bool:
        return bool(
            isinstance(value, dict)
            and set(value) == metric_keys
            and all(finite_number(metric) for metric in value.values())
            and isinstance(value.get("episode_count"), int)
            and value.get("episode_count", 0) > 0
        )

    metrics_valid = valid_metrics(parent) and valid_metrics(candidate)
    if metrics_valid:
        assert isinstance(parent, dict)
        assert isinstance(candidate, dict)
        metrics_valid = parent["episode_count"] == candidate["episode_count"]

    def relative_change(candidate_value: float, parent_value: float) -> float:
        return (candidate_value - parent_value) / max(abs(parent_value), 1e-12)

    relative_metric_names = {
        "stable_fraction": "stable_fraction",
        "maximum_stable_streak": "mean_maximum_stable_streak",
        "root_body_backward_speed": "root_body_backward_speed_mps",
        "root_body_lateral_speed": "root_body_lateral_speed_mps",
        "pelvis_yaw_speed": "pelvis_yaw_speed_rad_s",
    }
    expected_relative_changes: dict[str, float] = {}
    if metrics_valid:
        assert isinstance(parent, dict)
        assert isinstance(candidate, dict)
        expected_relative_changes = {
            name: relative_change(float(candidate[metric]), float(parent[metric]))
            for name, metric in relative_metric_names.items()
        }
    relative_changes_valid = bool(
        isinstance(relative_changes, dict)
        and set(relative_changes) == set(relative_metric_names)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(
                float(value), expected_relative_changes.get(name, float("inf")), abs_tol=1e-12
            )
            for name, value in relative_changes.items()
        )
    )
    gate_keys = {
        "coverage_passed",
        "success_or_stable_passed",
        "maximum_streak_passed",
        "backward_speed_passed",
        "lateral_speed_passed",
        "yaw_speed_passed",
        "termination_safety_passed",
    }
    threshold_names = (
        "minimum_state_coverage_fraction",
        "minimum_stable_improvement_fraction",
        "maximum_streak_regression_fraction",
        "minimum_backward_speed_improvement_fraction",
        "maximum_lateral_speed_regression_fraction",
        "maximum_yaw_speed_regression_fraction",
    )
    config_valid = False
    if isinstance(config, dict):
        num_environments = config.get("num_environments")
        horizon_steps = config.get("horizon_steps")
        random_seed = config.get("random_seed")
        thresholds = tuple(config.get(name) for name in threshold_names)
        candidate_adapter_gain = config.get("candidate_adapter_gain", 1.0)
        blind_compatible_failure_bank = config.get("allow_blind_compatible_failure_bank", False)
        config_valid = bool(
            config.get("schema_version")
            == "rosclaw_soccer.recovery_mjx_failure_state_exam_config.v1"
            and isinstance(num_environments, int)
            and not isinstance(num_environments, bool)
            and 96 <= num_environments <= 2_048
            and num_environments % 4 == 0
            and isinstance(horizon_steps, int)
            and not isinstance(horizon_steps, bool)
            and 100 <= horizon_steps <= 1_200
            and isinstance(random_seed, int)
            and not isinstance(random_seed, bool)
            and 0 <= random_seed < 2**31
            and all(finite_number(value) for value in thresholds)
            and finite_number(candidate_adapter_gain)
            and 1.0 <= float(candidate_adapter_gain) <= 4.0
            and isinstance(blind_compatible_failure_bank, bool)
            and 0.90 <= float(cast(int | float, thresholds[0])) <= 1.0
            and all(0.0 <= float(cast(int | float, value)) <= 0.25 for value in thresholds[1:])
            and config.get("activation_ceiling") == "SIM_ONLY"
            and config.get("hardware_authorized") is False
        )
    coverage_valid = False
    if isinstance(config, dict) and isinstance(coverage, dict) and isinstance(parent, dict):
        unique_count = coverage.get("unique_state_count")
        total_count = coverage.get("total_state_count")
        coverage_fraction = coverage.get("coverage_fraction")
        required_steps = coverage.get("required_control_steps")
        covered_steps = coverage.get("covered_control_steps")
        coverage_valid = bool(
            config_valid
            and isinstance(unique_count, int)
            and isinstance(total_count, int)
            and isinstance(coverage_fraction, (int, float))
            and not isinstance(coverage_fraction, bool)
            and isinstance(required_steps, list)
            and isinstance(covered_steps, list)
            and all(isinstance(value, int) for value in required_steps)
            and unique_count > 0
            and total_count > 0
            and unique_count <= total_count
            and math.isclose(
                float(coverage_fraction),
                float(unique_count) / float(total_count),
                abs_tol=1e-12,
            )
            and metrics_valid
            and parent.get("episode_count") == config.get("num_environments")
        )
    windows_valid = False
    if (
        coverage_valid
        and isinstance(coverage, dict)
        and isinstance(per_window, list)
        and isinstance(parent, dict)
    ):
        required_steps = coverage.get("required_control_steps", [])
        covered_steps = coverage.get("covered_control_steps", [])
        windows_valid = bool(
            set(covered_steps).issubset(required_steps)
            and len(per_window) == len(covered_steps)
            and [row.get("control_step") for row in per_window if isinstance(row, dict)]
            == covered_steps
            and all(
                isinstance(row, dict)
                and set(row)
                == {"control_step", "episode_count", "parent_metrics", "candidate_metrics"}
                and isinstance(row.get("episode_count"), int)
                and row.get("episode_count", 0) > 0
                and valid_metrics(row.get("parent_metrics"))
                and valid_metrics(row.get("candidate_metrics"))
                and row["parent_metrics"]["episode_count"] == row["episode_count"]
                and row["candidate_metrics"]["episode_count"] == row["episode_count"]
                for row in per_window
            )
            and sum(row["episode_count"] for row in per_window) == parent.get("episode_count")
        )
    expected_gates: dict[str, bool] = {}
    if coverage_valid and relative_changes_valid:
        assert isinstance(config, dict)
        assert isinstance(coverage, dict)
        assert isinstance(parent, dict)
        assert isinstance(candidate, dict)
        assert isinstance(relative_changes, dict)
        success_improved = float(candidate["success_rate"]) > float(parent["success_rate"])
        expected_gates = {
            "coverage_passed": (
                float(coverage["coverage_fraction"])
                >= float(config["minimum_state_coverage_fraction"])
                and coverage["covered_control_steps"] == coverage["required_control_steps"]
            ),
            "success_or_stable_passed": (
                float(candidate["success_rate"]) >= float(parent["success_rate"])
                and (
                    success_improved
                    or float(relative_changes["stable_fraction"])
                    >= float(config.get("minimum_stable_improvement_fraction", float("inf")))
                )
            ),
            "maximum_streak_passed": float(relative_changes["maximum_stable_streak"])
            >= -float(config.get("maximum_streak_regression_fraction", -float("inf"))),
            "backward_speed_passed": float(relative_changes["root_body_backward_speed"])
            <= -float(config.get("minimum_backward_speed_improvement_fraction", float("inf"))),
            "lateral_speed_passed": float(relative_changes["root_body_lateral_speed"])
            <= float(config.get("maximum_lateral_speed_regression_fraction", -float("inf"))),
            "yaw_speed_passed": float(relative_changes["pelvis_yaw_speed"])
            <= float(config.get("maximum_yaw_speed_regression_fraction", -float("inf"))),
            "termination_safety_passed": float(candidate["non_success_termination_rate"])
            <= float(parent["non_success_termination_rate"]),
        }
    gates_valid = bool(
        isinstance(gates, dict)
        and set(gates) == gate_keys
        and all(isinstance(value, bool) for value in gates.values())
        and gates == expected_gates
        and isinstance(payload.get("local_retention_passed"), bool)
        and payload.get("local_retention_passed") == all(gates.values())
    )
    if (
        exam_schema
        not in {
            "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v1",
            "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v2",
            "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v3",
            "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4",
        }
        or declared != hash_json(payload)
        or not all(
            isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
            for value in hashes
        )
        or payload.get("rollout_backend") != "MUJOCO_MJX"
        or payload.get("physics_truth_backend") != "CPU_MUJOCO_REQUIRED_FOR_PROMOTION"
        or payload.get("paired_identical_reset_keys") is not True
        or (
            payload.get("paired_reset_key_strategy")
            != "DETERMINISTIC_STRATIFIED_FULL_FAILURE_BANK_COVERAGE"
            and not (
                exam_schema == "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v1"
                and payload.get("paired_reset_key_strategy") is None
            )
        )
        or payload.get("diagnostic_failure_state_reset_fraction") != 1.0
        or payload.get("observed_failure_state_reset_fraction") != 1.0
        or not coverage_valid
        or not metrics_valid
        or not actor_contracts_valid
        or (modern_exam_schema and not snapshot_binding_valid)
        or (
            modern_exam_schema
            and payload.get("candidate_actor_observation_dim")
            != payload.get("parent_actor_observation_dim")
            and not cross_dimension_migration_valid
        )
        or (
            modern_exam_schema
            and payload.get("candidate_actor_observation_dim")
            == payload.get("parent_actor_observation_dim")
            and exam_observation_migration is not None
        )
        or (
            exam_schema == "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4"
            and not action_authority_valid
        )
        or (
            exam_schema != "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4"
            and action_authority is not None
        )
        or not relative_changes_valid
        or not windows_valid
        or not gates_valid
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX failure-state exam report is invalid")
    payload["report_hash"] = declared
    return payload


def validate_recovery_mjx_action_distribution_audit(path: Path) -> dict[str, Any]:
    """Validate normalized normal-route versus exact-failure action authority."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX action-distribution audit is invalid")
    declared = payload.pop("report_hash", None)
    normal = payload.get("normal_route_action_authority")
    failure = payload.get("failure_state_action_authority")
    ratios = payload.get("failure_to_normal_authority_ratio")

    def finite_nonnegative(value: Any) -> bool:
        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
        )

    authority_keys = {
        "mean_normalized_action_increment_rms",
        "mean_motor_target_increment_rms_rad",
    }
    authorities_valid = bool(
        isinstance(normal, dict)
        and set(normal) == authority_keys | {"evaluation_step", "average_episode_length"}
        and isinstance(normal.get("evaluation_step"), int)
        and not isinstance(normal.get("evaluation_step"), bool)
        and int(normal["evaluation_step"]) >= 0
        and finite_nonnegative(normal.get("average_episode_length"))
        and float(normal["average_episode_length"]) > 0.0
        and isinstance(failure, dict)
        and set(failure) == authority_keys
        and all(
            finite_nonnegative(row.get(name))
            for row in (normal, failure)
            for name in authority_keys
        )
    )
    expected_ratios: dict[str, float] = {}
    if authorities_valid:
        assert isinstance(normal, dict)
        assert isinstance(failure, dict)
        expected_ratios = {
            name: float(failure[name]) / max(float(normal[name]), 1.0e-12)
            for name in authority_keys
        }
    ratios_valid = bool(
        isinstance(ratios, dict)
        and set(ratios) == authority_keys
        and all(
            finite_nonnegative(value)
            and math.isclose(float(value), expected_ratios.get(name, float("inf")), abs_tol=1.0e-12)
            for name, value in ratios.items()
        )
    )
    threshold = payload.get("minimum_effective_normalized_action_rms")
    diagnosis = payload.get("diagnosis")
    expected_diagnosis = None
    if (
        authorities_valid
        and finite_nonnegative(threshold)
        and float(cast(int | float, threshold)) > 0.0
    ):
        assert isinstance(normal, dict)
        assert isinstance(failure, dict)
        threshold_value = float(cast(int | float, threshold))
        if (
            max(
                float(normal["mean_normalized_action_increment_rms"]),
                float(failure["mean_normalized_action_increment_rms"]),
            )
            < threshold_value
        ):
            expected_diagnosis = "GLOBAL_LOW_ACTION_AUTHORITY"
        elif expected_ratios["mean_normalized_action_increment_rms"] < 0.10:
            expected_diagnosis = "FAILURE_STATE_ACTION_AUTHORITY_COLLAPSE"
        else:
            expected_diagnosis = "ACTION_AUTHORITY_PRESENT_CHECK_CONTROL_DIRECTION"
    hashes = (
        payload.get("training_report_hash"),
        payload.get("failure_state_exam_report_hash"),
        payload.get("candidate_training_checkpoint_tree_hash"),
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_mjx_action_distribution_audit.v1"
        or declared != hash_json(payload)
        or not all(
            isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
            for value in hashes
        )
        or not authorities_valid
        or not ratios_valid
        or not finite_nonnegative(threshold)
        or not 0.0 < float(cast(int | float, threshold)) <= 0.25
        or diagnosis != expected_diagnosis
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX action-distribution audit is invalid")
    payload["report_hash"] = declared
    return payload


def build_recovery_mjx_action_distribution_audit(
    *,
    training_report_path: Path,
    failure_state_exam_path: Path,
    output_path: Path,
    minimum_effective_normalized_action_rms: float = 1.0e-3,
) -> dict[str, Any]:
    """Compare per-step action authority without mixing episode sums and means."""

    if (
        not math.isfinite(minimum_effective_normalized_action_rms)
        or not 0.0 < minimum_effective_normalized_action_rms <= 0.25
    ):
        raise ValueError("recovery MJX action-distribution threshold is invalid")
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("recovery MJX action-distribution audit refuses to overwrite evidence")
    training = validate_recovery_mjx_teacher_residual_report(training_report_path)
    exam = validate_recovery_mjx_failure_state_exam_report(failure_state_exam_path)
    action_authority = exam.get("candidate_action_authority")
    progress = training.get("progress")
    if (
        exam.get("schema_version") != "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4"
        or exam.get("candidate_training_report_hash") != training["report_hash"]
        or exam.get("candidate_training_checkpoint_tree_hash")
        != training.get("candidate_checkpoint_hash")
        or not isinstance(action_authority, dict)
        or not isinstance(progress, list)
        or not progress
    ):
        raise ValueError("recovery MJX action-distribution lineage is invalid")
    final_row = max(progress, key=lambda row: int(row["step"]))
    metrics = final_row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("recovery MJX action-distribution normal metrics are absent")
    episode_length = float(metrics["eval/avg_episode_length"])
    normal = {
        "evaluation_step": int(final_row["step"]),
        "average_episode_length": episode_length,
        "mean_normalized_action_increment_rms": float(metrics["eval/episode_adapter_residual_rms"])
        / episode_length,
        "mean_motor_target_increment_rms_rad": float(
            metrics["eval/episode_adapter_motor_target_delta_rms_rad"]
        )
        / episode_length,
    }
    failure = {
        "mean_normalized_action_increment_rms": float(
            action_authority["mean_normalized_action_increment_rms"]
        ),
        "mean_motor_target_increment_rms_rad": float(
            action_authority["mean_motor_target_increment_rms_rad"]
        ),
    }
    ratio_names = tuple(failure)
    ratios = {name: failure[name] / max(float(normal[name]), 1.0e-12) for name in ratio_names}
    maximum_authority = max(
        normal["mean_normalized_action_increment_rms"],
        failure["mean_normalized_action_increment_rms"],
    )
    diagnosis = (
        "GLOBAL_LOW_ACTION_AUTHORITY"
        if maximum_authority < minimum_effective_normalized_action_rms
        else "FAILURE_STATE_ACTION_AUTHORITY_COLLAPSE"
        if ratios["mean_normalized_action_increment_rms"] < 0.10
        else "ACTION_AUTHORITY_PRESENT_CHECK_CONTROL_DIRECTION"
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_action_distribution_audit.v1",
        "training_report_hash": training["report_hash"],
        "failure_state_exam_report_hash": exam["report_hash"],
        "candidate_training_checkpoint_tree_hash": training["candidate_checkpoint_hash"],
        "normal_route_action_authority": normal,
        "failure_state_action_authority": failure,
        "failure_to_normal_authority_ratio": ratios,
        "minimum_effective_normalized_action_rms": (minimum_effective_normalized_action_rms),
        "diagnosis": diagnosis,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return validate_recovery_mjx_action_distribution_audit(target)


def select_recovery_mjx_teacher_residual_generation(
    *,
    training_report_path: Path,
    output_path: Path,
    minimum_stable_improvement_fraction: float = 0.02,
    maximum_linear_speed_regression_fraction: float = 0.03,
    maximum_angular_speed_regression_fraction: float = 0.02,
    maximum_mean_normalized_residual: float = 0.05,
    minimum_maximum_stable_streak_improvement_fraction: float = 0.02,
    minimum_backward_speed_improvement_fraction: float = 0.02,
    maximum_lateral_speed_regression_fraction: float = 0.03,
    maximum_yaw_rate_regression_fraction: float = 0.02,
) -> dict[str, Any]:
    """Select a non-regressing development generation from PPO evaluations.

    This is intentionally a development-only retention gate.  Even a selected
    checkpoint remains teacher-coupled and cannot gain promotion authority.
    """

    thresholds = (
        minimum_stable_improvement_fraction,
        maximum_linear_speed_regression_fraction,
        maximum_angular_speed_regression_fraction,
        maximum_mean_normalized_residual,
        minimum_maximum_stable_streak_improvement_fraction,
        minimum_backward_speed_improvement_fraction,
        maximum_lateral_speed_regression_fraction,
        maximum_yaw_rate_regression_fraction,
    )
    if (
        any(not math.isfinite(value) or value < 0.0 for value in thresholds)
        or minimum_stable_improvement_fraction > 1.0
        or maximum_linear_speed_regression_fraction > 1.0
        or maximum_angular_speed_regression_fraction > 1.0
        or minimum_maximum_stable_streak_improvement_fraction > 1.0
        or minimum_backward_speed_improvement_fraction > 1.0
        or maximum_lateral_speed_regression_fraction > 1.0
        or maximum_yaw_rate_regression_fraction > 1.0
        or not 0.0 < maximum_mean_normalized_residual <= 1.0
    ):
        raise ValueError("recovery MJX generation thresholds are invalid")
    source_path = training_report_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("recovery MJX generation selection refuses to overwrite evidence")
    training = validate_recovery_mjx_teacher_residual_report(source_path)
    training_config = training["config"]
    evaluation_env_count = int(training_config.get("num_eval_envs", 0))
    if evaluation_env_count <= 0:
        raise ValueError("recovery MJX evaluation environment count is invalid")
    progress = training.get("progress")
    if not isinstance(progress, list) or len(progress) < 2:
        raise ValueError("recovery MJX training report has insufficient evaluations")

    required_metrics = (
        "eval/avg_episode_length",
        "eval/episode_reward",
        "eval/episode_root_angular_speed",
        "eval/episode_root_linear_speed",
        "eval/episode_stable",
        "eval/episode_success",
        "eval/episode_residual_rms",
    )
    directional_metrics = (
        "eval/episode_root_body_backward_speed",
        "eval/episode_root_body_lateral_speed",
        "eval/episode_pelvis_yaw_speed",
    )

    def read_row(raw: Any) -> dict[str, float | int]:
        if not isinstance(raw, dict) or not isinstance(raw.get("metrics"), dict):
            raise ValueError("recovery MJX evaluation row is invalid")
        metrics = raw["metrics"]
        try:
            row: dict[str, float | int] = {"step": int(raw["step"])}
            row.update({name: float(metrics[name]) for name in required_metrics})
            if "eval/episode_maximum_stable_streak" in metrics:
                row["eval/episode_maximum_stable_streak"] = float(
                    metrics["eval/episode_maximum_stable_streak"]
                )
            backward_present = directional_metrics[0] in metrics
            lateral_present = directional_metrics[1] in metrics
            yaw_speed_present = directional_metrics[2] in metrics
            yaw_rate_present = "eval/episode_pelvis_yaw_rate" in metrics
            any_directional = (
                backward_present or lateral_present or yaw_speed_present or yaw_rate_present
            )
            if any_directional and not (
                backward_present and lateral_present and (yaw_speed_present or yaw_rate_present)
            ):
                raise ValueError("recovery MJX directional evidence is incomplete")
            if any_directional:
                row[directional_metrics[0]] = float(metrics[directional_metrics[0]])
                row[directional_metrics[1]] = float(metrics[directional_metrics[1]])
                row[directional_metrics[2]] = (
                    float(metrics[directional_metrics[2]])
                    if yaw_speed_present
                    else abs(float(metrics["eval/episode_pelvis_yaw_rate"]))
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("recovery MJX evaluation metrics are incomplete") from exc
        if any(not math.isfinite(float(value)) for key, value in row.items() if key != "step"):
            raise ValueError("recovery MJX evaluation metrics must be finite")
        return row

    rows = [read_row(item) for item in progress]
    streak_rows = ["eval/episode_maximum_stable_streak" in row for row in rows]
    if any(streak_rows) and not all(streak_rows):
        raise ValueError("recovery MJX maximum-streak evidence is incomplete")
    maximum_streak_enforced = all(streak_rows)
    directional_rows = [all(name in row for name in directional_metrics) for row in rows]
    if any(directional_rows) and not all(directional_rows):
        raise ValueError("recovery MJX directional evidence is incomplete")
    directional_retention_enforced = all(directional_rows)
    baseline = rows[0]
    baseline_episode_length = float(baseline["eval/avg_episode_length"])
    if int(baseline["step"]) != 0 or baseline_episode_length <= 0.0:
        raise ValueError("recovery MJX generation baseline is invalid")
    baseline_stable = float(baseline["eval/episode_stable"]) / baseline_episode_length
    baseline_linear = float(baseline["eval/episode_root_linear_speed"]) / baseline_episode_length
    baseline_angular = float(baseline["eval/episode_root_angular_speed"]) / baseline_episode_length
    baseline_reward = float(baseline["eval/episode_reward"]) / baseline_episode_length
    baseline_success = float(baseline["eval/episode_success"])
    if (
        min(baseline_stable, baseline_linear, baseline_angular) <= 0.0
        or not 0.0 <= baseline_success <= 1.0
    ):
        raise ValueError("recovery MJX generation baseline cannot define retention")
    baseline_maximum_streak = float(baseline.get("eval/episode_maximum_stable_streak", 0.0))
    if maximum_streak_enforced and baseline_maximum_streak <= 0.0:
        raise ValueError("recovery MJX maximum-streak baseline cannot define retention")
    baseline_backward = baseline_lateral = baseline_yaw = 0.0
    if directional_retention_enforced:
        baseline_backward = (
            float(baseline["eval/episode_root_body_backward_speed"]) / baseline_episode_length
        )
        baseline_lateral = (
            float(baseline["eval/episode_root_body_lateral_speed"]) / baseline_episode_length
        )
        baseline_yaw = float(baseline["eval/episode_pelvis_yaw_speed"]) / baseline_episode_length
        if min(baseline_backward, baseline_lateral, baseline_yaw) <= 0.0:
            raise ValueError("recovery MJX directional baseline cannot define retention")

    evaluated: list[dict[str, Any]] = []
    for row in rows[1:]:
        episode_length = float(row["eval/avg_episode_length"])
        if episode_length <= 0.0:
            raise ValueError("recovery MJX candidate episode length is invalid")
        stable_gain = float(row["eval/episode_stable"]) / episode_length / baseline_stable - 1.0
        linear_change = (
            float(row["eval/episode_root_linear_speed"]) / episode_length / baseline_linear - 1.0
        )
        angular_change = (
            float(row["eval/episode_root_angular_speed"]) / episode_length / baseline_angular - 1.0
        )
        candidate_reward = float(row["eval/episode_reward"]) / episode_length
        reward_gain = (candidate_reward - baseline_reward) / max(abs(baseline_reward), 1.0e-9)
        residual_mean = float(row["eval/episode_residual_rms"]) / episode_length
        strict_success_rate = float(row["eval/episode_success"])
        if not 0.0 <= strict_success_rate <= 1.0:
            raise ValueError("recovery MJX strict success rate is invalid")
        strict_success_improved = strict_success_rate > baseline_success
        stable_evidence_passed = stable_gain >= minimum_stable_improvement_fraction or (
            strict_success_improved and stable_gain >= 0.0
        )
        maximum_streak_gain = (
            float(row["eval/episode_maximum_stable_streak"]) / baseline_maximum_streak - 1.0
            if maximum_streak_enforced
            else None
        )
        backward_change = (
            float(row["eval/episode_root_body_backward_speed"]) / episode_length / baseline_backward
            - 1.0
            if directional_retention_enforced
            else None
        )
        lateral_change = (
            float(row["eval/episode_root_body_lateral_speed"]) / episode_length / baseline_lateral
            - 1.0
            if directional_retention_enforced
            else None
        )
        yaw_change = (
            float(row["eval/episode_pelvis_yaw_speed"]) / episode_length / baseline_yaw - 1.0
            if directional_retention_enforced
            else None
        )
        directional_retention_passed = bool(
            backward_change is None
            or (
                backward_change <= -minimum_backward_speed_improvement_fraction
                and lateral_change is not None
                and lateral_change <= maximum_lateral_speed_regression_fraction
                and yaw_change is not None
                and yaw_change <= maximum_yaw_rate_regression_fraction
            )
        )
        retention_passed = bool(
            int(row["step"]) > 0
            and stable_evidence_passed
            and linear_change <= maximum_linear_speed_regression_fraction
            and angular_change <= maximum_angular_speed_regression_fraction
            and residual_mean <= maximum_mean_normalized_residual
            and directional_retention_passed
            and (
                maximum_streak_gain is None
                or maximum_streak_gain >= minimum_maximum_stable_streak_improvement_fraction
                or strict_success_improved
            )
        )
        evaluated.append(
            {
                "step": int(row["step"]),
                "average_episode_length": episode_length,
                "stable_improvement_fraction": stable_gain,
                "linear_speed_change_fraction": linear_change,
                "angular_speed_change_fraction": angular_change,
                "reward_improvement_fraction": reward_gain,
                "mean_normalized_residual": residual_mean,
                "strict_success_rate": strict_success_rate,
                "estimated_successful_evaluation_episodes": (
                    strict_success_rate * evaluation_env_count
                ),
                "strict_success_improved": strict_success_improved,
                "stable_evidence_passed": stable_evidence_passed,
                "maximum_stable_streak_improvement_fraction": maximum_streak_gain,
                "backward_speed_change_fraction": backward_change,
                "lateral_speed_change_fraction": lateral_change,
                "yaw_rate_change_fraction": yaw_change,
                "directional_retention_passed": directional_retention_passed,
                "retention_passed": retention_passed,
            }
        )
    accepted = [row for row in evaluated if row["retention_passed"]]
    selected = max(
        accepted,
        key=lambda row: (
            row["strict_success_rate"],
            row["maximum_stable_streak_improvement_fraction"] or -1.0,
            row["stable_improvement_fraction"],
            row["reward_improvement_fraction"],
            -row["angular_speed_change_fraction"],
        ),
        default=None,
    )
    checkpoint_files = training.get("candidate_checkpoint_files")
    if not isinstance(checkpoint_files, list):
        raise ValueError("recovery MJX checkpoint file evidence is missing")
    selected_checkpoint_files: list[dict[str, Any]] = []
    if selected is not None:
        prefix = f"{int(selected['step']):012d}/"
        selected_checkpoint_files = [
            dict(row)
            for row in checkpoint_files
            if isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and str(row["path"]).startswith(prefix)
        ]
        if not selected_checkpoint_files:
            raise ValueError("selected recovery MJX checkpoint has no bound files")
    result: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_generation_selection.v4",
        "training_report_hash": training["report_hash"],
        "candidate_checkpoint_tree_hash": training["candidate_checkpoint_hash"],
        "evaluation_environment_count": evaluation_env_count,
        "thresholds": {
            "minimum_stable_improvement_fraction": minimum_stable_improvement_fraction,
            "strict_success_may_replace_stable_gain_only_without_stable_regression": True,
            "maximum_linear_speed_regression_fraction": (maximum_linear_speed_regression_fraction),
            "maximum_angular_speed_regression_fraction": (
                maximum_angular_speed_regression_fraction
            ),
            "maximum_mean_normalized_residual": maximum_mean_normalized_residual,
            "minimum_maximum_stable_streak_improvement_fraction": (
                minimum_maximum_stable_streak_improvement_fraction
            ),
            "minimum_backward_speed_improvement_fraction": (
                minimum_backward_speed_improvement_fraction
            ),
            "maximum_lateral_speed_regression_fraction": (
                maximum_lateral_speed_regression_fraction
            ),
            "maximum_yaw_rate_regression_fraction": maximum_yaw_rate_regression_fraction,
        },
        "maximum_stable_streak_enforced": maximum_streak_enforced,
        "directional_retention_enforced": directional_retention_enforced,
        "evaluations": evaluated,
        "selected_step": selected["step"] if selected is not None else None,
        "selected_checkpoint_files": selected_checkpoint_files,
        "selected_checkpoint_hash": (
            hash_json(selected_checkpoint_files) if selected_checkpoint_files else None
        ),
        "development_retention_passed": selected is not None,
        "strict_success_observed": bool(
            selected is not None and selected["strict_success_rate"] > 0.0
        ),
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    result["report_hash"] = hash_json(result)
    _atomic_json(target, result)
    return result


def _validate_recovery_mjx_generation_selection(
    path: Path, *, training_report: dict[str, Any]
) -> dict[str, Any]:
    """Validate the development selector before composing a stronger gate."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX generation selection is invalid")
    declared = payload.pop("report_hash", None)
    evaluations = payload.get("evaluations")
    selected_step = payload.get("selected_step")
    evaluation_steps = (
        [row.get("step") for row in evaluations if isinstance(row, dict)]
        if isinstance(evaluations, list)
        else []
    )
    checkpoint_steps = sorted(
        {
            int(str(row["path"]).split("/", maxsplit=1)[0])
            for row in training_report.get("candidate_checkpoint_files", [])
            if isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and str(row["path"]).split("/", maxsplit=1)[0].isdigit()
        }
    )
    selected_rows = (
        [row for row in evaluations if isinstance(row, dict) and row.get("retention_passed")]
        if isinstance(evaluations, list)
        else []
    )
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_mjx_generation_selection.v4"
        or declared != hash_json(payload)
        or payload.get("training_report_hash") != training_report.get("report_hash")
        or payload.get("candidate_checkpoint_tree_hash")
        != training_report.get("candidate_checkpoint_hash")
        or not checkpoint_steps
        or evaluation_steps != checkpoint_steps
        or len(evaluation_steps) != len(set(evaluation_steps))
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("retention_passed"), bool)
            or not isinstance(row.get("step"), int)
            or isinstance(row.get("step"), bool)
            for row in (evaluations if isinstance(evaluations, list) else [])
        )
        or (selected_step is None) != (not selected_rows)
        or (
            selected_step is not None
            and not any(row.get("step") == selected_step for row in selected_rows)
        )
        or payload.get("development_retention_passed") is not (selected_step is not None)
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX generation selection is invalid")
    payload["report_hash"] = declared
    return payload


def _failure_exam_checkpoint_step(*, exam: dict[str, Any], training_report: dict[str, Any]) -> int:
    """Resolve an exam to exactly one content-bound persisted checkpoint."""

    exam_files = exam.get("candidate_checkpoint_files")
    if not isinstance(exam_files, list) or not exam_files:
        raise ValueError("failure-state exam candidate files are absent")
    normalized_exam_files = [dict(row) for row in exam_files if isinstance(row, dict)]
    if len(normalized_exam_files) != len(exam_files):
        raise ValueError("failure-state exam candidate files are invalid")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in training_report.get("candidate_checkpoint_files", []):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        prefix, separator, suffix = str(row["path"]).partition("/")
        if not separator or not prefix.isdigit() or not suffix:
            continue
        normalized = dict(row)
        normalized["path"] = suffix
        grouped.setdefault(int(prefix), []).append(normalized)
    matches = [step for step, files in grouped.items() if files == normalized_exam_files]
    if len(matches) != 1:
        raise ValueError("failure-state exam does not bind exactly one training checkpoint")
    step = matches[0]
    if exam.get("candidate_checkpoint_hash") != hash_json(normalized_exam_files):
        raise ValueError("failure-state exam checkpoint hash differs")
    return step


def _normalized_failure_constraint_residuals(exam: dict[str, Any]) -> dict[str, float]:
    """Return signed, dimensionless g(x) values for projected dual ascent."""

    config = exam["config"]
    relative = exam["relative_changes"]

    def scale(value: float) -> float:
        return max(abs(value), 0.01)

    minimum_stable = float(config["minimum_stable_improvement_fraction"])
    maximum_streak_regression = float(config["maximum_streak_regression_fraction"])
    minimum_backward = float(config["minimum_backward_speed_improvement_fraction"])
    maximum_lateral = float(config["maximum_lateral_speed_regression_fraction"])
    maximum_yaw = float(config["maximum_yaw_speed_regression_fraction"])
    return {
        # Lower-bound constraints are written target - observation <= 0.
        "stable": (minimum_stable - float(relative["stable_fraction"])) / scale(minimum_stable),
        "maximum_stable_streak": (
            -maximum_streak_regression - float(relative["maximum_stable_streak"])
        )
        / scale(maximum_streak_regression),
        # Cost constraints are observation - ceiling <= 0.
        "backward": (float(relative["root_body_backward_speed"]) + minimum_backward)
        / scale(minimum_backward),
        "lateral": (float(relative["root_body_lateral_speed"]) - maximum_lateral)
        / scale(maximum_lateral),
        "yaw": (float(relative["pelvis_yaw_speed"]) - maximum_yaw) / scale(maximum_yaw),
    }


def validate_recovery_mjx_failure_constrained_selection_report(path: Path) -> dict[str, Any]:
    """Validate a joint route/failure selector and projected-dual recommendation."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX failure-constrained selection is invalid")
    declared = payload.pop("report_hash", None)
    evaluations = payload.get("constraint_evaluations")
    current = payload.get("current_effective_multipliers")
    updated = payload.get("updated_effective_multipliers")
    patch = payload.get("recommended_next_generation_config_patch")
    selected_step = payload.get("selected_step")
    joint_rows = (
        [row for row in evaluations if isinstance(row, dict) and row.get("joint_retention_passed")]
        if isinstance(evaluations, list)
        else []
    )
    multiplier_keys = {"stable_streak", "backward", "lateral", "yaw"}
    if (
        payload.get("schema_version")
        != "rosclaw_soccer.recovery_mjx_failure_constrained_selection.v1"
        or declared != hash_json(payload)
        or not isinstance(evaluations, list)
        or not evaluations
        or any(
            not isinstance(row, dict)
            or set(row.get("normalized_signed_constraint_residuals", {}))
            != {"stable", "maximum_stable_streak", "backward", "lateral", "yaw"}
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in row.get("normalized_signed_constraint_residuals", {}).values()
            )
            or row.get("joint_retention_passed")
            is not bool(row.get("normal_retention_passed") and row.get("local_retention_passed"))
            for row in evaluations
        )
        or not isinstance(current, dict)
        or not isinstance(updated, dict)
        or set(current) != multiplier_keys
        or set(updated) != multiplier_keys
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in (*current.values(), *updated.values())
        )
        or not isinstance(patch, dict)
        or patch
        != {
            "failure_state_directional_penalty_scale": 1.0,
            "failure_state_backward_cost_weight": updated["backward"],
            "failure_state_lateral_cost_weight": updated["lateral"],
            "failure_state_yaw_cost_weight": updated["yaw"],
            "failure_state_stable_streak_reward_scale": updated["stable_streak"],
        }
        or (selected_step is None) != (not joint_rows)
        or (
            selected_step is not None
            and not any(row.get("step") == selected_step for row in joint_rows)
        )
        or payload.get("development_retention_passed") is not (selected_step is not None)
        or payload.get("all_persisted_checkpoints_examined") is not True
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX failure-constrained selection is invalid")
    payload["report_hash"] = declared
    return payload


def select_recovery_mjx_failure_constrained_generation(
    *,
    training_report_path: Path,
    generation_selection_path: Path,
    failure_state_exam_paths: tuple[Path, ...],
    output_path: Path,
    config: RecoveryMJXFailureConstraintConfig | None = None,
) -> dict[str, Any]:
    """Jointly gate every PPO checkpoint and recommend the next dual update.

    Normal-route averages and exact failure resets are deliberately conjunctive.
    Missing checkpoint exams, mixed reset keys, mixed parents, or mismatched
    checkpoint files fail closed.  If no checkpoint passes, the closest
    infeasible point drives one bounded projected-dual update for the next
    immutable generation.
    """

    active = config or RecoveryMJXFailureConstraintConfig()
    target = output_path.expanduser().resolve()
    if target.exists() or not failure_state_exam_paths:
        raise ValueError("failure-constrained selection output or exams are invalid")
    training = validate_recovery_mjx_teacher_residual_report(training_report_path)
    normal_selection = _validate_recovery_mjx_generation_selection(
        generation_selection_path, training_report=training
    )
    exams = [
        validate_recovery_mjx_failure_state_exam_report(path) for path in failure_state_exam_paths
    ]
    training_steps = [int(row["step"]) for row in normal_selection["evaluations"]]
    exams_by_step: dict[int, dict[str, Any]] = {}
    for exam in exams:
        step = _failure_exam_checkpoint_step(exam=exam, training_report=training)
        if step in exams_by_step:
            raise ValueError("failure-constrained selection has duplicate checkpoint exams")
        exams_by_step[step] = exam
    if active.require_all_persisted_checkpoint_exams and sorted(exams_by_step) != training_steps:
        raise ValueError("every persisted checkpoint requires one failure-state exam")

    first_exam = exams[0]
    invariant_exam_fields = (
        "failure_state_manifest_hash",
        "failure_state_archive_hash",
        "parent_training_report_hash",
        "parent_checkpoint_hash",
        "route_manifest_hash",
        "route_group_hash",
        "snapshot_manifest_hash",
        "paired_reset_key_strategy",
    )
    reference_config = dict(first_exam["config"])
    for exam in exams:
        if (
            exam.get("candidate_training_report_hash") != training["report_hash"]
            or exam.get("candidate_training_checkpoint_tree_hash")
            != training["candidate_checkpoint_hash"]
            or any(exam.get(name) != first_exam.get(name) for name in invariant_exam_fields)
            or exam.get("config") != reference_config
            or exam.get("paired_identical_reset_keys") is not True
            or exam.get("observed_failure_state_reset_fraction") != 1.0
        ):
            raise ValueError("failure-constrained checkpoint exams are not comparable")

    normal_by_step = {int(row["step"]): row for row in normal_selection["evaluations"]}
    evaluations: list[dict[str, Any]] = []
    for step in training_steps:
        exam = exams_by_step[step]
        residuals = _normalized_failure_constraint_residuals(exam)
        gates = exam["retention_gates"]
        hard_failures = [
            name
            for name in ("coverage_passed", "termination_safety_passed")
            if gates.get(name) is not True
        ]
        positive_violation = sum(max(value, 0.0) for value in residuals.values())
        normal_passed = normal_by_step[step]["retention_passed"] is True
        local_passed = exam["local_retention_passed"] is True
        evaluations.append(
            {
                "step": step,
                "normal_retention_passed": normal_passed,
                "local_retention_passed": local_passed,
                "joint_retention_passed": bool(normal_passed and local_passed),
                "local_exam_report_hash": exam["report_hash"],
                "normalized_signed_constraint_residuals": residuals,
                "positive_constraint_violation": positive_violation,
                "hard_gate_failures": hard_failures,
                "infeasibility_score": (
                    positive_violation
                    + 100.0 * len(hard_failures)
                    + (0.0 if normal_passed else 1.0)
                ),
                "normal_metrics": normal_by_step[step],
                "local_relative_changes": exam["relative_changes"],
                "local_candidate_metrics": exam["candidate_metrics"],
            }
        )
    accepted = [row for row in evaluations if row["joint_retention_passed"]]
    selected = max(
        accepted,
        key=lambda row: (
            float(row["local_candidate_metrics"]["success_rate"]),
            float(row["local_relative_changes"]["stable_fraction"]),
            float(row["local_relative_changes"]["maximum_stable_streak"]),
            -float(row["positive_constraint_violation"]),
        ),
        default=None,
    )
    update_source = min(
        evaluations,
        key=lambda row: (
            float(row["infeasibility_score"]),
            -float(row["local_candidate_metrics"]["success_rate"]),
            -int(row["step"]),
        ),
    )
    training_config = training["config"]
    directional_scale = float(training_config.get("failure_state_directional_penalty_scale", 0.0))
    current = {
        "stable_streak": float(
            training_config.get("failure_state_stable_streak_reward_scale", 0.0)
        ),
        "backward": directional_scale
        * float(training_config.get("failure_state_backward_cost_weight", 3.0)),
        "lateral": directional_scale
        * float(training_config.get("failure_state_lateral_cost_weight", 0.25)),
        "yaw": directional_scale
        * float(training_config.get("failure_state_yaw_cost_weight", 0.10)),
    }
    residuals = update_source["normalized_signed_constraint_residuals"]

    def projected(value: float, residual: float, lower: float, upper: float) -> float:
        bounded_residual = min(
            max(float(residual), -active.maximum_normalized_update),
            active.maximum_normalized_update,
        )
        return min(max(value + active.dual_learning_rate * bounded_residual, lower), upper)

    updated = {
        "stable_streak": projected(
            current["stable_streak"],
            max(float(residuals["stable"]), float(residuals["maximum_stable_streak"])),
            active.minimum_stable_streak_multiplier,
            active.maximum_stable_streak_multiplier,
        ),
        "backward": projected(
            current["backward"],
            float(residuals["backward"]),
            active.minimum_backward_multiplier,
            active.maximum_directional_multiplier,
        ),
        "lateral": projected(
            current["lateral"],
            float(residuals["lateral"]),
            active.minimum_lateral_multiplier,
            active.maximum_directional_multiplier,
        ),
        "yaw": projected(
            current["yaw"],
            float(residuals["yaw"]),
            active.minimum_yaw_multiplier,
            active.maximum_directional_multiplier,
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_constrained_selection.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "training_report_hash": training["report_hash"],
        "generation_selection_report_hash": normal_selection["report_hash"],
        "failure_state_exam_report_hashes": [
            exams_by_step[step]["report_hash"] for step in training_steps
        ],
        "failure_state_manifest_hash": first_exam["failure_state_manifest_hash"],
        "paired_exam_random_seed": reference_config["random_seed"],
        "paired_exam_checkpoint_steps": training_steps,
        "all_persisted_checkpoints_examined": sorted(exams_by_step) == training_steps,
        "selection_rule": "NORMAL_ROUTE_AND_EXACT_FAILURE_STATE_CONJUNCTION",
        "constraint_update_rule": "PROJECTED_DUAL_ASCENT_FROM_CLOSEST_INFEASIBLE_CHECKPOINT",
        "constraint_evaluations": evaluations,
        "selected_step": selected["step"] if selected is not None else None,
        "development_retention_passed": selected is not None,
        "dual_update_source_step": update_source["step"],
        "current_effective_multipliers": current,
        "updated_effective_multipliers": updated,
        "recommended_next_generation_config_patch": {
            "failure_state_directional_penalty_scale": 1.0,
            "failure_state_backward_cost_weight": updated["backward"],
            "failure_state_lateral_cost_weight": updated["lateral"],
            "failure_state_yaw_cost_weight": updated["yaw"],
            "failure_state_stable_streak_reward_scale": updated["stable_streak"],
        },
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return validate_recovery_mjx_failure_constrained_selection_report(target)


def run_recovery_mjx_probe(
    *,
    scene_xml_path: Path,
    snapshot_manifest_path: Path,
    output_path: Path,
    config: RecoveryMJXProbeConfig | None = None,
    source_checkout_path: Path | None = None,
) -> dict[str, Any]:
    """Run CPU parity and sharded MJX throughput checks on real snapshots."""

    active = config or RecoveryMJXProbeConfig()
    scene_path = scene_xml_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not scene_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("recovery MJX probe inputs are incomplete")
    if target.exists():
        raise ValueError("recovery MJX probe refuses to overwrite evidence")
    if source_checkout_path is not None:
        checkout = source_checkout_path.expanduser().resolve()
        if target == checkout or checkout in target.parents:
            raise ValueError("recovery MJX evidence must remain outside source checkout")

    # Optional dependencies are imported only inside the explicit GPU probe.
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    mujoco = importlib.import_module("mujoco")
    mjx = importlib.import_module("mujoco.mjx")

    snapshots = load_recovery_snapshot_corpus(snapshot_path)
    if not snapshots:
        raise ValueError("recovery MJX probe snapshot corpus is empty")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = active.simulation_dt_sec
    if (model.nq, model.nv, model.nu) != (_QPOS_WIDTH, _QVEL_WIDTH, _JOINT_COUNT):
        raise ValueError("recovery MJX probe requires the OpenTrack G1 29-DoF model")
    model_contract = compiled_mujoco_model_contract(model)
    mjx_model = mjx.put_model(model)
    kp = jnp.asarray(_KPS)
    kd = jnp.asarray(_KDS)
    torque_limit = jnp.asarray(_TORQUE_LIMIT)

    def initialize_one(qpos: Any, qvel: Any) -> Any:
        data = mjx.make_data(mjx_model).replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32),
        )
        return mjx.forward(mjx_model, data)

    def control_step_one(data: Any, target_qpos: Any) -> Any:
        def simulation_step(current: Any, unused: Any) -> tuple[Any, None]:
            del unused
            torque = jnp.clip(
                kp * (target_qpos - current.qpos[7:]) + kd * (-current.qvel[6:]),
                -torque_limit,
                torque_limit,
            )
            return mjx.step(mjx_model, current.replace(ctrl=torque)), None

        result, _ = jax.lax.scan(
            simulation_step,
            data,
            None,
            length=active.substeps,
        )
        return result

    initialize_jit = jax.jit(initialize_one)
    control_step_jit = jax.jit(control_step_one)
    parity: list[dict[str, Any]] = []
    compile_started = time.perf_counter()
    for snapshot in snapshots[: active.parity_snapshot_count]:
        target_qpos = np.asarray(snapshot.qpos[7:], dtype=np.float32)
        cpu_data = mujoco.MjData(model)
        cpu_data.qpos[:] = np.asarray(snapshot.qpos, dtype=np.float64)
        cpu_data.qvel[:] = np.asarray(snapshot.qvel, dtype=np.float64)
        mujoco.mj_forward(model, cpu_data)
        cpu_initial_contacts = int(cpu_data.ncon)
        for _ in range(active.substeps):
            raw_torque = _KPS * (target_qpos - cpu_data.qpos[7:]) + _KDS * (-cpu_data.qvel[6:])
            cpu_data.ctrl[:] = np.clip(raw_torque, -_TORQUE_LIMIT, _TORQUE_LIMIT)
            mujoco.mj_step(model, cpu_data)

        gpu_data = initialize_jit(
            jnp.asarray(snapshot.qpos, dtype=jnp.float32),
            jnp.asarray(snapshot.qvel, dtype=jnp.float32),
        )
        # MJX uses a static contact-buffer shape.  ``ncon`` therefore reflects
        # allocated candidate slots, not the number of penetrating contacts.
        gpu_initial_contacts = int(np.count_nonzero(np.asarray(gpu_data.contact.dist) < 0.0))
        gpu_data = control_step_jit(gpu_data, jnp.asarray(target_qpos))
        jax.block_until_ready(gpu_data.qpos)
        cpu_qpos = np.asarray(cpu_data.qpos, dtype=np.float64)
        cpu_qvel = np.asarray(cpu_data.qvel, dtype=np.float64)
        gpu_qpos = np.asarray(gpu_data.qpos, dtype=np.float64)
        gpu_qvel = np.asarray(gpu_data.qvel, dtype=np.float64)
        finite = bool(
            np.all(np.isfinite(cpu_qpos))
            and np.all(np.isfinite(cpu_qvel))
            and np.all(np.isfinite(gpu_qpos))
            and np.all(np.isfinite(gpu_qvel))
        )
        qpos_rms = float(np.sqrt(np.mean(np.square(cpu_qpos - gpu_qpos))))
        qvel_rms = float(np.sqrt(np.mean(np.square(cpu_qvel - gpu_qvel))))
        parity.append(
            {
                "snapshot_hash": snapshot.snapshot_hash,
                "posture_cluster": snapshot.posture_cluster,
                "finite": finite,
                "qpos_rms_error": qpos_rms,
                "qvel_rms_error": qvel_rms,
                "cpu_initial_contact_count": cpu_initial_contacts,
                "mjx_initial_contact_count": gpu_initial_contacts,
                "cpu_final_contact_count": int(cpu_data.ncon),
                "mjx_final_contact_count": int(
                    np.count_nonzero(np.asarray(gpu_data.contact.dist) < 0.0)
                ),
            }
        )
    parity_compile_and_run_sec = time.perf_counter() - compile_started

    devices = tuple(jax.devices())
    if len(devices) < active.required_gpu_count:
        raise RuntimeError(
            f"recovery MJX probe requires {active.required_gpu_count} GPUs, found {len(devices)}"
        )
    selected_devices = devices[: active.required_gpu_count]
    total_batch = active.required_gpu_count * active.batch_size_per_gpu
    repeated = tuple(snapshots[index % len(snapshots)] for index in range(total_batch))
    qpos_batch = (
        np.stack([item.qpos for item in repeated])
        .astype(np.float32)
        .reshape(
            active.required_gpu_count,
            active.batch_size_per_gpu,
            _QPOS_WIDTH,
        )
    )
    qvel_batch = (
        np.stack([item.qvel for item in repeated])
        .astype(np.float32)
        .reshape(
            active.required_gpu_count,
            active.batch_size_per_gpu,
            _QVEL_WIDTH,
        )
    )
    target_batch = qpos_batch[..., 7:].copy()

    def initialize_device(qpos: Any, qvel: Any) -> Any:
        return jax.vmap(initialize_one)(qpos, qvel)

    def step_device(data: Any, target_qpos: Any) -> Any:
        return jax.vmap(control_step_one)(data, target_qpos)

    initialize_sharded = jax.pmap(initialize_device, devices=selected_devices)
    step_sharded = jax.pmap(step_device, devices=selected_devices)
    batch_compile_started = time.perf_counter()
    sharded_data = initialize_sharded(jnp.asarray(qpos_batch), jnp.asarray(qvel_batch))
    sharded_data = step_sharded(sharded_data, jnp.asarray(target_batch))
    jax.block_until_ready(sharded_data.qpos)
    batch_compile_sec = time.perf_counter() - batch_compile_started
    benchmark_started = time.perf_counter()
    for _ in range(active.benchmark_control_steps):
        sharded_data = step_sharded(sharded_data, jnp.asarray(target_batch))
    jax.block_until_ready(sharded_data.qpos)
    benchmark_sec = time.perf_counter() - benchmark_started
    simulated_steps = total_batch * active.benchmark_control_steps
    control_steps_per_sec = simulated_steps / benchmark_sec
    simulated_realtime_factor = simulated_steps * active.control_dt_sec / benchmark_sec
    finite_batch = bool(
        np.all(np.isfinite(np.asarray(sharded_data.qpos)))
        and np.all(np.isfinite(np.asarray(sharded_data.qvel)))
    )

    maximum_qpos_rms = max(row["qpos_rms_error"] for row in parity)
    maximum_qvel_rms = max(row["qvel_rms_error"] for row in parity)
    parity_passed = bool(
        all(row["finite"] for row in parity)
        and maximum_qpos_rms <= active.maximum_qpos_rms_error_rad
        and maximum_qvel_rms <= active.maximum_qvel_rms_error_rad_s
    )
    throughput_passed = bool(
        finite_batch and control_steps_per_sec >= active.minimum_parallel_control_steps_per_sec
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_probe_report.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "parallelization": "JAX_PMAP_VMAP",
        "device_count": len(selected_devices),
        "devices": [str(device) for device in selected_devices],
        "batch_size_per_device": active.batch_size_per_gpu,
        "total_parallel_environments": total_batch,
        "scene_xml_hash": hash_bytes(scene_path.read_bytes()),
        "compiled_model_contract": model_contract,
        "snapshot_manifest_hash": hash_bytes(snapshot_path.read_bytes()),
        "snapshot_count": len(snapshots),
        "snapshot_source_physics_scene_hash": snapshots[0].physics_scene_hash,
        "snapshot_scene_equivalent": False,
        "cross_scene_transfer_required": True,
        "parity": parity,
        "maximum_qpos_rms_error": maximum_qpos_rms,
        "maximum_qvel_rms_error": maximum_qvel_rms,
        "parity_compile_and_run_sec": parity_compile_and_run_sec,
        "batch_compile_sec": batch_compile_sec,
        "benchmark_sec": benchmark_sec,
        "parallel_control_steps_per_sec": control_steps_per_sec,
        "simulated_realtime_factor": simulated_realtime_factor,
        "finite_parallel_state": finite_batch,
        "parity_passed": parity_passed,
        "throughput_passed": throughput_passed,
        "probe_passed": parity_passed and throughput_passed,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Probe G1 recovery on modern MuJoCo/MJX")
    parser.add_argument("--scene-xml", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument("--batch-size-per-gpu", default=32, type=int)
    parser.add_argument("--benchmark-control-steps", default=24, type=int)
    parser.add_argument("--parity-snapshot-count", default=3, type=int)
    args = parser.parse_args()
    result = run_recovery_mjx_probe(
        scene_xml_path=args.scene_xml,
        snapshot_manifest_path=args.snapshot_manifest,
        output_path=args.output,
        source_checkout_path=args.source_checkout,
        config=RecoveryMJXProbeConfig(
            batch_size_per_gpu=args.batch_size_per_gpu,
            benchmark_control_steps=args.benchmark_control_steps,
            parity_snapshot_count=args.parity_snapshot_count,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "RecoveryMJXFailureConstraintConfig",
    "RecoveryMJXTeacherResidualPPOConfig",
    "RecoveryMJXProbeConfig",
    "RecoveryMJXPPOConfig",
    "build_recovery_mjx_action_distribution_audit",
    "build_recovery_mjx_directional_curriculum",
    "build_recovery_mjx_failure_window_plan",
    "compiled_mujoco_model_contract",
    "run_recovery_mjx_probe",
    "select_recovery_mjx_failure_constrained_generation",
    "select_recovery_mjx_teacher_residual_generation",
    "validate_recovery_mjx_directional_curriculum",
    "validate_recovery_mjx_action_distribution_audit",
    "validate_recovery_mjx_failure_state_manifest",
    "validate_recovery_mjx_failure_state_exam_report",
    "validate_recovery_mjx_failure_constrained_selection_report",
    "validate_recovery_mjx_failure_window_plan",
    "validate_recovery_mjx_probe_report",
    "validate_recovery_mjx_teacher_residual_report",
]
