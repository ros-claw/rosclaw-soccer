"""Backend-neutral multi-step goalkeeper learning contract.

The analytical S9--S11 trainer generated candidates from independent samples.
That is useful for cheap exploration, but it cannot teach impact absorption,
recovery, or a second save.  This module is the stateful contract shared by
physics backends.  Isaac Lab can evaluate it with batched tensors while CPU
MuJoCo can replay the exact same phase and reward semantics with NumPy.

The accumulator does not command a robot and never grants hardware authority.
It only converts causal per-step measurements into bounded learning signals.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json


class GoalkeeperEpisodePhase(IntEnum):
    """Monotonic episode phases used by every physics backend."""

    READY = 0
    FIRST_FLIGHT = 1
    FIRST_IMPACT = 2
    FIRST_RECOVERY = 3
    SECOND_FLIGHT = 4
    SECOND_IMPACT = 5
    COMPLETE = 6
    FAILED = 7


@dataclass(frozen=True)
class GoalkeeperMultiStepConfig:
    """Content-addressed curriculum and safety contract."""

    control_dt_sec: float = 0.02
    episode_duration_sec: float = 5.0
    first_shot_release_sec: float = 0.25
    second_shot_release_sec: float = 2.65
    minimum_pelvis_height_m: float = 0.60
    minimum_upright_projection: float = 0.78
    maximum_recovered_linear_speed_mps: float = 0.28
    maximum_recovered_angular_speed_rad_s: float = 0.55
    recovery_hold_sec: float = 0.24
    contact_bonus: float = 2.0
    hand_contact_bonus: float = 4.0
    true_save_bonus: float = 5.0
    hand_save_bonus: float = 12.0
    second_hand_save_bonus: float = 28.0
    second_save_bonus: float = 8.0
    recovery_bonus: float = 3.0
    recovery_progress_reward_scale: float = 0.0
    recovery_progress_linear_speed_decay: float = 2.0
    recovery_progress_angular_speed_decay: float = 0.50
    reach_reward_scale: float = 0.70
    reach_reward_semantics: str = "STATE_DENSITY"
    hard_height_reach_reward_scale: float = 0.0
    hard_height_reach_threshold_m: float = 1.10
    hard_height_reach_distance_decay: float = 1.25
    hard_height_reach_reward_semantics: str = "POTENTIAL_PROGRESS_ONLY"
    bimanual_reach_reward_scale: float = 1.00
    second_shot_reach_reward_multiplier: float = 1.0
    task_motion_reward_scale: float = 0.0
    task_motion_lateral_standoff_m: float = 0.32
    task_motion_vertical_offset_m: float = 0.35
    task_motion_minimum_pelvis_target_m: float = 0.58
    task_motion_maximum_pelvis_target_m: float = 0.86
    task_motion_distance_decay: float = 3.0
    task_motion_approach_horizon_sec: float = 0.65
    task_motion_arrival_horizon_sec: float = 0.14
    task_motion_minimum_progress_weight: float = 0.25
    task_motion_arrival_bonus_scale: float = 1.50
    task_motion_arrival_readiness_threshold: float = 0.80
    upright_reward_scale: float = 0.20
    action_rate_penalty_scale: float = 0.015
    joint_acceleration_penalty_scale: float = 0.003
    torque_penalty_scale: float = 0.00002
    root_linear_speed_penalty_scale: float = 0.015
    root_angular_speed_penalty_scale: float = 0.030
    root_angular_speed_soft_limit_rad_s: float = 3.50
    root_angular_speed_excess_penalty_scale: float = 0.0
    flight_root_angular_penalty_scale: float = 1.0
    action_magnitude_penalty_scale: float = 0.004
    unsafe_penalty: float = 10.0
    save_then_unsafe_penalty: float = 0.0
    false_contact_penalty: float = 0.50
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_multistep_config.v16"

    def __post_init__(self) -> None:
        positive = (
            self.control_dt_sec,
            self.episode_duration_sec,
            self.first_shot_release_sec,
            self.second_shot_release_sec,
            self.minimum_pelvis_height_m,
            self.minimum_upright_projection,
            self.maximum_recovered_linear_speed_mps,
            self.maximum_recovered_angular_speed_rad_s,
            self.recovery_hold_sec,
            self.contact_bonus,
            self.hand_contact_bonus,
            self.true_save_bonus,
            self.hand_save_bonus,
            self.second_hand_save_bonus,
            self.second_save_bonus,
            self.recovery_bonus,
            self.reach_reward_scale,
            self.bimanual_reach_reward_scale,
            self.second_shot_reach_reward_multiplier,
            self.upright_reward_scale,
            self.action_rate_penalty_scale,
            self.joint_acceleration_penalty_scale,
            self.torque_penalty_scale,
            self.root_linear_speed_penalty_scale,
            self.root_angular_speed_penalty_scale,
            self.flight_root_angular_penalty_scale,
            self.action_magnitude_penalty_scale,
            self.unsafe_penalty,
            self.false_contact_penalty,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("goalkeeper multi-step settings must be finite and positive")
        if not (
            math.isfinite(self.save_then_unsafe_penalty)
            and 0.0 <= self.save_then_unsafe_penalty <= 2_000.0
        ):
            raise ValueError("goalkeeper save-then-unsafe penalty is invalid")
        if not (
            math.isfinite(self.root_angular_speed_soft_limit_rad_s)
            and 0.50 <= self.root_angular_speed_soft_limit_rad_s <= 8.0
            and math.isfinite(self.root_angular_speed_excess_penalty_scale)
            and 0.0 <= self.root_angular_speed_excess_penalty_scale <= 100.0
            and self.flight_root_angular_penalty_scale <= 1.0
        ):
            raise ValueError("goalkeeper root-angular tail penalty is invalid")
        if (
            not math.isfinite(self.recovery_progress_reward_scale)
            or not 0.0 <= self.recovery_progress_reward_scale <= 100.0
            or not math.isfinite(self.recovery_progress_linear_speed_decay)
            or not 0.10 <= self.recovery_progress_linear_speed_decay <= 20.0
            or not math.isfinite(self.recovery_progress_angular_speed_decay)
            or not 0.05 <= self.recovery_progress_angular_speed_decay <= 10.0
        ):
            raise ValueError("goalkeeper recovery-progress reward is invalid")
        if (
            not math.isfinite(self.hard_height_reach_reward_scale)
            or not 0.0 <= self.hard_height_reach_reward_scale <= 10.0
            or not math.isfinite(self.hard_height_reach_threshold_m)
            or not 0.80 <= self.hard_height_reach_threshold_m <= 1.40
            or not math.isfinite(self.hard_height_reach_distance_decay)
            or not 0.50 <= self.hard_height_reach_distance_decay <= 4.0
        ):
            raise ValueError("goalkeeper hard-height reach reward is invalid")
        if self.hard_height_reach_reward_semantics != "POTENTIAL_PROGRESS_ONLY":
            raise ValueError("goalkeeper hard-height reach semantics are invalid")
        if self.reach_reward_semantics not in {"STATE_DENSITY", "POTENTIAL_PROGRESS_ONLY"}:
            raise ValueError("goalkeeper reach reward semantics are invalid")
        if (
            not math.isfinite(self.task_motion_reward_scale)
            or not 0.0 <= self.task_motion_reward_scale <= 20.0
            or not math.isfinite(self.task_motion_lateral_standoff_m)
            or not 0.15 <= self.task_motion_lateral_standoff_m <= 0.60
            or not math.isfinite(self.task_motion_vertical_offset_m)
            or not 0.10 <= self.task_motion_vertical_offset_m <= 0.60
            or not math.isfinite(self.task_motion_minimum_pelvis_target_m)
            or not 0.35 <= self.task_motion_minimum_pelvis_target_m <= 0.75
            or not math.isfinite(self.task_motion_maximum_pelvis_target_m)
            or not 0.70 <= self.task_motion_maximum_pelvis_target_m <= 1.00
            or self.task_motion_minimum_pelvis_target_m >= self.task_motion_maximum_pelvis_target_m
            or not math.isfinite(self.task_motion_distance_decay)
            or not 0.50 <= self.task_motion_distance_decay <= 10.0
            or not math.isfinite(self.task_motion_approach_horizon_sec)
            or not 0.30 <= self.task_motion_approach_horizon_sec <= 1.20
            or not math.isfinite(self.task_motion_arrival_horizon_sec)
            or not 0.05 <= self.task_motion_arrival_horizon_sec <= 0.30
            or self.task_motion_arrival_horizon_sec >= self.task_motion_approach_horizon_sec
            or not math.isfinite(self.task_motion_minimum_progress_weight)
            or not 0.0 <= self.task_motion_minimum_progress_weight <= 0.50
            or not math.isfinite(self.task_motion_arrival_bonus_scale)
            or not 0.0 <= self.task_motion_arrival_bonus_scale <= 5.0
            or not math.isfinite(self.task_motion_arrival_readiness_threshold)
            or not 0.50 <= self.task_motion_arrival_readiness_threshold <= 0.95
        ):
            raise ValueError("goalkeeper task-motion reward is invalid")
        if not self.first_shot_release_sec < self.second_shot_release_sec:
            raise ValueError("goalkeeper second shot must follow the first shot")
        if self.second_shot_release_sec >= self.episode_duration_sec - self.recovery_hold_sec:
            raise ValueError("goalkeeper episode leaves no time for second-save recovery")
        if not 0.0 < self.minimum_upright_projection <= 1.0:
            raise ValueError("goalkeeper upright projection must be in (0, 1]")
        if not 1.0 <= self.second_shot_reach_reward_multiplier <= 3.0:
            raise ValueError("goalkeeper second-shot reach multiplier is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper multi-step training is SIM_ONLY")

    @property
    def recovery_hold_steps(self) -> int:
        return int(math.ceil(self.recovery_hold_sec / self.control_dt_sec))

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class GoalkeeperStepBatch:
    """Causal measurements emitted by one batched physics step.

    Array shapes are ``(N,)`` for scalar channels, ``(N, 3)`` for vectors,
    and ``(N, A)`` for action/joint channels.  ``true_save`` must be computed
    from post-contact ball motion, never from proximity alone.
    """

    time_sec: NDArray[np.float64]
    ball_position_m: NDArray[np.float64]
    ball_velocity_mps: NDArray[np.float64]
    intercept_target_m: NDArray[np.float64]
    left_hand_position_m: NDArray[np.float64]
    right_hand_position_m: NDArray[np.float64]
    pelvis_height_m: NDArray[np.float64]
    root_linear_velocity_mps: NDArray[np.float64]
    root_angular_velocity_rad_s: NDArray[np.float64]
    upright_projection: NDArray[np.float64]
    action: NDArray[np.float64]
    previous_action: NDArray[np.float64]
    joint_acceleration_rad_s2: NDArray[np.float64]
    applied_torque_nm: NDArray[np.float64]
    ball_contact: NDArray[np.bool_]
    hand_contact: NDArray[np.bool_]
    true_save: NDArray[np.bool_]
    shot_index: NDArray[np.int64]
    posture_exception_granted: NDArray[np.bool_] | None = None
    pelvis_position_m: NDArray[np.float64] | None = None

    def validate(self) -> int:
        time = np.asarray(self.time_sec)
        if time.ndim != 1 or time.size == 0:
            raise ValueError("goalkeeper step batch time must have shape (N,)")
        count = int(time.size)
        vectors = {
            "ball_position_m": self.ball_position_m,
            "ball_velocity_mps": self.ball_velocity_mps,
            "intercept_target_m": self.intercept_target_m,
            "left_hand_position_m": self.left_hand_position_m,
            "right_hand_position_m": self.right_hand_position_m,
            "root_linear_velocity_mps": self.root_linear_velocity_mps,
            "root_angular_velocity_rad_s": self.root_angular_velocity_rad_s,
        }
        for name, value in vectors.items():
            array = np.asarray(value)
            if array.shape != (count, 3) or not np.all(np.isfinite(array)):
                raise ValueError(f"goalkeeper step {name} must have shape (N, 3) and be finite")
        scalars = {
            "time_sec": self.time_sec,
            "pelvis_height_m": self.pelvis_height_m,
            "upright_projection": self.upright_projection,
        }
        for name, value in scalars.items():
            array = np.asarray(value)
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"goalkeeper step {name} must have shape (N,) and be finite")
        matrices = {
            "action": self.action,
            "previous_action": self.previous_action,
            "joint_acceleration_rad_s2": self.joint_acceleration_rad_s2,
            "applied_torque_nm": self.applied_torque_nm,
        }
        widths: dict[str, int] = {}
        for name, value in matrices.items():
            array = np.asarray(value)
            if array.ndim != 2 or array.shape[0] != count or not np.all(np.isfinite(array)):
                raise ValueError(f"goalkeeper step {name} must have finite shape (N, A)")
            widths[name] = int(array.shape[1])
        if widths["action"] != widths["previous_action"]:
            raise ValueError("goalkeeper current and previous action widths must match")
        if widths["joint_acceleration_rad_s2"] != widths["applied_torque_nm"]:
            raise ValueError("goalkeeper joint acceleration and torque widths must match")
        for name, event_value in {
            "ball_contact": self.ball_contact,
            "hand_contact": self.hand_contact,
            "true_save": self.true_save,
            "shot_index": self.shot_index,
        }.items():
            if np.asarray(event_value).shape != (count,):
                raise ValueError(f"goalkeeper step {name} must have shape (N,)")
        if self.posture_exception_granted is not None:
            exception = np.asarray(self.posture_exception_granted)
            if exception.shape != (count,) or not np.issubdtype(exception.dtype, np.bool_):
                raise ValueError("goalkeeper posture exception channel must be boolean shape (N,)")
        if self.pelvis_position_m is not None:
            pelvis_position = np.asarray(self.pelvis_position_m)
            if pelvis_position.shape != (count, 3) or not np.all(np.isfinite(pelvis_position)):
                raise ValueError("goalkeeper pelvis position channel must have finite shape (N, 3)")
        shot = np.asarray(self.shot_index)
        if not np.issubdtype(shot.dtype, np.integer) or np.any((shot < 0) | (shot > 2)):
            raise ValueError("goalkeeper shot index must be integer 0, 1, or 2")
        for name, event_value in {
            "ball_contact": self.ball_contact,
            "hand_contact": self.hand_contact,
            "true_save": self.true_save,
        }.items():
            if not np.issubdtype(np.asarray(event_value).dtype, np.bool_):
                raise ValueError(f"goalkeeper {name} channel must be boolean")
        if np.any(np.asarray(self.hand_contact) & ~np.asarray(self.ball_contact)):
            raise ValueError("goalkeeper hand contact must also be a robot-ball contact")
        if np.any(np.abs(np.asarray(self.upright_projection)) > 1.0 + 1e-9):
            raise ValueError("goalkeeper upright projection must remain in [-1, 1]")
        return count


@dataclass(frozen=True)
class GoalkeeperRewardBatch:
    total: NDArray[np.float64]
    reach: NDArray[np.float64]
    bimanual_reach: NDArray[np.float64]
    task_motion: NDArray[np.float64]
    upright: NDArray[np.float64]
    recovery_progress: NDArray[np.float64]
    smoothness_penalty: NDArray[np.float64]
    action_rate_penalty: NDArray[np.float64]
    joint_acceleration_penalty: NDArray[np.float64]
    root_linear_speed_penalty: NDArray[np.float64]
    root_angular_speed_penalty: NDArray[np.float64]
    root_angular_excess_penalty: NDArray[np.float64]
    action_magnitude_penalty: NDArray[np.float64]
    effort_penalty: NDArray[np.float64]
    event_bonus: NDArray[np.float64]
    safety_penalty: NDArray[np.float64]
    phase: NDArray[np.int64]
    terminated: NDArray[np.bool_]
    first_save: NDArray[np.bool_]
    first_hand_save: NDArray[np.bool_]
    recovered_after_first: NDArray[np.bool_]
    second_attempt_save: NDArray[np.bool_]
    second_attempt_hand_save: NDArray[np.bool_]
    second_save: NDArray[np.bool_]
    second_hand_save: NDArray[np.bool_]


class GoalkeeperMultiStepAccumulator:
    """Stateful vector reward and phase machine for long-horizon physics RL."""

    def __init__(self, environment_count: int, config: GoalkeeperMultiStepConfig | None = None):
        if not 1 <= environment_count <= 262_144:
            raise ValueError("goalkeeper environment count is outside [1, 262144]")
        self.environment_count = environment_count
        self.config = config or GoalkeeperMultiStepConfig()
        self.phase = np.full(environment_count, GoalkeeperEpisodePhase.READY, dtype=np.int64)
        self.first_contact = np.zeros(environment_count, dtype=np.bool_)
        self.first_save = np.zeros(environment_count, dtype=np.bool_)
        self.first_hand_save = np.zeros(environment_count, dtype=np.bool_)
        self.recovered_after_first = np.zeros(environment_count, dtype=np.bool_)
        self.second_contact = np.zeros(environment_count, dtype=np.bool_)
        self.second_attempt_save = np.zeros(environment_count, dtype=np.bool_)
        self.second_attempt_hand_save = np.zeros(environment_count, dtype=np.bool_)
        self.second_save = np.zeros(environment_count, dtype=np.bool_)
        self._previous_contact = np.zeros(environment_count, dtype=np.bool_)
        self.second_hand_save = np.zeros(environment_count, dtype=np.bool_)
        self._previous_hand_contact = np.zeros(environment_count, dtype=np.bool_)
        self._previous_save = np.zeros(environment_count, dtype=np.bool_)
        self._stable_steps = np.zeros(environment_count, dtype=np.int64)
        self._bimanual_reach_steps = np.zeros(environment_count, dtype=np.int64)
        self._active_flight_steps = np.zeros(environment_count, dtype=np.int64)
        self._previous_hard_height_potential = np.zeros(environment_count, dtype=np.float64)
        self._previous_reach_potential = np.zeros(environment_count, dtype=np.float64)
        self._previous_bimanual_potential = np.zeros(environment_count, dtype=np.float64)
        self._previous_task_motion_potential = np.zeros(environment_count, dtype=np.float64)
        self._previous_task_motion_arrival_window = np.zeros(environment_count, dtype=np.bool_)
        self._previous_recovery_potential = np.zeros(environment_count, dtype=np.float64)
        self._previous_shot = np.zeros(environment_count, dtype=np.int64)

    def reset(self, environment_ids: NDArray[np.int64] | None = None) -> None:
        ids = (
            np.arange(self.environment_count, dtype=np.int64)
            if environment_ids is None
            else np.asarray(environment_ids, dtype=np.int64)
        )
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= self.environment_count)):
            raise ValueError("goalkeeper reset ids are out of range")
        self.phase[ids] = GoalkeeperEpisodePhase.READY
        self.first_contact[ids] = False
        self.first_save[ids] = False
        self.first_hand_save[ids] = False
        self.recovered_after_first[ids] = False
        self.second_contact[ids] = False
        self.second_attempt_save[ids] = False
        self.second_attempt_hand_save[ids] = False
        self.second_save[ids] = False
        self._previous_contact[ids] = False
        self.second_hand_save[ids] = False
        self._previous_hand_contact[ids] = False
        self._previous_save[ids] = False
        self._stable_steps[ids] = 0
        self._bimanual_reach_steps[ids] = 0
        self._active_flight_steps[ids] = 0
        self._previous_hard_height_potential[ids] = 0.0
        self._previous_reach_potential[ids] = 0.0
        self._previous_bimanual_potential[ids] = 0.0
        self._previous_task_motion_potential[ids] = 0.0
        self._previous_task_motion_arrival_window[ids] = False
        self._previous_recovery_potential[ids] = 0.0
        self._previous_shot[ids] = 0

    def step(self, sample: GoalkeeperStepBatch) -> GoalkeeperRewardBatch:
        if sample.validate() != self.environment_count:
            raise ValueError("goalkeeper step batch does not match accumulator environment count")
        cfg = self.config
        time = np.asarray(sample.time_sec, dtype=np.float64)
        contact = np.asarray(sample.ball_contact, dtype=np.bool_)
        hand_contact = np.asarray(sample.hand_contact, dtype=np.bool_)
        save = np.asarray(sample.true_save, dtype=np.bool_)
        shot = np.asarray(sample.shot_index, dtype=np.int64)
        new_contact = contact & ~self._previous_contact
        new_hand_contact = hand_contact & ~self._previous_hand_contact
        new_save = save & ~self._previous_save

        posture_unsafe = (np.asarray(sample.pelvis_height_m) < cfg.minimum_pelvis_height_m) | (
            np.asarray(sample.upright_projection) < cfg.minimum_upright_projection
        )
        posture_exception = (
            np.zeros(self.environment_count, dtype=np.bool_)
            if sample.posture_exception_granted is None
            else np.asarray(sample.posture_exception_granted, dtype=np.bool_)
        )
        unsafe = posture_unsafe & ~posture_exception
        active = ~np.isin(
            self.phase,
            (GoalkeeperEpisodePhase.COMPLETE, GoalkeeperEpisodePhase.FAILED),
        )
        self.phase[active & (shot == 1)] = GoalkeeperEpisodePhase.FIRST_FLIGHT
        self.phase[active & (shot == 2)] = GoalkeeperEpisodePhase.SECOND_FLIGHT

        first_contact_event = active & new_contact & (shot == 1)
        first_save_event = active & new_save & (shot == 1)
        first_hand_save_event = first_save_event & hand_contact
        # Learn every second-shot outcome, then keep the stricter consecutive
        # double-save objective conditional on a stable first-save recovery.
        # Previously only the conditional event was observable, so a genuine
        # second save after a first-shot miss supplied no learning signal.
        second_attempt_contact_event = active & new_contact & (shot == 2)
        second_attempt_save_event = active & new_save & (shot == 2)
        second_attempt_hand_save_event = second_attempt_save_event & hand_contact
        second_eligible = active & self.recovered_after_first & (shot == 2)
        second_contact_event = second_eligible & new_contact
        second_save_event = second_eligible & new_save
        second_hand_save_event = second_save_event & hand_contact
        self.first_contact |= first_contact_event
        self.first_save |= first_save_event
        self.first_hand_save |= first_hand_save_event
        self.second_attempt_save |= second_attempt_save_event
        self.second_attempt_hand_save |= second_attempt_hand_save_event
        self.second_contact |= second_contact_event
        self.second_save |= second_save_event
        self.second_hand_save |= second_hand_save_event
        self.phase[first_contact_event | first_save_event] = GoalkeeperEpisodePhase.FIRST_IMPACT
        self.phase[second_attempt_contact_event | second_attempt_save_event] = (
            GoalkeeperEpisodePhase.SECOND_IMPACT
        )

        recovering = (
            active & self.first_save & ~self.recovered_after_first & ~first_save_event & (shot != 2)
        )
        root_speed = np.linalg.norm(np.asarray(sample.root_linear_velocity_mps), axis=1)
        angular_speed = np.linalg.norm(np.asarray(sample.root_angular_velocity_rad_s), axis=1)
        stable = (
            recovering
            & ~posture_unsafe
            & (root_speed <= cfg.maximum_recovered_linear_speed_mps)
            & (angular_speed <= cfg.maximum_recovered_angular_speed_rad_s)
        )
        self._stable_steps = np.where(stable, self._stable_steps + 1, 0)
        just_recovered = (
            recovering
            & ~self.recovered_after_first
            & (self._stable_steps >= cfg.recovery_hold_steps)
        )
        self.recovered_after_first |= just_recovered
        self.phase[recovering & ~just_recovered] = GoalkeeperEpisodePhase.FIRST_RECOVERY

        left_hand = np.asarray(sample.left_hand_position_m)
        right_hand = np.asarray(sample.right_hand_position_m)
        target = np.asarray(sample.intercept_target_m)
        left_distance = np.linalg.norm(left_hand - target, axis=1)
        right_distance = np.linalg.norm(right_hand - target, axis=1)
        hand_distance = np.minimum(left_distance, right_distance)
        phase_reach_multiplier = np.where(shot == 2, cfg.second_shot_reach_reward_multiplier, 1.0)
        reach_potential = (shot > 0).astype(np.float64) * np.exp(-4.0 * np.square(hand_distance))
        continued_flight = (shot > 0) & (shot == self._previous_shot)
        reach_signal = (
            np.where(
                continued_flight,
                reach_potential - self._previous_reach_potential,
                0.0,
            )
            if cfg.reach_reward_semantics == "POTENTIAL_PROGRESS_ONLY"
            else reach_potential
        )
        reach = cfg.reach_reward_scale * phase_reach_multiplier * reach_signal
        hard_height = (shot > 0) & (target[:, 2] >= cfg.hard_height_reach_threshold_m)
        hard_potential = hard_height.astype(np.float64) * np.exp(
            -cfg.hard_height_reach_distance_decay * np.square(hand_distance)
        )
        continued_shot = hard_height & (shot == self._previous_shot)
        hard_progress = np.where(
            continued_shot,
            hard_potential - self._previous_hard_height_potential,
            0.0,
        )
        reach += cfg.hard_height_reach_reward_scale * phase_reach_multiplier * hard_progress
        hand_midpoint = 0.5 * (left_hand + right_hand)
        bilateral_opportunity = (
            (shot > 0)
            & (np.abs(target[:, 1] - hand_midpoint[:, 1]) <= 0.42)
            & (target[:, 2] >= 0.62)
        )
        bimanual_distance = np.maximum(left_distance, right_distance)
        bimanual_potential = bilateral_opportunity.astype(np.float64) * np.exp(
            -3.0 * np.square(bimanual_distance)
        )
        bimanual_signal = (
            np.where(
                continued_flight,
                bimanual_potential - self._previous_bimanual_potential,
                0.0,
            )
            if cfg.reach_reward_semantics == "POTENTIAL_PROGRESS_ONLY"
            else bimanual_potential
        )
        bimanual_reach = cfg.bimanual_reach_reward_scale * phase_reach_multiplier * bimanual_signal
        if cfg.task_motion_reward_scale > 0.0 and sample.pelvis_position_m is None:
            raise ValueError("goalkeeper task-motion reward requires pelvis position")
        if sample.pelvis_position_m is None:
            task_motion_potential = np.zeros(self.environment_count, dtype=np.float64)
        else:
            pelvis_position = np.asarray(sample.pelvis_position_m, dtype=np.float64)
            lateral_remaining = np.maximum(
                np.abs(target[:, 1] - pelvis_position[:, 1]) - cfg.task_motion_lateral_standoff_m,
                0.0,
            )
            pelvis_target_height = np.clip(
                target[:, 2] + cfg.task_motion_vertical_offset_m,
                cfg.task_motion_minimum_pelvis_target_m,
                cfg.task_motion_maximum_pelvis_target_m,
            )
            task_motion_error_sq = np.square(lateral_remaining) + np.square(
                pelvis_position[:, 2] - pelvis_target_height
            )
            task_motion_potential = (shot > 0).astype(np.float64) * np.exp(
                -cfg.task_motion_distance_decay * task_motion_error_sq
            )
        task_motion_progress = np.where(
            continued_flight,
            task_motion_potential - self._previous_task_motion_potential,
            0.0,
        )
        ball_position = np.asarray(sample.ball_position_m, dtype=np.float64)
        ball_velocity = np.asarray(sample.ball_velocity_mps, dtype=np.float64)
        forward_velocity = ball_velocity[:, 0]
        valid_arrival_clock = (shot > 0) & (np.abs(forward_velocity) >= 0.10)
        time_to_intercept = np.divide(
            target[:, 0] - ball_position[:, 0],
            forward_velocity,
            out=np.full(self.environment_count, np.inf, dtype=np.float64),
            where=valid_arrival_clock,
        )
        valid_arrival_clock &= time_to_intercept >= 0.0
        arrival_progress = np.where(
            valid_arrival_clock,
            np.clip(
                1.0 - time_to_intercept / cfg.task_motion_approach_horizon_sec,
                0.0,
                1.0,
            ),
            0.0,
        )
        progress_weight = (
            cfg.task_motion_minimum_progress_weight
            + (1.0 - cfg.task_motion_minimum_progress_weight) * arrival_progress
        )
        arrival_window = valid_arrival_clock & (
            time_to_intercept <= cfg.task_motion_arrival_horizon_sec
        )
        arrival_event = arrival_window & ~self._previous_task_motion_arrival_window
        arrival_coupling = np.sqrt(np.clip(task_motion_potential * reach_potential, 0.0, 1.0))
        qualified_arrival_coupling = np.clip(
            (arrival_coupling - cfg.task_motion_arrival_readiness_threshold)
            / (1.0 - cfg.task_motion_arrival_readiness_threshold),
            0.0,
            1.0,
        )
        task_motion = (
            cfg.task_motion_reward_scale
            * phase_reach_multiplier
            * (
                progress_weight * task_motion_progress
                + cfg.task_motion_arrival_bonus_scale
                * arrival_event.astype(np.float64)
                * qualified_arrival_coupling
            )
        )
        self._active_flight_steps += (shot > 0).astype(np.int64)
        self._bimanual_reach_steps += (bilateral_opportunity & (bimanual_distance <= 0.42)).astype(
            np.int64
        )
        upright = cfg.upright_reward_scale * np.clip(
            (np.asarray(sample.upright_projection) - cfg.minimum_upright_projection)
            / (1.0 - cfg.minimum_upright_projection),
            0.0,
            1.0,
        )
        normalized_upright = np.clip(
            (np.asarray(sample.upright_projection) - cfg.minimum_upright_projection)
            / (1.0 - cfg.minimum_upright_projection),
            0.0,
            1.0,
        )
        recovery_potential = normalized_upright * np.exp(
            -cfg.recovery_progress_linear_speed_decay * np.square(root_speed)
            - cfg.recovery_progress_angular_speed_decay * np.square(angular_speed)
        )
        recovery_progress = cfg.recovery_progress_reward_scale * np.where(
            recovering,
            recovery_potential - self._previous_recovery_potential,
            0.0,
        )
        action_rate = np.mean(
            np.square(np.asarray(sample.action) - np.asarray(sample.previous_action)), axis=1
        )
        acceleration = np.mean(np.square(np.asarray(sample.joint_acceleration_rad_s2)), axis=1)
        torque = np.mean(np.square(np.asarray(sample.applied_torque_nm)), axis=1)
        root_linear_speed = np.sum(np.square(np.asarray(sample.root_linear_velocity_mps)), axis=1)
        root_angular_speed = np.sum(
            np.square(np.asarray(sample.root_angular_velocity_rad_s)), axis=1
        )
        action_magnitude = np.mean(np.square(np.asarray(sample.action)), axis=1)
        action_rate_penalty = cfg.action_rate_penalty_scale * action_rate
        joint_acceleration_penalty = cfg.joint_acceleration_penalty_scale * acceleration
        root_linear_speed_penalty = cfg.root_linear_speed_penalty_scale * root_linear_speed
        # The environment-owned posture exception is a bounded dynamic-skill
        # clock spanning anticipation, flight and initial landing.  Outside
        # that clock, recovery keeps the full angular stability cost.
        angular_phase_scale = np.where(
            posture_exception,
            cfg.flight_root_angular_penalty_scale,
            1.0,
        )
        root_angular_speed_penalty = (
            angular_phase_scale * cfg.root_angular_speed_penalty_scale * root_angular_speed
        )
        root_angular_excess = np.maximum(
            np.sqrt(root_angular_speed) - cfg.root_angular_speed_soft_limit_rad_s,
            0.0,
        )
        root_angular_excess_penalty = (
            angular_phase_scale
            * cfg.root_angular_speed_excess_penalty_scale
            * np.square(root_angular_excess)
        )
        action_magnitude_penalty = cfg.action_magnitude_penalty_scale * action_magnitude
        smoothness_penalty = (
            action_rate_penalty
            + joint_acceleration_penalty
            + root_linear_speed_penalty
            + root_angular_speed_penalty
            + root_angular_excess_penalty
            + action_magnitude_penalty
        )
        effort_penalty = cfg.torque_penalty_scale * torque
        false_contact = (
            new_contact & ~save & (np.linalg.norm(sample.ball_velocity_mps, axis=1) < 0.10)
        )
        event_bonus = cfg.contact_bonus * new_contact.astype(np.float64)
        event_bonus += cfg.hand_contact_bonus * new_hand_contact.astype(np.float64)
        event_bonus += cfg.true_save_bonus * first_save_event.astype(np.float64)
        event_bonus += cfg.hand_save_bonus * first_hand_save_event.astype(np.float64)
        event_bonus += cfg.true_save_bonus * second_attempt_save_event.astype(np.float64)
        event_bonus += cfg.hand_save_bonus * second_attempt_hand_save_event.astype(np.float64)
        event_bonus += cfg.second_save_bonus * second_save_event.astype(np.float64)
        event_bonus += cfg.second_hand_save_bonus * second_hand_save_event.astype(np.float64)
        event_bonus += cfg.recovery_bonus * just_recovered.astype(np.float64)
        event_bonus -= cfg.false_contact_penalty * false_contact.astype(np.float64)
        safety_penalty = cfg.unsafe_penalty * unsafe.astype(np.float64)
        safety_penalty += cfg.save_then_unsafe_penalty * (unsafe & self.first_save).astype(
            np.float64
        )

        timed_out = time >= cfg.episode_duration_sec
        self.phase[unsafe & active] = GoalkeeperEpisodePhase.FAILED
        # Terminal phases are monotonic.  A physics backend may quarantine a
        # failed world by restoring a finite, upright posture; that recovery
        # must not launder the recorded failure into COMPLETE at timeout.
        complete = timed_out & ~unsafe & active
        self.phase[complete] = GoalkeeperEpisodePhase.COMPLETE
        terminated = unsafe | timed_out
        total = (
            reach
            + bimanual_reach
            + task_motion
            + upright
            + recovery_progress
            + event_bonus
            - smoothness_penalty
            - effort_penalty
            - safety_penalty
        )
        self._previous_contact = contact.copy()
        self._previous_hand_contact = hand_contact.copy()
        self._previous_save = save.copy()
        self._previous_hard_height_potential = np.where(shot > 0, hard_potential, 0.0)
        self._previous_reach_potential = np.where(shot > 0, reach_potential, 0.0)
        self._previous_bimanual_potential = np.where(shot > 0, bimanual_potential, 0.0)
        self._previous_task_motion_potential = np.where(shot > 0, task_motion_potential, 0.0)
        self._previous_task_motion_arrival_window = np.where(shot > 0, arrival_window, False)
        self._previous_recovery_potential = recovery_potential
        self._previous_shot = shot.copy()
        return GoalkeeperRewardBatch(
            total=np.asarray(total, dtype=np.float64),
            reach=np.asarray(reach, dtype=np.float64),
            bimanual_reach=np.asarray(bimanual_reach, dtype=np.float64),
            task_motion=np.asarray(task_motion, dtype=np.float64),
            upright=np.asarray(upright, dtype=np.float64),
            recovery_progress=np.asarray(recovery_progress, dtype=np.float64),
            smoothness_penalty=np.asarray(smoothness_penalty, dtype=np.float64),
            action_rate_penalty=np.asarray(action_rate_penalty, dtype=np.float64),
            joint_acceleration_penalty=np.asarray(joint_acceleration_penalty, dtype=np.float64),
            root_linear_speed_penalty=np.asarray(root_linear_speed_penalty, dtype=np.float64),
            root_angular_speed_penalty=np.asarray(root_angular_speed_penalty, dtype=np.float64),
            root_angular_excess_penalty=np.asarray(root_angular_excess_penalty, dtype=np.float64),
            action_magnitude_penalty=np.asarray(action_magnitude_penalty, dtype=np.float64),
            effort_penalty=np.asarray(effort_penalty, dtype=np.float64),
            event_bonus=np.asarray(event_bonus, dtype=np.float64),
            safety_penalty=np.asarray(safety_penalty, dtype=np.float64),
            phase=self.phase.copy(),
            terminated=np.asarray(terminated, dtype=np.bool_),
            first_save=self.first_save.copy(),
            first_hand_save=self.first_hand_save.copy(),
            recovered_after_first=self.recovered_after_first.copy(),
            second_attempt_save=self.second_attempt_save.copy(),
            second_attempt_hand_save=self.second_attempt_hand_save.copy(),
            second_save=self.second_save.copy(),
            second_hand_save=self.second_hand_save.copy(),
        )

    def summary(self) -> dict[str, Any]:
        """Return explicit training outcomes without implying promotion."""

        return {
            "schema_version": "rosclaw_soccer.goalkeeper_multistep_summary.v6",
            "config_hash": self.config.config_hash,
            "environment_count": self.environment_count,
            "first_contact_rate": float(np.mean(self.first_contact)),
            "first_save_rate": float(np.mean(self.first_save)),
            "first_hand_save_rate": float(np.mean(self.first_hand_save)),
            "recovery_rate": float(np.mean(self.recovered_after_first)),
            "second_attempt_save_rate": float(np.mean(self.second_attempt_save)),
            "second_attempt_hand_save_rate": float(np.mean(self.second_attempt_hand_save)),
            "second_contact_rate": float(np.mean(self.second_contact)),
            "second_save_rate": float(np.mean(self.second_save)),
            "second_hand_save_rate": float(np.mean(self.second_hand_save)),
            "failed_rate": float(np.mean(self.phase == GoalkeeperEpisodePhase.FAILED)),
            "bimanual_reach_fraction": float(
                np.sum(self._bimanual_reach_steps) / max(1, int(np.sum(self._active_flight_steps)))
            ),
            "promotion_status": "TRAINING_METRICS_ONLY_NOT_PROMOTED",
            "activation_ceiling": "SIM_ONLY",
        }


__all__ = [
    "GoalkeeperEpisodePhase",
    "GoalkeeperMultiStepAccumulator",
    "GoalkeeperMultiStepConfig",
    "GoalkeeperRewardBatch",
    "GoalkeeperStepBatch",
]
