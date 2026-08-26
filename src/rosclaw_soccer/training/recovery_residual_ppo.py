"""Contracts and numerical helpers for proprioceptive recovery residual PPO.

The frozen episodic skill memory is the stability anchor.  A recurrent actor
may only observe body proprioception plus the *current* internal memory target
error and proposes a bounded joint-position residual.  External reference
phase, teacher identity, future reference states, and hardware authority are
outside this contract.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

RecoveryCurriculumSource = Literal[
    "CAPABILITY_FRONTIER",
    "RECENT_FAILURE",
    "HISTORICAL_ANCHOR",
    "NIGHTMARE",
    "SOCIAL_TEACHER",
]

_SOURCES: tuple[RecoveryCurriculumSource, ...] = (
    "CAPABILITY_FRONTIER",
    "RECENT_FAILURE",
    "HISTORICAL_ANCHOR",
    "NIGHTMARE",
    "SOCIAL_TEACHER",
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_PROPRIO_DIM = 93


@dataclass(frozen=True)
class RecoveryResidualObservationSpec:
    """Deployment observation contract for the residual recovery actor."""

    proprioception_dim: int = _PROPRIO_DIM
    internal_memory_error_dim: int = _JOINT_COUNT
    actor_observation_dim: int = _PROPRIO_DIM + _JOINT_COUNT
    critic_privileged_dim: int = 6
    actor_features: tuple[str, ...] = (
        "projected_gravity_body_3",
        "pelvis_gyro_scaled_3",
        "joint_position_from_default_29",
        "joint_velocity_scaled_29",
        "last_absolute_motor_target_29",
        "current_internal_memory_target_error_29",
    )
    forbidden_actor_features: tuple[str, ...] = (
        "external_reference_joint_position",
        "external_reference_joint_velocity",
        "external_reference_phase",
        "teacher_identity",
        "future_reference_state",
        "ground_truth_successor_value",
    )
    action_semantics: str = "BOUNDED_PD_TARGET_RESIDUAL_AROUND_FROZEN_SKILL_MEMORY"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_residual_observation_spec.v1"

    def __post_init__(self) -> None:
        if (
            self.proprioception_dim != _PROPRIO_DIM
            or self.internal_memory_error_dim != _JOINT_COUNT
            or self.actor_observation_dim != _PROPRIO_DIM + _JOINT_COUNT
            or self.critic_privileged_dim != 6
            or len(self.actor_features) != len(set(self.actor_features))
            or len(self.forbidden_actor_features) != len(set(self.forbidden_actor_features))
            or set(self.actor_features) & set(self.forbidden_actor_features)
            or self.action_semantics != "BOUNDED_PD_TARGET_RESIDUAL_AROUND_FROZEN_SKILL_MEMORY"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery residual observation spec is invalid")

    @property
    def spec_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryResidualPPOConfig:
    """Bounded exploration and PPO settings for one four-rank training run."""

    iterations: int = 160
    rollout_steps: int = 256
    recurrent_chunk_steps: int = 64
    update_epochs: int = 4
    chunks_per_minibatch: int = 2
    hidden_size: int = 192
    encoder_size: int = 192
    learning_rate: float = 2.0e-4
    discount: float = 0.999
    gae_lambda: float = 0.97
    clip_ratio: float = 0.15
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.001
    zero_residual_anchor_coefficient: float = 0.015
    warmstart_retention_coefficient: float = 0.15
    maximum_gradient_norm: float = 0.80
    initial_log_standard_deviation: float = -2.8
    residual_limit_lower_body_rad: float = 0.15
    residual_limit_waist_rad: float = 0.15
    residual_limit_arm_rad: float = 0.18
    residual_filter_fraction: float = 0.25
    maximum_residual_step_rad: float = 0.025
    initial_mismatch_deadband: float = 0.02
    initial_mismatch_full_authority: float = 0.30
    maximum_episode_steps: int = 2_000
    warmstart_optimizer_steps: int = 120
    warmstart_chunk_steps: int = 64
    warmstart_chunks_per_rank: int = 6
    warmstart_learning_rate: float = 3.0e-4
    random_seed: int = 5301
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_residual_ppo_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.discount,
            self.gae_lambda,
            self.clip_ratio,
            self.value_coefficient,
            self.entropy_coefficient,
            self.zero_residual_anchor_coefficient,
            self.warmstart_retention_coefficient,
            self.maximum_gradient_norm,
            self.initial_log_standard_deviation,
            self.residual_limit_lower_body_rad,
            self.residual_limit_waist_rad,
            self.residual_limit_arm_rad,
            self.residual_filter_fraction,
            self.maximum_residual_step_rad,
            self.initial_mismatch_deadband,
            self.initial_mismatch_full_authority,
            self.warmstart_learning_rate,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 1 <= self.iterations <= 100_000
            or not 64 <= self.rollout_steps <= 8_192
            or not 8 <= self.recurrent_chunk_steps <= self.rollout_steps
            or self.rollout_steps % self.recurrent_chunk_steps
            or not 1 <= self.update_epochs <= 16
            or not 1
            <= self.chunks_per_minibatch
            <= self.rollout_steps // self.recurrent_chunk_steps
            or not 32 <= self.hidden_size <= 1_024
            or not 32 <= self.encoder_size <= 1_024
            or not 0.0 < self.learning_rate <= 1e-2
            or not 0.90 <= self.discount < 1.0
            or not 0.80 <= self.gae_lambda <= 1.0
            or not 0.01 <= self.clip_ratio <= 0.40
            or min(
                self.value_coefficient,
                self.entropy_coefficient,
                self.zero_residual_anchor_coefficient,
                self.warmstart_retention_coefficient,
                self.maximum_gradient_norm,
                self.warmstart_learning_rate,
            )
            <= 0.0
            or not -6.0 <= self.initial_log_standard_deviation <= -0.5
            or not 0.01 <= self.residual_limit_lower_body_rad <= 0.20
            or not 0.01 <= self.residual_limit_waist_rad <= 0.15
            or not 0.01 <= self.residual_limit_arm_rad <= 0.25
            or not 0.05 <= self.residual_filter_fraction <= 1.0
            or not 0.005 <= self.maximum_residual_step_rad <= 0.05
            or not 0.0 <= self.initial_mismatch_deadband < 0.10
            or not self.initial_mismatch_deadband < self.initial_mismatch_full_authority <= 1.0
            or not 500 <= self.maximum_episode_steps <= 3_000
            or not 0 <= self.warmstart_optimizer_steps <= 10_000
            or not 8 <= self.warmstart_chunk_steps <= 256
            or not 1 <= self.warmstart_chunks_per_rank <= 64
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery residual PPO config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def residual_limits_rad(self) -> NDArray[np.float32]:
        return np.asarray(
            (self.residual_limit_lower_body_rad,) * 12
            + (self.residual_limit_waist_rad,) * 3
            + (self.residual_limit_arm_rad,) * 14,
            dtype=np.float32,
        )


@dataclass(frozen=True)
class RecoveryRewardConfig:
    potential_progress_scale: float = 8.0
    nominal_tracking_scale: float = 0.002
    stable_step_bonus: float = 0.50
    success_bonus: float = 80.0
    failure_penalty: float = 40.0
    residual_energy_scale: float = 0.015
    residual_delta_scale: float = 0.010
    torque_saturation_scale: float = 0.15
    ready_pelvis_height_m: float = 0.62
    ready_upright_projection: float = 0.75
    maximum_stable_linear_speed_mps: float = 0.50
    maximum_stable_angular_speed_rad_s: float = 1.50
    final_stable_frames: int = 100
    schema_version: str = "rosclaw_soccer.recovery_residual_reward_config.v1"

    def __post_init__(self) -> None:
        values = tuple(
            value
            for key, value in asdict(self).items()
            if key != "schema_version" and not isinstance(value, int)
        )
        if (
            any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values)
            or not 50 <= self.final_stable_frames <= 250
        ):
            raise ValueError("recovery residual reward config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryCurriculumState:
    state_hash: str
    base_snapshot_hash: str
    source: RecoveryCurriculumSource
    difficulty: float

    def __post_init__(self) -> None:
        if (
            not _SHA256.fullmatch(self.state_hash)
            or not _SHA256.fullmatch(self.base_snapshot_hash)
            or self.source not in _SOURCES
            or not math.isfinite(self.difficulty)
            or not 0.0 <= self.difficulty <= 1.0
        ):
            raise ValueError("recovery curriculum state is invalid")


class FailurePrioritizedRecoveryCurriculum:
    """Failure-conditioned sampler with non-starving stability anchors."""

    _SOURCE_WEIGHTS: Mapping[RecoveryCurriculumSource, float] = MappingProxyType(
        {
            "CAPABILITY_FRONTIER": 0.40,
            "RECENT_FAILURE": 0.25,
            "HISTORICAL_ANCHOR": 0.15,
            "NIGHTMARE": 0.10,
            "SOCIAL_TEACHER": 0.10,
        }
    )

    def __init__(
        self,
        states: Sequence[RecoveryCurriculumState],
        *,
        seed: int,
    ) -> None:
        if seed < 0 or not states or len({item.state_hash for item in states}) != len(states):
            raise ValueError("recovery curriculum requires unique states and a valid seed")
        grouped: dict[RecoveryCurriculumSource, list[RecoveryCurriculumState]] = {
            source: [] for source in _SOURCES
        }
        for state in states:
            grouped[state.source].append(state)
        if any(not grouped[source] for source in _SOURCES):
            raise ValueError("recovery curriculum must cover all five sources")
        self._states = tuple(states)
        self._grouped = grouped
        self._attempts = {item.state_hash: 0 for item in states}
        self._successes = {item.state_hash: 0 for item in states}
        self._rng = random.Random(seed)

    @property
    def source_weights(self) -> Mapping[RecoveryCurriculumSource, float]:
        return self._SOURCE_WEIGHTS

    def record(self, state_hash: str, *, succeeded: bool) -> None:
        if state_hash not in self._attempts:
            raise ValueError("recovery curriculum result references an unknown state")
        self._attempts[state_hash] += 1
        self._successes[state_hash] += int(succeeded)

    def sample(self, *, source: RecoveryCurriculumSource | None = None) -> RecoveryCurriculumState:
        if source is None:
            source = self._rng.choices(
                list(_SOURCES),
                weights=[self._SOURCE_WEIGHTS[item] for item in _SOURCES],
                k=1,
            )[0]
        elif source not in _SOURCES:
            raise ValueError("recovery curriculum source is invalid")
        candidates = self._grouped[source]
        weights = [self._state_weight(item) for item in candidates]
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _state_weight(self, state: RecoveryCurriculumState) -> float:
        attempts = self._attempts[state.state_hash]
        successes = self._successes[state.state_hash]
        if attempts == 0:
            return 1.0 + 0.25 * state.difficulty
        rate = successes / attempts
        if state.source == "HISTORICAL_ANCHOR":
            return 1.0 + 2.0 * (1.0 - rate)
        if state.source == "CAPABILITY_FRONTIER":
            return 0.25 + math.exp(-abs(rate - 0.50) / 0.15)
        if state.source in {"RECENT_FAILURE", "NIGHTMARE"}:
            return 0.25 + 2.0 * (1.0 - rate)
        return 0.50 + (1.0 - rate)

    def metrics(self) -> dict[str, Any]:
        by_source: dict[str, dict[str, int | float | None]] = {}
        for source in _SOURCES:
            rows = self._grouped[source]
            attempts = sum(self._attempts[item.state_hash] for item in rows)
            successes = sum(self._successes[item.state_hash] for item in rows)
            by_source[source] = {
                "state_count": len(rows),
                "attempts": attempts,
                "successes": successes,
                "success_rate": None if attempts == 0 else successes / attempts,
            }
        return {"by_source": by_source, "source_weights": dict(self._SOURCE_WEIGHTS)}


def build_recovery_residual_actor_observation(
    *,
    proprioception: NDArray[np.floating[Any]] | Sequence[float],
    internal_memory_target_rad: NDArray[np.floating[Any]] | Sequence[float],
    joint_position_rad: NDArray[np.floating[Any]] | Sequence[float],
    spec: RecoveryResidualObservationSpec | None = None,
) -> NDArray[np.float32]:
    """Append only the current internal skill-memory tracking error."""

    active = spec or RecoveryResidualObservationSpec()
    proprio = np.asarray(proprioception, dtype=np.float32)
    target = np.asarray(internal_memory_target_rad, dtype=np.float32)
    position = np.asarray(joint_position_rad, dtype=np.float32)
    if (
        proprio.shape != (active.proprioception_dim,)
        or target.shape != (_JOINT_COUNT,)
        or position.shape != (_JOINT_COUNT,)
        or not np.all(np.isfinite(proprio))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(position))
    ):
        raise ValueError("recovery residual actor observation inputs are invalid")
    result = np.concatenate((proprio, target - position)).astype(np.float32)
    if result.shape != (active.actor_observation_dim,) or not np.all(np.isfinite(result)):
        raise ValueError("recovery residual actor observation is invalid")
    return np.asarray(result, dtype=np.float32)


def recovery_successor_potential(
    *,
    pelvis_height_m: float,
    upright_projection: float,
    root_linear_speed_mps: float,
    root_angular_speed_rad_s: float,
    config: RecoveryRewardConfig | None = None,
) -> float:
    _ = config or RecoveryRewardConfig()
    values = (
        pelvis_height_m,
        upright_projection,
        root_linear_speed_mps,
        root_angular_speed_rad_s,
    )
    if (
        any(not math.isfinite(value) for value in values)
        or min(pelvis_height_m, root_linear_speed_mps, root_angular_speed_rad_s) < 0.0
    ):
        raise ValueError("recovery successor state is invalid")
    height = float(np.clip((pelvis_height_m - 0.08) / 0.62, 0.0, 1.0))
    upright = float(np.clip((upright_projection + 1.0) / 2.0, 0.0, 1.0))
    speed = math.exp(-0.60 * root_linear_speed_mps - 0.20 * root_angular_speed_rad_s)
    ready_geometry = height * upright
    return 0.55 * height + 0.25 * upright + 0.20 * ready_geometry * speed


def compute_recovery_residual_reward(
    *,
    previous_potential: float,
    current_potential: float,
    nominal_tracking_rmse_rad: float,
    normalized_residual_rms: float,
    normalized_residual_delta_rms: float,
    torque_saturation_fraction: float,
    stable: bool,
    succeeded: bool,
    failed: bool,
    config: RecoveryRewardConfig | None = None,
) -> float:
    active = config or RecoveryRewardConfig()
    values = (
        previous_potential,
        current_potential,
        nominal_tracking_rmse_rad,
        normalized_residual_rms,
        normalized_residual_delta_rms,
        torque_saturation_fraction,
    )
    if (
        any(not math.isfinite(value) for value in values)
        or min(
            nominal_tracking_rmse_rad,
            normalized_residual_rms,
            normalized_residual_delta_rms,
            torque_saturation_fraction,
        )
        < 0.0
        or torque_saturation_fraction > 1.0
        or succeeded
        and failed
    ):
        raise ValueError("recovery residual reward inputs are invalid")
    tracking = math.exp(-4.0 * nominal_tracking_rmse_rad)
    reward = (
        active.potential_progress_scale * (current_potential - previous_potential)
        + active.nominal_tracking_scale * tracking
        + (active.stable_step_bonus if stable else 0.0)
        - active.residual_energy_scale * normalized_residual_rms**2
        - active.residual_delta_scale * normalized_residual_delta_rms**2
        - active.torque_saturation_scale * torque_saturation_fraction
    )
    if succeeded:
        reward += active.success_bonus
    if failed:
        reward -= active.failure_penalty
    return float(reward)


def generalized_advantage_estimate(
    *,
    rewards: NDArray[np.floating[Any]],
    values: NDArray[np.floating[Any]],
    dones: NDArray[np.bool_],
    bootstrap_value: float,
    discount: float,
    gae_lambda: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    reward = np.asarray(rewards, dtype=np.float32)
    value = np.asarray(values, dtype=np.float32)
    done = np.asarray(dones, dtype=np.bool_)
    if (
        reward.ndim != 1
        or value.shape != reward.shape
        or done.shape != reward.shape
        or reward.size == 0
        or not np.all(np.isfinite(reward))
        or not np.all(np.isfinite(value))
        or not math.isfinite(bootstrap_value)
        or not 0.0 < discount <= 1.0
        or not 0.0 < gae_lambda <= 1.0
    ):
        raise ValueError("GAE inputs are invalid")
    advantage = np.zeros_like(reward)
    accumulator = 0.0
    next_value = float(bootstrap_value)
    for index in range(reward.size - 1, -1, -1):
        continuation = 0.0 if bool(done[index]) else 1.0
        delta = float(reward[index]) + discount * next_value * continuation - float(value[index])
        accumulator = delta + discount * gae_lambda * continuation * accumulator
        advantage[index] = accumulator
        next_value = float(value[index])
    returns = advantage + value
    return advantage.astype(np.float32), returns.astype(np.float32)


__all__ = [
    "FailurePrioritizedRecoveryCurriculum",
    "RecoveryCurriculumSource",
    "RecoveryCurriculumState",
    "RecoveryResidualObservationSpec",
    "RecoveryResidualPPOConfig",
    "RecoveryRewardConfig",
    "build_recovery_residual_actor_observation",
    "compute_recovery_residual_reward",
    "generalized_advantage_estimate",
    "recovery_successor_potential",
]
