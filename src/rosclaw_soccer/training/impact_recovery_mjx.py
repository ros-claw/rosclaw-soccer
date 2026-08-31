"""Goal-conditioned MJX learning for post-impact whole-body recovery.

This training-host-only module turns the content-bound full-chain curriculum
into a direct 29-joint PD residual policy.  The actor observes deployable body
state plus the desired field heading; failed episodes are reset states while
successful episodes provide immutable motion-memory targets.  MJX supplies
rollout throughput only.  Promotion still requires a separate CPU-MuJoCo
full-chain exam.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo_train
from mujoco import mjx

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_curriculum import (
    validate_impact_recovery_curriculum,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.recovery_mjx import (
    _KDS,
    _KPS,
    _TORQUE_LIMIT,
    _atomic_json,
    compiled_mujoco_model_contract,
)

ResetPopulation = Literal["MIXED", "RETENTION", "ACQUISITION"]
LearningStage = Literal["BALANCE", "GOAL_READY"]
RetentionMemoryMode = Literal["BLENDED", "DIRECT_REPLAY"]
GainMemoryMode = Literal["FIXED", "DYNAMIC"]
ResidualGateMode = Literal["NONE", "TEACHER_NOVELTY"]

_G1_QPOS_WIDTH = 36
_G1_QVEL_WIDTH = 35
_JOINT_COUNT = 29
_FRAME_DIM_BASE = 131
_GAIN_CONTEXT_DIM = 58
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImpactRecoveryMJXConfig:
    """Bounded four-GPU recovery learner configuration."""

    total_timesteps: int = 2_097_152
    num_envs: int = 256
    episode_length: int = 250
    unroll_length: int = 25
    batch_size: int = 64
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    num_evals: int = 5
    num_eval_envs: int = 64
    learning_rate: float = 1.0e-4
    entropy_cost: float = 2.0e-4
    discounting: float = 0.995
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.15
    maximum_gradient_norm: float = 0.75
    learning_stage: LearningStage = "BALANCE"
    retention_memory_mode: RetentionMemoryMode = "BLENDED"
    gain_memory_mode: GainMemoryMode = "FIXED"
    residual_gate_mode: ResidualGateMode = "NONE"
    novelty_gate_lower: float = 0.08
    novelty_gate_upper: float = 0.30
    residual_authority_steps: int = 0
    history_frames: int = 4
    acquisition_reset_fraction: float = 0.60
    residual_limit_lower_body_rad: float = 0.18
    residual_limit_waist_rad: float = 0.10
    residual_limit_arm_rad: float = 0.15
    maximum_target_step_rad: float = 0.035
    memory_blend_steps: int = 75
    joint_position_noise_rad: float = 0.015
    joint_velocity_noise_rad_s: float = 0.04
    root_linear_velocity_noise_mps: float = 0.04
    root_angular_velocity_noise_rad_s: float = 0.08
    ready_pelvis_height_m: float = 0.70
    ready_upright_projection: float = 0.95
    ready_linear_speed_mps: float = 0.18
    ready_angular_speed_rad_s: float = 0.45
    ready_heading_error_rad: float = 0.25
    ready_foot_height_m: float = 0.055
    success_stable_steps: int = 25
    residual_penalty_scale: float = 0.025
    action_delta_penalty_scale: float = 0.04
    tracking_penalty_scale: float = 0.006
    torque_saturation_penalty_scale: float = 0.12
    drift_penalty_scale: float = 0.04
    soft_balance_reward_scale: float = 0.80
    linear_motion_penalty_scale: float = 0.15
    angular_motion_penalty_scale: float = 0.12
    support_penalty_scale: float = 0.08
    height_penalty_scale: float = 0.12
    upright_penalty_scale: float = 0.12
    required_gpu_count: int = 4
    random_seed: int = 5711
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_mjx_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.entropy_cost,
            self.discounting,
            self.gae_lambda,
            self.clipping_epsilon,
            self.maximum_gradient_norm,
            self.acquisition_reset_fraction,
            self.novelty_gate_lower,
            self.novelty_gate_upper,
            self.residual_limit_lower_body_rad,
            self.residual_limit_waist_rad,
            self.residual_limit_arm_rad,
            self.maximum_target_step_rad,
            self.joint_position_noise_rad,
            self.joint_velocity_noise_rad_s,
            self.root_linear_velocity_noise_mps,
            self.root_angular_velocity_noise_rad_s,
            self.ready_pelvis_height_m,
            self.ready_upright_projection,
            self.ready_linear_speed_mps,
            self.ready_angular_speed_rad_s,
            self.ready_heading_error_rad,
            self.ready_foot_height_m,
            self.residual_penalty_scale,
            self.action_delta_penalty_scale,
            self.tracking_penalty_scale,
            self.torque_saturation_penalty_scale,
            self.drift_penalty_scale,
            self.soft_balance_reward_scale,
            self.linear_motion_penalty_scale,
            self.angular_motion_penalty_scale,
            self.support_penalty_scale,
            self.height_penalty_scale,
            self.upright_penalty_scale,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 65_536 <= self.total_timesteps <= 1_000_000_000
            or not 32 <= self.num_envs <= 65_536
            or not 1 <= self.required_gpu_count <= 16
            or self.num_envs % self.required_gpu_count
            or not 100 <= self.episode_length <= 1_000
            or not 8 <= self.unroll_length <= 128
            or self.episode_length % self.unroll_length
            or not 8 <= self.batch_size <= 8_192
            or not 1 <= self.num_minibatches <= 64
            or (self.batch_size * self.num_minibatches) % self.num_envs
            or not 1 <= self.num_updates_per_batch <= 16
            or not 2 <= self.num_evals <= 50
            or not 4 <= self.num_eval_envs <= 1_024
            or self.num_eval_envs % self.required_gpu_count
            or not 0.0 < self.learning_rate <= 1.0e-2
            or not 0.0 <= self.entropy_cost <= 0.1
            or not 0.90 <= self.discounting < 1.0
            or not 0.80 <= self.gae_lambda <= 1.0
            or not 0.05 <= self.clipping_epsilon <= 0.40
            or not 0.0 < self.maximum_gradient_norm <= 10.0
            or self.learning_stage not in {"BALANCE", "GOAL_READY"}
            or self.retention_memory_mode not in {"BLENDED", "DIRECT_REPLAY"}
            or self.gain_memory_mode not in {"FIXED", "DYNAMIC"}
            or self.residual_gate_mode not in {"NONE", "TEACHER_NOVELTY"}
            or (self.residual_gate_mode == "TEACHER_NOVELTY" and self.gain_memory_mode != "DYNAMIC")
            or not 0.0 <= self.novelty_gate_lower < self.novelty_gate_upper <= 2.0
            or not (
                self.residual_authority_steps == 0
                or 10 <= self.residual_authority_steps <= self.episode_length
            )
            or not 1 <= self.history_frames <= 8
            or not 0.25 <= self.acquisition_reset_fraction <= 0.90
            or not 0.01 <= self.residual_limit_lower_body_rad <= 0.20
            or not 0.01 <= self.residual_limit_waist_rad <= 0.15
            or not 0.01 <= self.residual_limit_arm_rad <= 0.25
            or not 0.005 <= self.maximum_target_step_rad <= 0.08
            or not 25 <= self.memory_blend_steps <= self.episode_length
            or not 0.0 <= self.joint_position_noise_rad <= 0.10
            or not 0.0 <= self.joint_velocity_noise_rad_s <= 0.50
            or not 0.0 <= self.root_linear_velocity_noise_mps <= 0.30
            or not 0.0 <= self.root_angular_velocity_noise_rad_s <= 0.75
            or not 0.60 <= self.ready_pelvis_height_m <= 0.82
            or not 0.75 <= self.ready_upright_projection <= 1.0
            or not 0.05 <= self.ready_linear_speed_mps <= 0.50
            or not 0.10 <= self.ready_angular_speed_rad_s <= 1.50
            or not 0.05 <= self.ready_heading_error_rad <= 0.60
            or not 0.035 <= self.ready_foot_height_m <= 0.10
            or not 10 <= self.success_stable_steps <= 100
            or any(
                not 0.0 <= value <= 1.0
                for value in (
                    self.residual_penalty_scale,
                    self.action_delta_penalty_scale,
                    self.tracking_penalty_scale,
                    self.torque_saturation_penalty_scale,
                    self.drift_penalty_scale,
                )
            )
            or not 0.0 < self.soft_balance_reward_scale <= 2.0
            or not 0.0 < self.linear_motion_penalty_scale <= 1.0
            or not 0.0 < self.angular_motion_penalty_scale <= 1.0
            or not 0.0 < self.support_penalty_scale <= 1.0
            or not 0.0 < self.height_penalty_scale <= 1.0
            or not 0.0 < self.upright_penalty_scale <= 1.0
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery MJX config is invalid")

    @property
    def residual_limits_rad(self) -> np.ndarray[Any, Any]:
        return np.asarray(
            (self.residual_limit_lower_body_rad,) * 12
            + (self.residual_limit_waist_rad,) * 3
            + (self.residual_limit_arm_rad,) * 14,
            dtype=np.float32,
        )

    @property
    def observation_dim(self) -> int:
        frame_dim = _FRAME_DIM_BASE + (
            _GAIN_CONTEXT_DIM if self.gain_memory_mode == "DYNAMIC" else 0
        )
        return frame_dim * self.history_frames

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ImpactRecoveryMJXEvaluationConfig:
    """Independent acquisition/retention evaluation contract."""

    num_envs: int = 32
    seeds: tuple[int, ...] = (57_151, 57_152)
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_mjx_evaluation_config.v1"

    def __post_init__(self) -> None:
        if (
            not 8 <= self.num_envs <= 256
            or not 2 <= len(self.seeds) <= 8
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31
                for seed in self.seeds
            )
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery MJX evaluation config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _rotation_matrix(quaternion_wxyz: jax.Array) -> jax.Array:
    w, x, y, z = quaternion_wxyz
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=jnp.float32,
    )


def _memory_blend_fraction(
    memory_step: jax.Array,
    acquisition_reset: jax.Array,
    config: ImpactRecoveryMJXConfig,
) -> jax.Array:
    blend = jnp.clip(
        memory_step.astype(jnp.float32) / config.memory_blend_steps,
        0.0,
        1.0,
    )
    if config.retention_memory_mode == "DIRECT_REPLAY":
        return jnp.where(acquisition_reset, blend, jnp.ones_like(blend))
    return blend


def _teacher_novelty_gate(
    qpos: jax.Array,
    qvel: jax.Array,
    reference_qpos: jax.Array,
    reference_qvel: jax.Array,
    config: ImpactRecoveryMJXConfig,
) -> jax.Array:
    """Return continuous plasticity permission outside teacher memory.

    The six normalized components are deliberately deployable body signals.
    A state on the successful teacher manifold receives no learned residual;
    permission increases smoothly only as the body leaves that manifold.
    """

    gravity = _rotation_matrix(qpos[3:7]).T @ jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.float32)
    reference_gravity = _rotation_matrix(reference_qpos[3:7]).T @ jnp.asarray(
        (0.0, 0.0, -1.0), dtype=jnp.float32
    )
    components = jnp.asarray(
        (
            jnp.abs(qpos[2] - reference_qpos[2]) / 0.10,
            jnp.linalg.norm(gravity - reference_gravity) / 0.25,
            jnp.linalg.norm(qvel[:3] - reference_qvel[:3]),
            jnp.linalg.norm(qvel[3:6] - reference_qvel[3:6]) / 2.0,
            jnp.sqrt(jnp.mean(jnp.square((qpos[7:36] - reference_qpos[7:36]) / 0.50))),
            jnp.sqrt(jnp.mean(jnp.square((qvel[6:35] - reference_qvel[6:35]) / 3.0))),
        )
    )
    novelty = jnp.sqrt(jnp.mean(jnp.square(components)))
    return jnp.clip(
        (novelty - config.novelty_gate_lower)
        / (config.novelty_gate_upper - config.novelty_gate_lower),
        0.0,
        1.0,
    )


def _upright(quaternion_wxyz: jax.Array) -> jax.Array:
    _, x, y, _ = quaternion_wxyz
    return 1.0 - 2.0 * (x * x + y * y)


def _heading_error(quaternion_wxyz: jax.Array, desired_heading_rad: float) -> jax.Array:
    w, x, y, z = quaternion_wxyz
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.arctan2(
        jnp.sin(desired_heading_rad - yaw),
        jnp.cos(desired_heading_rad - yaw),
    )


def _pseudo_huber(value: jax.Array, delta: float = 0.5) -> jax.Array:
    scaled = value / delta
    return delta * delta * (jnp.sqrt(1.0 + scaled * scaled) - 1.0)


def _tree_hash(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "hash": hash_bytes(path.read_bytes()),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    if not rows:
        raise ValueError("impact-recovery checkpoint tree is empty")
    return str(hash_json(rows)), rows


class ImpactRecoveryMJXEnv(Env):  # type: ignore[misc]
    """Current-G1 MJX environment with direct learned joint corrections."""

    def __init__(
        self,
        *,
        model_path: Path,
        curriculum_arrays: dict[str, np.ndarray[Any, Any]],
        desired_heading_rad: float,
        reset_population: ResetPopulation,
        config: ImpactRecoveryMJXConfig,
        acquisition_probabilities: np.ndarray[Any, Any] | None = None,
    ) -> None:
        if reset_population not in {"MIXED", "RETENTION", "ACQUISITION"}:
            raise ValueError("impact-recovery reset population is invalid")
        self._config = config
        self._reset_population = reset_population
        self._desired_heading = desired_heading_rad
        self._mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self._mj_model.opt.timestep = 0.002
        if (
            self._mj_model.nq < _G1_QPOS_WIDTH
            or self._mj_model.nv < _G1_QVEL_WIDTH
            or self._mj_model.nu != _JOINT_COUNT
        ):
            raise ValueError("impact-recovery MJX model lacks the G1 state contract")
        expected_joint_addresses = np.arange(7, 36, dtype=np.int32)
        expected_dof_addresses = np.arange(6, 35, dtype=np.int32)
        if not np.array_equal(
            self._mj_model.jnt_qposadr[1:30], expected_joint_addresses
        ) or not np.array_equal(self._mj_model.jnt_dofadr[1:30], expected_dof_addresses):
            raise ValueError("impact-recovery G1 joint address contract changed")
        self._left_foot_site = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        )
        self._right_foot_site = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        )
        if self._left_foot_site < 0 or self._right_foot_site < 0:
            raise ValueError("impact-recovery G1 foot sites are unavailable")
        self._mjx_model = mjx.put_model(self._mj_model)
        self._qpos = jnp.asarray(curriculum_arrays["qpos"])
        self._qvel = jnp.asarray(curriculum_arrays["qvel"])
        self._memory = jnp.asarray(curriculum_arrays["frozen_memory_target_rad"])
        self._initial_motor_target = jnp.asarray(curriculum_arrays["initial_motor_target_rad"])
        self._elapsed_since_contact = jnp.asarray(curriculum_arrays["elapsed_since_contact_sec"])
        if config.gain_memory_mode == "DYNAMIC":
            gain_names = {
                "initial_kp",
                "initial_kd",
                "frozen_memory_kp",
                "frozen_memory_kd",
            }
            if not gain_names.issubset(curriculum_arrays):
                raise ValueError("impact-recovery dynamic gain memory is unavailable")
            self._initial_kp = jnp.asarray(curriculum_arrays["initial_kp"])
            self._initial_kd = jnp.asarray(curriculum_arrays["initial_kd"])
            self._memory_kp = jnp.asarray(curriculum_arrays["frozen_memory_kp"])
            self._memory_kd = jnp.asarray(curriculum_arrays["frozen_memory_kd"])
        else:
            self._initial_kp = None
            self._initial_kd = None
            self._memory_kp = None
            self._memory_kd = None
        if config.residual_gate_mode == "TEACHER_NOVELTY":
            state_names = {"frozen_memory_qpos", "frozen_memory_qvel"}
            if not state_names.issubset(curriculum_arrays):
                raise ValueError("impact-recovery teacher state memory is unavailable")
            self._memory_qpos = jnp.asarray(curriculum_arrays["frozen_memory_qpos"])
            self._memory_qvel = jnp.asarray(curriculum_arrays["frozen_memory_qvel"])
        else:
            self._memory_qpos = None
            self._memory_qvel = None
        source_succeeded = np.asarray(curriculum_arrays["source_succeeded"], dtype=np.bool_)
        self._retention_indexes = jnp.asarray(np.flatnonzero(source_succeeded), dtype=jnp.int32)
        self._acquisition_indexes = jnp.asarray(np.flatnonzero(~source_succeeded), dtype=jnp.int32)
        if self._retention_indexes.size == 0 or self._acquisition_indexes.size == 0:
            raise ValueError("impact-recovery MJX requires both curriculum populations")
        if acquisition_probabilities is None:
            self._acquisition_probabilities = None
        else:
            probabilities = np.asarray(acquisition_probabilities, dtype=np.float64)
            if (
                probabilities.shape != (self._acquisition_indexes.size,)
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities <= 0.0)
                or not math.isclose(float(np.sum(probabilities)), 1.0, abs_tol=1.0e-9)
            ):
                raise ValueError("impact-recovery acquisition probabilities are invalid")
            self._acquisition_probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
        if self._memory.shape[1] < config.episode_length:
            raise ValueError("impact-recovery memory is shorter than the training episode")
        self._model_qpos0 = jnp.asarray(self._mj_model.qpos0)
        self._default = jnp.asarray(self._mj_model.qpos0[7:36])
        self._joint_lower = jnp.asarray(self._mj_model.jnt_range[1:30, 0])
        self._joint_upper = jnp.asarray(self._mj_model.jnt_range[1:30, 1])
        self._residual_limits = jnp.asarray(config.residual_limits_rad)
        self._kp = jnp.asarray(_KPS)
        self._kd = jnp.asarray(_KDS)
        self._torque_limit = jnp.asarray(_TORQUE_LIMIT)

    @property
    def observation_size(self) -> int:
        return self._config.observation_dim

    @property
    def action_size(self) -> int:
        return _JOINT_COUNT

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    def _kinematics(self, data: Any) -> tuple[jax.Array, ...]:
        rotation = _rotation_matrix(data.qpos[3:7])
        body_linear_velocity = rotation.T @ data.qvel[:3]
        body_angular_velocity = rotation.T @ data.qvel[3:6]
        heading_error = _heading_error(data.qpos[3:7], self._desired_heading)
        foot_heights = jnp.asarray(
            (
                data.site_xpos[self._left_foot_site, 2],
                data.site_xpos[self._right_foot_site, 2],
            )
        )
        return body_linear_velocity, body_angular_velocity, heading_error, foot_heights

    def _frame(
        self,
        data: Any,
        last_target: jax.Array,
        memory_target: jax.Array,
        kp: jax.Array,
        kd: jax.Array,
    ) -> jax.Array:
        rotation = _rotation_matrix(data.qpos[3:7])
        body_linear_velocity, body_angular_velocity, heading_error, foot_heights = self._kinematics(
            data
        )
        gravity_body = rotation.T @ jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.float32)
        foot_features = jnp.concatenate(
            (
                jnp.clip((foot_heights - 0.035) / 0.10, 0.0, 1.0),
                (foot_heights <= self._config.ready_foot_height_m).astype(jnp.float32),
            )
        )
        values = [
            gravity_body,
            jnp.clip(body_linear_velocity / 2.0, -1.0, 1.0),
            jnp.clip(body_angular_velocity / 3.0, -1.0, 1.0),
            jnp.stack((jnp.sin(heading_error), jnp.cos(heading_error))),
            data.qpos[7:36] - self._default,
            data.qvel[6:35] * 0.05,
            last_target,
            memory_target - data.qpos[7:36],
            foot_features,
        ]
        if self._config.gain_memory_mode == "DYNAMIC":
            values.extend((jnp.clip(kp / 300.0, 0.0, 1.5), jnp.clip(kd / 10.0, 0.0, 1.5)))
        frame = jnp.concatenate(tuple(values))
        return jnp.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_index(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        population_rng, retention_rng, acquisition_rng = jax.random.split(rng, 3)
        retention = self._retention_indexes[
            jax.random.randint(retention_rng, (), 0, self._retention_indexes.shape[0])
        ]
        if self._acquisition_probabilities is None:
            acquisition_offset = jax.random.randint(
                acquisition_rng, (), 0, self._acquisition_indexes.shape[0]
            )
        else:
            acquisition_offset = jax.random.choice(
                acquisition_rng,
                self._acquisition_indexes.shape[0],
                p=self._acquisition_probabilities,
            )
        acquisition = self._acquisition_indexes[acquisition_offset]
        if self._reset_population == "RETENTION":
            return retention, jnp.asarray(False)
        if self._reset_population == "ACQUISITION":
            return acquisition, jnp.asarray(True)
        use_acquisition = (
            jax.random.uniform(population_rng, ()) < self._config.acquisition_reset_fraction
        )
        return jnp.where(use_acquisition, acquisition, retention), use_acquisition

    def reset(self, rng: jax.Array) -> State:
        rng, index_rng, position_rng, velocity_rng = jax.random.split(rng, 4)
        index, acquisition = self._select_index(index_rng)
        joint_noise = jax.random.uniform(
            position_rng,
            (_JOINT_COUNT,),
            minval=-self._config.joint_position_noise_rad,
            maxval=self._config.joint_position_noise_rad,
        )
        velocity_noise = jax.random.uniform(
            velocity_rng, (_G1_QVEL_WIDTH,), minval=-1.0, maxval=1.0
        )
        qpos = self._model_qpos0.at[:_G1_QPOS_WIDTH].set(self._qpos[index])
        qpos = qpos.at[7:36].set(
            jnp.clip(qpos[7:36] + joint_noise, self._joint_lower, self._joint_upper)
        )
        qvel = jnp.zeros((self._mj_model.nv,), dtype=jnp.float32)
        qvel = qvel.at[:_G1_QVEL_WIDTH].set(self._qvel[index])
        qvel = qvel.at[:3].add(velocity_noise[:3] * self._config.root_linear_velocity_noise_mps)
        qvel = qvel.at[3:6].add(
            velocity_noise[3:6] * self._config.root_angular_velocity_noise_rad_s
        )
        qvel = qvel.at[6:35].add(velocity_noise[6:35] * self._config.joint_velocity_noise_rad_s)
        data = mjx.make_data(self._mjx_model).replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32),
        )
        data = mjx.forward(self._mjx_model, data)
        target = self._initial_motor_target[index]
        if self._config.gain_memory_mode == "DYNAMIC":
            assert self._initial_kp is not None
            assert self._initial_kd is not None
            initial_kp = self._initial_kp[index]
            initial_kd = self._initial_kd[index]
        else:
            initial_kp = self._kp
            initial_kd = self._kd
        frame = self._frame(data, target, target, initial_kp, initial_kd)
        history = jnp.repeat(frame[jnp.newaxis, :], self._config.history_frames, axis=0)
        body_linear, body_angular, heading, foot_heights = self._kinematics(data)
        zero = jnp.zeros((), dtype=jnp.float32)
        return State(
            pipeline_state=data,
            obs=history.reshape((-1,)),
            reward=zero,
            done=zero,
            metrics={
                "reward": zero,
                "success": zero,
                "ready": zero,
                "goal_ready": zero,
                "maximum_stable_streak": zero,
                "acquisition_reset": acquisition.astype(jnp.float32),
                "curriculum_row_once": zero,
                "elapsed_since_contact_once": zero,
                "pelvis_height": data.qpos[2],
                "upright": _upright(data.qpos[3:7]),
                "root_body_backward_speed": jnp.maximum(-body_linear[0], 0.0),
                "root_body_lateral_speed": jnp.abs(body_linear[1]),
                "root_body_linear_speed": jnp.linalg.norm(body_linear),
                "root_angular_speed": jnp.linalg.norm(body_angular),
                "heading_error_abs": jnp.abs(heading),
                "bilateral_support": jnp.all(
                    foot_heights <= self._config.ready_foot_height_m
                ).astype(jnp.float32),
                "height_ready": (data.qpos[2] >= self._config.ready_pelvis_height_m).astype(
                    jnp.float32
                ),
                "upright_ready": (
                    _upright(data.qpos[3:7]) >= self._config.ready_upright_projection
                ).astype(jnp.float32),
                "linear_speed_ready": (
                    jnp.linalg.norm(body_linear) <= self._config.ready_linear_speed_mps
                ).astype(jnp.float32),
                "angular_speed_ready": (
                    jnp.linalg.norm(body_angular) <= self._config.ready_angular_speed_rad_s
                ).astype(jnp.float32),
                "soft_balance": zero,
                "residual_rms": zero,
                "torque_saturation": zero,
                "residual_gate": zero,
            },
            info={
                "rng": rng,
                "curriculum_index": index,
                "acquisition_reset": acquisition,
                "memory_step": jnp.zeros((), dtype=jnp.int32),
                "stable_streak": jnp.zeros((), dtype=jnp.int32),
                "maximum_stable_streak": jnp.zeros((), dtype=jnp.int32),
                "last_target": target,
                "initial_target": target,
                "initial_kp": initial_kp,
                "initial_kd": initial_kd,
                "last_action": jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32),
                "history": history,
                "initial_xy": data.qpos[:2],
                "potential": self._potential(data),
            },
        )

    def _potential(self, data: Any) -> jax.Array:
        body_linear, body_angular, heading, foot_heights = self._kinematics(data)
        height = jnp.clip((data.qpos[2] - 0.45) / 0.27, 0.0, 1.0)
        upright = jnp.clip((_upright(data.qpos[3:7]) - 0.50) / 0.50, 0.0, 1.0)
        motion = jnp.exp(
            -jnp.square(jnp.linalg.norm(body_linear) / 0.35)
            - jnp.square(jnp.linalg.norm(body_angular) / 0.9)
        )
        heading_score = (
            0.5 * (jnp.cos(heading) + 1.0)
            if self._config.learning_stage == "GOAL_READY"
            else jnp.zeros((), dtype=jnp.float32)
        )
        support = jnp.mean(jnp.exp(-jnp.maximum(foot_heights - 0.04, 0.0) / 0.035))
        return 1.5 * height + 1.5 * upright + motion + heading_score + 0.5 * support

    def _memory_blend(self, memory_step: jax.Array, acquisition_reset: jax.Array) -> jax.Array:
        return _memory_blend_fraction(memory_step, acquisition_reset, self._config)

    def _residual_gate(
        self,
        data: Any,
        curriculum_index: jax.Array,
        memory_step: jax.Array,
    ) -> jax.Array:
        if self._config.residual_gate_mode == "NONE":
            gate = jnp.ones((), dtype=jnp.float32)
        else:
            assert self._memory_qpos is not None
            assert self._memory_qvel is not None
            reference_qpos = self._memory_qpos[curriculum_index, memory_step]
            reference_qvel = self._memory_qvel[curriculum_index, memory_step]
            gate = _teacher_novelty_gate(
                data.qpos,
                data.qvel,
                reference_qpos,
                reference_qvel,
                self._config,
            )
        if self._config.residual_authority_steps > 0:
            gate = gate * (memory_step < self._config.residual_authority_steps).astype(jnp.float32)
        return gate

    def step(self, state: State, action: jax.Array) -> State:
        data = state.pipeline_state
        index = state.info["curriculum_index"]
        memory_step = jnp.minimum(state.info["memory_step"], self._memory.shape[1] - 1)
        memory_target = self._memory[index, memory_step]
        blend = self._memory_blend(memory_step, state.info["acquisition_reset"])
        baseline_target = (1.0 - blend) * state.info["initial_target"] + blend * memory_target
        if self._config.gain_memory_mode == "DYNAMIC":
            assert self._memory_kp is not None
            assert self._memory_kd is not None
            memory_kp = self._memory_kp[index, memory_step]
            memory_kd = self._memory_kd[index, memory_step]
            kp = (1.0 - blend) * state.info["initial_kp"] + blend * memory_kp
            kd = (1.0 - blend) * state.info["initial_kd"] + blend * memory_kd
        else:
            kp = self._kp
            kd = self._kd
        residual_gate = self._residual_gate(data, index, memory_step)
        bounded_action = jnp.clip(action, -1.0, 1.0) * residual_gate
        desired_target = jnp.clip(
            baseline_target + bounded_action * self._residual_limits,
            self._joint_lower,
            self._joint_upper,
        )
        target_delta = jnp.clip(
            desired_target - state.info["last_target"],
            -self._config.maximum_target_step_rad,
            self._config.maximum_target_step_rad,
        )
        motor_target = state.info["last_target"] + target_delta

        def simulation_step(current: Any, unused: Any) -> tuple[Any, jax.Array]:
            del unused
            raw_torque = kp * (motor_target - current.qpos[7:36]) - kd * (current.qvel[6:35])
            saturation = jnp.mean((jnp.abs(raw_torque) > self._torque_limit).astype(jnp.float32))
            torque = jnp.clip(raw_torque, -self._torque_limit, self._torque_limit)
            return mjx.step(self._mjx_model, current.replace(ctrl=torque)), saturation

        data, saturation_trace = jax.lax.scan(simulation_step, data, None, length=10)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        data = data.replace(
            qpos=jnp.nan_to_num(data.qpos, nan=0.0, posinf=0.0, neginf=0.0),
            qvel=jnp.nan_to_num(data.qvel, nan=0.0, posinf=0.0, neginf=0.0),
        )
        body_linear, body_angular, heading, foot_heights = self._kinematics(data)
        upright = _upright(data.qpos[3:7])
        linear_speed = jnp.linalg.norm(body_linear)
        angular_speed = jnp.linalg.norm(body_angular)
        bilateral_support = jnp.all(foot_heights <= self._config.ready_foot_height_m)
        balance_ready = (
            (data.qpos[2] >= self._config.ready_pelvis_height_m)
            & (upright >= self._config.ready_upright_projection)
            & (linear_speed <= self._config.ready_linear_speed_mps)
            & (angular_speed <= self._config.ready_angular_speed_rad_s)
            & bilateral_support
        )
        goal_ready = balance_ready & (jnp.abs(heading) <= self._config.ready_heading_error_rad)
        ready = goal_ready if self._config.learning_stage == "GOAL_READY" else balance_ready
        stable_streak = jnp.where(ready, state.info["stable_streak"] + 1, 0)
        maximum_stable_streak = jnp.maximum(state.info["maximum_stable_streak"], stable_streak)
        success = stable_streak >= self._config.success_stable_steps
        fallen = (data.qpos[2] < 0.42) | (upright < 0.20)
        residual_rms = jnp.sqrt(jnp.mean(jnp.square(bounded_action)))
        action_delta_rms = jnp.sqrt(
            jnp.mean(jnp.square(bounded_action - state.info["last_action"]))
        )
        tracking_rms = jnp.sqrt(jnp.mean(jnp.square(data.qpos[7:36] - baseline_target)))
        torque_saturation = jnp.mean(saturation_trace)
        drift = jnp.linalg.norm(data.qpos[:2] - state.info["initial_xy"])
        current_potential = self._potential(data)
        height_soft = jnp.clip((data.qpos[2] - 0.45) / 0.27, 0.0, 1.0)
        upright_soft = jnp.clip((upright - 0.50) / 0.50, 0.0, 1.0)
        linear_soft = jnp.exp(-jnp.square(linear_speed / 0.50))
        angular_soft = jnp.exp(-jnp.square(angular_speed / 1.20))
        support_soft = jnp.mean(jnp.exp(-jnp.maximum(foot_heights - 0.04, 0.0) / 0.035))
        soft_balance = jnp.power(
            jnp.maximum(
                height_soft * upright_soft * linear_soft * angular_soft * support_soft,
                1.0e-8,
            ),
            0.20,
        )
        linear_motion_cost = _pseudo_huber(linear_speed / 0.35)
        angular_motion_cost = _pseudo_huber(angular_speed / 0.90)
        support_cost = jnp.mean(_pseudo_huber(jnp.maximum(foot_heights - 0.045, 0.0) / 0.04))
        height_cost = _pseudo_huber(jnp.maximum(0.72 - data.qpos[2], 0.0) / 0.10)
        upright_cost = _pseudo_huber(jnp.maximum(0.97 - upright, 0.0) / 0.10)
        stable_fraction = jnp.clip(
            stable_streak.astype(jnp.float32) / self._config.success_stable_steps,
            0.0,
            1.0,
        )
        dense_ready = (
            0.08 * jnp.clip((data.qpos[2] - 0.45) / 0.27, 0.0, 1.0)
            + 0.08 * jnp.clip((upright - 0.50) / 0.50, 0.0, 1.0)
            + 0.10 * jnp.exp(-jnp.square(jnp.linalg.norm(body_linear) / 0.35))
            + 0.08 * jnp.exp(-jnp.square(jnp.linalg.norm(body_angular) / 0.9))
            + (
                0.08 * (0.5 * (jnp.cos(heading) + 1.0))
                if self._config.learning_stage == "GOAL_READY"
                else 0.0
            )
            + 0.05 * bilateral_support.astype(jnp.float32)
        )
        reward = (
            5.0 * (current_potential - state.info["potential"])
            + dense_ready
            + self._config.soft_balance_reward_scale * soft_balance
            + 0.75 * ready.astype(jnp.float32)
            + 1.5 * jnp.square(stable_fraction)
            + 60.0 * success.astype(jnp.float32)
            - self._config.residual_penalty_scale * residual_rms
            - self._config.action_delta_penalty_scale * action_delta_rms
            - self._config.tracking_penalty_scale * tracking_rms
            - self._config.torque_saturation_penalty_scale * torque_saturation
            - self._config.drift_penalty_scale * drift
            - self._config.linear_motion_penalty_scale * linear_motion_cost
            - self._config.angular_motion_penalty_scale * angular_motion_cost
            - self._config.support_penalty_scale * support_cost
            - self._config.height_penalty_scale * height_cost
            - self._config.upright_penalty_scale * upright_cost
            - 40.0 * fallen.astype(jnp.float32)
            - 40.0 * (~finite).astype(jnp.float32)
        )
        reward = jnp.nan_to_num(reward, nan=-40.0, posinf=-40.0, neginf=-40.0)
        done = success | fallen | (~finite)
        wrapper_step = state.info.get("steps", jnp.zeros((), dtype=jnp.int32))
        episode_end = done | (wrapper_step + 1 >= self._config.episode_length)
        next_memory_step = jnp.where(episode_end, 0, state.info["memory_step"] + 1)
        next_target = jnp.where(episode_end, state.info["initial_target"], motor_target)
        next_memory_index = jnp.minimum(next_memory_step, self._memory.shape[1] - 1)
        next_blend = self._memory_blend(next_memory_step, state.info["acquisition_reset"])
        next_baseline = (1.0 - next_blend) * state.info["initial_target"] + next_blend * (
            self._memory[index, next_memory_index]
        )
        if self._config.gain_memory_mode == "DYNAMIC":
            assert self._memory_kp is not None
            assert self._memory_kd is not None
            next_kp = (1.0 - next_blend) * state.info["initial_kp"] + next_blend * (
                self._memory_kp[index, next_memory_index]
            )
            next_kd = (1.0 - next_blend) * state.info["initial_kd"] + next_blend * (
                self._memory_kd[index, next_memory_index]
            )
        else:
            next_kp = self._kp
            next_kd = self._kd
        frame = self._frame(data, next_target, next_baseline, next_kp, next_kd)
        history = jnp.concatenate((state.info["history"][1:], frame[jnp.newaxis, :]), axis=0)
        info = dict(state.info)
        info.update(
            memory_step=next_memory_step,
            stable_streak=jnp.where(episode_end, 0, stable_streak),
            maximum_stable_streak=jnp.where(episode_end, 0, maximum_stable_streak),
            last_target=next_target,
            last_action=jnp.where(episode_end, jnp.zeros_like(bounded_action), bounded_action),
            history=history,
            potential=jnp.where(
                episode_end, self._potential(state.pipeline_state), current_potential
            ),
        )
        return state.replace(
            pipeline_state=data,
            obs=history.reshape((-1,)),
            reward=reward,
            done=done.astype(jnp.float32),
            metrics={
                "reward": reward,
                "success": success.astype(jnp.float32),
                "ready": ready.astype(jnp.float32),
                "goal_ready": goal_ready.astype(jnp.float32),
                "acquisition_reset": state.info["acquisition_reset"].astype(jnp.float32),
                "curriculum_row_once": jnp.where(
                    memory_step == 0, index.astype(jnp.float32) + 1.0, 0.0
                ),
                "elapsed_since_contact_once": jnp.where(
                    memory_step == 0, self._elapsed_since_contact[index], 0.0
                ),
                "pelvis_height": data.qpos[2],
                "upright": upright,
                "root_body_backward_speed": jnp.maximum(-body_linear[0], 0.0),
                "root_body_lateral_speed": jnp.abs(body_linear[1]),
                "root_body_linear_speed": linear_speed,
                "root_angular_speed": angular_speed,
                "heading_error_abs": jnp.abs(heading),
                "bilateral_support": bilateral_support.astype(jnp.float32),
                "height_ready": (data.qpos[2] >= self._config.ready_pelvis_height_m).astype(
                    jnp.float32
                ),
                "upright_ready": (upright >= self._config.ready_upright_projection).astype(
                    jnp.float32
                ),
                "linear_speed_ready": (linear_speed <= self._config.ready_linear_speed_mps).astype(
                    jnp.float32
                ),
                "angular_speed_ready": (
                    angular_speed <= self._config.ready_angular_speed_rad_s
                ).astype(jnp.float32),
                "soft_balance": soft_balance,
                "maximum_stable_streak": maximum_stable_streak.astype(jnp.float32),
                "residual_rms": residual_rms,
                "torque_saturation": torque_saturation,
                "residual_gate": residual_gate,
            },
            info=info,
        )


def _load_curriculum_arrays(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    manifest = validate_impact_recovery_curriculum(manifest_path)
    archive_path = manifest_path.parent / str(manifest["archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return manifest, arrays


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        array = np.asarray(value)
        return float(array) if array.shape == () else array.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def train_impact_recovery_mjx(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    restore_checkpoint_path: Path | None = None,
    frontier_manifest_path: Path | None = None,
    config: ImpactRecoveryMJXConfig | None = None,
) -> dict[str, Any]:
    """Run four-GPU PPO without granting the result promotion authority."""

    active = config or ImpactRecoveryMJXConfig()
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    restore_checkpoint = (
        restore_checkpoint_path.expanduser().resolve()
        if restore_checkpoint_path is not None
        else None
    )
    frontier_path = (
        frontier_manifest_path.expanduser().resolve()
        if frontier_manifest_path is not None
        else None
    )
    model_path = root / "g1_description" / "g1_liao.xml"
    if not model_path.is_file() or not curriculum_path.is_file():
        raise FileNotFoundError("impact-recovery MJX inputs are incomplete")
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery MJX output must be new and external")
    if restore_checkpoint is not None and (
        not restore_checkpoint.is_dir()
        or not (restore_checkpoint / "ppo_network_config.json").is_file()
        or restore_checkpoint == checkout
        or checkout in restore_checkpoint.parents
    ):
        raise ValueError("impact-recovery MJX restore checkpoint is invalid")
    if frontier_path is not None and not frontier_path.is_file():
        raise FileNotFoundError("impact-recovery frontier manifest is unavailable")
    devices = tuple(jax.devices())
    if len(devices) < active.required_gpu_count or any(
        getattr(device, "platform", "") != "gpu" for device in devices[: active.required_gpu_count]
    ):
        raise RuntimeError(f"impact-recovery MJX requires {active.required_gpu_count} GPU devices")
    manifest, arrays = _load_curriculum_arrays(curriculum_path)
    if (
        manifest.get("schema_version")
        not in {
            "rosclaw_soccer.impact_recovery_curriculum.v2",
            "rosclaw_soccer.impact_recovery_curriculum.v3",
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }
        or manifest.get("body_hash") != g1_body_hash(root)
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
        or active.episode_length
        > int(cast(dict[str, Any], manifest["config"])["memory_horizon_steps"])
        or (
            active.gain_memory_mode == "DYNAMIC"
            and manifest.get("schema_version")
            not in {
                "rosclaw_soccer.impact_recovery_curriculum.v4",
                "rosclaw_soccer.impact_recovery_curriculum.v5",
                "rosclaw_soccer.impact_recovery_curriculum.v6",
            }
        )
        or (
            active.residual_gate_mode == "TEACHER_NOVELTY"
            and manifest.get("schema_version")
            not in {
                "rosclaw_soccer.impact_recovery_curriculum.v5",
                "rosclaw_soccer.impact_recovery_curriculum.v6",
            }
        )
    ):
        raise ValueError("impact-recovery MJX body or horizon binding changed")
    frontier: dict[str, Any] | None = None
    acquisition_probabilities: np.ndarray[Any, Any] | None = None
    parent_checkpoint_hash = (
        _tree_hash(restore_checkpoint)[0] if restore_checkpoint is not None else None
    )
    if frontier_path is not None:
        from rosclaw_soccer.training.impact_recovery_frontier import (
            validate_impact_recovery_frontier,
        )

        frontier = validate_impact_recovery_frontier(frontier_path)
        source_succeeded = np.asarray(arrays["source_succeeded"], dtype=np.bool_)
        acquisition_indexes = np.flatnonzero(~source_succeeded)
        frontier_rows = cast(list[dict[str, Any]], frontier["rows"])
        frontier_by_row = {int(row["archive_row"]): row for row in frontier_rows}
        if (
            restore_checkpoint is None
            or frontier.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
            or frontier.get("source_checkpoint_hash") != parent_checkpoint_hash
            or set(frontier_by_row) != set(acquisition_indexes.tolist())
        ):
            raise ValueError("impact-recovery training frontier binding changed")
        acquisition_probabilities = np.asarray(
            [frontier_by_row[int(index)]["sampling_probability"] for index in acquisition_indexes],
            dtype=np.float64,
        )
    desired_heading = float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"])
    training_environment = ImpactRecoveryMJXEnv(
        model_path=model_path,
        curriculum_arrays=arrays,
        desired_heading_rad=desired_heading,
        reset_population="MIXED",
        config=active,
        acquisition_probabilities=acquisition_probabilities,
    )
    evaluation_environment = ImpactRecoveryMJXEnv(
        model_path=model_path,
        curriculum_arrays=arrays,
        desired_heading_rad=desired_heading,
        reset_population="ACQUISITION",
        config=active,
    )
    destination.mkdir(parents=True)
    progress: list[dict[str, Any]] = []

    def progress_fn(step: int, metrics: Any) -> None:
        row = {"step": int(step), "metrics": _jsonable(metrics)}
        progress.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    checkpoint_dir = destination / "checkpoints"
    started = time.perf_counter()
    _make_policy, _params, final_metrics = ppo_train.train(
        environment=training_environment,
        num_timesteps=active.total_timesteps,
        max_devices_per_host=active.required_gpu_count,
        num_envs=active.num_envs,
        episode_length=active.episode_length,
        action_repeat=1,
        learning_rate=active.learning_rate,
        entropy_cost=active.entropy_cost,
        discounting=active.discounting,
        unroll_length=active.unroll_length,
        batch_size=active.batch_size,
        num_minibatches=active.num_minibatches,
        num_updates_per_batch=active.num_updates_per_batch,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=active.clipping_epsilon,
        gae_lambda=active.gae_lambda,
        max_grad_norm=active.maximum_gradient_norm,
        network_factory=_make_recovery_ppo_networks,
        seed=active.random_seed,
        num_evals=active.num_evals,
        num_eval_envs=active.num_eval_envs,
        deterministic_eval=True,
        eval_env=evaluation_environment,
        progress_fn=progress_fn,
        save_checkpoint_path=str(checkpoint_dir),
        restore_checkpoint_path=(
            str(restore_checkpoint) if restore_checkpoint is not None else None
        ),
    )
    training_sec = time.perf_counter() - started
    checkpoint_hash, checkpoint_files = _tree_hash(checkpoint_dir)
    report: dict[str, Any] = {
        "schema_version": (
            "rosclaw_soccer.impact_recovery_mjx_training_report.v3"
            if frontier is not None
            else "rosclaw_soccer.impact_recovery_mjx_training_report.v2"
        ),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "curriculum_archive_hash": manifest["archive_hash"],
        "body_hash": manifest["body_hash"],
        "compiled_model_contract": compiled_mujoco_model_contract(training_environment.mj_model),
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "parallelization": "BRAX_PPO_JAX_PMAP_VMAP",
        "devices": [str(device) for device in devices[: active.required_gpu_count]],
        "training_reset_population": (
            "MIXED_FAILURE_FRONTIER_PRIORITIZED"
            if frontier is not None
            else "MIXED_FAILURE_PRIORITIZED"
        ),
        "evaluation_reset_population": "ACQUISITION_FAILURE_ONLY",
        "learning_stage": active.learning_stage,
        "continued_from_checkpoint": restore_checkpoint is not None,
        "parent_checkpoint_hash": parent_checkpoint_hash,
        "training_frontier_manifest_hash": (
            frontier["manifest_hash"] if frontier is not None else None
        ),
        "training_frontier_source_evaluation_hash": (
            frontier["source_evaluation_report_hash"] if frontier is not None else None
        ),
        "training_frontier_source_checkpoint_hash": (
            frontier["source_checkpoint_hash"] if frontier is not None else None
        ),
        "failed_sources_used_as_teacher_count": 0,
        "actor_observation_dim": active.observation_dim,
        "actor_observation": (
            "DEPLOYABLE_PROPRIOCEPTION_HISTORY_GOAL_HEADING_AND_DYNAMIC_GAINS"
            if active.gain_memory_mode == "DYNAMIC"
            else "DEPLOYABLE_PROPRIOCEPTION_HISTORY_AND_GOAL_HEADING"
        ),
        "action_semantics": (
            "TEACHER_NOVELTY_GATED_BOUNDED_29_JOINT_PD_RESIDUAL_AROUND_FROZEN_MEMORY"
            if active.residual_gate_mode == "TEACHER_NOVELTY"
            else "DIRECT_BOUNDED_29_JOINT_PD_RESIDUAL_AROUND_FROZEN_MEMORY"
        ),
        "training_sec": training_sec,
        "progress": progress,
        "final_metrics": _jsonable(final_metrics),
        "checkpoint_tree_hash": checkpoint_hash,
        "checkpoint_files": checkpoint_files,
        "sealed_full_chain_holdouts_loaded": 0,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "GPU rollout candidate; CPU-MuJoCo full-chain retention is required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "training-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_mjx_report(report_path)


def validate_impact_recovery_mjx_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery MJX report must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        if not isinstance(config_value, dict):
            raise ValueError("impact-recovery MJX report config is missing")
        config = ImpactRecoveryMJXConfig(**config_value)
        report_schema = report.get("schema_version")
        expected_config_hash = hash_json(config_value)
        devices = report.get("devices")
        checkpoint_files = report.get("checkpoint_files")
        checkpoint_root = resolved.parent / "checkpoints"

        def checkpoint_row_valid(row: Any) -> bool:
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "size_bytes", "hash"}
                or not isinstance(row.get("path"), str)
            ):
                return False
            relative = Path(str(row["path"]))
            candidate = (checkpoint_root / relative).resolve()
            return bool(
                not relative.is_absolute()
                and ".." not in relative.parts
                and checkpoint_root in candidate.parents
                and candidate.is_file()
                and candidate.stat().st_size == row.get("size_bytes")
                and hash_bytes(candidate.read_bytes()) == row.get("hash")
            )

        checkpoint_rows_valid = bool(
            isinstance(checkpoint_files, list)
            and checkpoint_files
            and all(checkpoint_row_valid(row) for row in checkpoint_files)
            and report.get("checkpoint_tree_hash") == hash_json(checkpoint_files)
        )
        if (
            report_schema
            not in {
                "rosclaw_soccer.impact_recovery_mjx_training_report.v1",
                "rosclaw_soccer.impact_recovery_mjx_training_report.v2",
                "rosclaw_soccer.impact_recovery_mjx_training_report.v3",
            }
            or declared != hash_json(report)
            or report.get("config_hash") != expected_config_hash
            or report.get("rollout_backend") != "MUJOCO_MJX"
            or report.get("physics_truth_backend") != "CPU_MUJOCO"
            or report.get("parallelization") != "BRAX_PPO_JAX_PMAP_VMAP"
            or report.get("training_reset_population")
            != (
                "MIXED_FAILURE_FRONTIER_PRIORITIZED"
                if report_schema == "rosclaw_soccer.impact_recovery_mjx_training_report.v3"
                else "MIXED_FAILURE_PRIORITIZED"
            )
            or report.get("evaluation_reset_population") != "ACQUISITION_FAILURE_ONLY"
            or (
                report_schema
                in {
                    "rosclaw_soccer.impact_recovery_mjx_training_report.v2",
                    "rosclaw_soccer.impact_recovery_mjx_training_report.v3",
                }
                and (
                    report.get("learning_stage") != config.learning_stage
                    or not isinstance(report.get("continued_from_checkpoint"), bool)
                    or (
                        report.get("continued_from_checkpoint") is True
                        and _SHA256.fullmatch(str(report.get("parent_checkpoint_hash", ""))) is None
                    )
                    or (
                        report.get("continued_from_checkpoint") is False
                        and report.get("parent_checkpoint_hash") is not None
                    )
                )
            )
            or (
                report_schema == "rosclaw_soccer.impact_recovery_mjx_training_report.v3"
                and (
                    report.get("continued_from_checkpoint") is not True
                    or report.get("training_frontier_source_checkpoint_hash")
                    != report.get("parent_checkpoint_hash")
                    or any(
                        _SHA256.fullmatch(str(report.get(name, ""))) is None
                        for name in (
                            "training_frontier_manifest_hash",
                            "training_frontier_source_evaluation_hash",
                            "training_frontier_source_checkpoint_hash",
                        )
                    )
                )
            )
            or report.get("failed_sources_used_as_teacher_count") != 0
            or report.get("sealed_full_chain_holdouts_loaded") != 0
            or report.get("actor_observation")
            != (
                "DEPLOYABLE_PROPRIOCEPTION_HISTORY_GOAL_HEADING_AND_DYNAMIC_GAINS"
                if config.gain_memory_mode == "DYNAMIC"
                else "DEPLOYABLE_PROPRIOCEPTION_HISTORY_AND_GOAL_HEADING"
            )
            or report.get("action_semantics")
            != (
                "TEACHER_NOVELTY_GATED_BOUNDED_29_JOINT_PD_RESIDUAL_AROUND_FROZEN_MEMORY"
                if config.residual_gate_mode == "TEACHER_NOVELTY"
                else "DIRECT_BOUNDED_29_JOINT_PD_RESIDUAL_AROUND_FROZEN_MEMORY"
            )
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or not isinstance(devices, list)
            or len(devices) != config.required_gpu_count
            or any(
                re.fullmatch(r"(?:cuda:\d+|TFRT_CUDA_\d+)", str(device)) is None
                for device in devices
            )
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "curriculum_manifest_hash",
                    "curriculum_archive_hash",
                    "body_hash",
                    "checkpoint_tree_hash",
                )
            )
            or not isinstance(report.get("compiled_model_contract"), dict)
            or _SHA256.fullmatch(
                str(cast(dict[str, Any], report["compiled_model_contract"]).get("model_hash", ""))
            )
            is None
            or report.get("actor_observation_dim") != config.observation_dim
            or not checkpoint_rows_valid
        ):
            raise ValueError("impact-recovery MJX report authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def evaluate_impact_recovery_memory_baseline(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    population: Literal["ACQUISITION", "RETENTION"],
    evaluation_config: ImpactRecoveryMJXEvaluationConfig | None = None,
    controller_config: ImpactRecoveryMJXConfig | None = None,
) -> dict[str, Any]:
    """Run a deterministic zero-residual exam of content-bound muscle memory."""

    from brax.envs.wrappers import training as training_wrappers
    from brax.training import acting

    if population not in {"ACQUISITION", "RETENTION"}:
        raise ValueError("impact-recovery memory baseline population is invalid")
    active_evaluation = evaluation_config or ImpactRecoveryMJXEvaluationConfig(
        seeds=(57_151, 57_152, 57_153, 57_154)
    )
    active_controller = controller_config or ImpactRecoveryMJXConfig(
        retention_memory_mode="DIRECT_REPLAY",
        gain_memory_mode="DYNAMIC",
    )
    if (
        active_controller.retention_memory_mode != "DIRECT_REPLAY"
        or active_controller.gain_memory_mode != "DYNAMIC"
        or active_controller.residual_gate_mode != "NONE"
    ):
        raise ValueError("impact-recovery memory baseline requires direct dynamic-gain replay")
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if not model_path.is_file() or not curriculum_path.is_file():
        raise FileNotFoundError("impact-recovery memory baseline inputs are incomplete")
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery memory baseline output must be new and external")
    manifest, arrays = _load_curriculum_arrays(curriculum_path)
    if (
        manifest.get("schema_version")
        not in {
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
        or manifest.get("body_hash") != g1_body_hash(root)
    ):
        raise ValueError("impact-recovery dynamic memory binding changed")
    devices = jax.devices()
    if not devices or getattr(devices[0], "platform", "") != "gpu":
        raise RuntimeError("impact-recovery memory baseline requires a GPU device")
    environment = ImpactRecoveryMJXEnv(
        model_path=model_path,
        curriculum_arrays=arrays,
        desired_heading_rad=float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"]),
        reset_population=population,
        config=active_controller,
    )
    wrapped = training_wrappers.wrap(
        environment,
        episode_length=active_controller.episode_length,
        action_repeat=1,
    )

    def zero_policy(unused: Any) -> Any:
        del unused

        def policy(observation: jax.Array, rng: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
            del rng
            action = jnp.zeros(observation.shape[:-1] + (_JOINT_COUNT,), dtype=jnp.float32)
            return action, {}

        return policy

    evaluator = acting.Evaluator(
        wrapped,
        zero_policy,
        num_eval_envs=active_evaluation.num_envs,
        episode_length=active_controller.episode_length,
        action_repeat=1,
        key=jax.random.PRNGKey(active_evaluation.seeds[0]),
    )
    repeats: list[dict[str, Any]] = []
    elapsed_bins: dict[str, dict[str, int]] = {}
    for seed in active_evaluation.seeds:
        evaluator._key = jax.random.PRNGKey(seed)
        metrics = evaluator.run_evaluation(None, {}, aggregate_episodes=False)
        episode_metrics = {
            name.removeprefix("eval/episode_"): np.asarray(value)
            for name, value in metrics.items()
            if name.startswith("eval/episode_")
        }
        success = np.asarray(episode_metrics["success"], dtype=np.float64)
        elapsed = np.asarray(episode_metrics["elapsed_since_contact_once"], dtype=np.float64)
        if success.shape != elapsed.shape or success.size != active_evaluation.num_envs:
            raise ValueError("impact-recovery memory baseline episode metrics changed")
        for elapsed_value, succeeded in zip(elapsed.tolist(), success.tolist(), strict=True):
            lower = max(0, int(math.floor(elapsed_value + 1.0e-6)))
            key = f"{lower}-{lower + 1}"
            row = elapsed_bins.setdefault(key, {"attempts": 0, "successes": 0})
            row["attempts"] += 1
            row["successes"] += int(succeeded > 0.0)
        repeats.append(
            {
                "seed": seed,
                "success_count": int(np.count_nonzero(success > 0.0)),
                "success_rate": float(np.mean(success > 0.0)),
            }
        )
    episode_count = active_evaluation.num_envs * len(active_evaluation.seeds)
    success_count = sum(int(row["success_count"]) for row in repeats)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_memory_baseline_diagnostic.v1",
        "mode": (
            "DIRECT_REPLAY_DYNAMIC_GAINS"
            if population == "RETENTION"
            else "DYNAMIC_GAIN_ROUTE_ZERO_RESIDUAL"
        ),
        "population": population,
        "num_envs": active_evaluation.num_envs,
        "seeds": list(active_evaluation.seeds),
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count,
        "elapsed_bins": dict(sorted(elapsed_bins.items())),
        "repeats": repeats,
        "evaluation_config": asdict(active_evaluation),
        "evaluation_config_hash": active_evaluation.config_hash,
        "controller_config": asdict(active_controller),
        "controller_config_hash": active_controller.config_hash,
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "curriculum_archive_hash": manifest["archive_hash"],
        "body_hash": manifest["body_hash"],
        "compiled_model_contract": compiled_mujoco_model_contract(environment.mj_model),
        "physics_backend": "MUJOCO_MJX",
        "action_semantics": "ZERO_RESIDUAL_AROUND_CONTENT_BOUND_DYNAMIC_GAIN_MEMORY",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Isolated MJX muscle-memory diagnostic; no full-chain promotion claim",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    report_path = destination / "diagnostic.json"
    _atomic_json(report_path, report)
    from rosclaw_soccer.training.impact_recovery_selection import (
        validate_impact_recovery_memory_diagnostic,
    )

    return validate_impact_recovery_memory_diagnostic(report_path)


def evaluate_impact_recovery_mjx_checkpoint(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    training_report_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryMJXEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on independent failure and retention resets."""

    from brax.envs.wrappers import training as training_wrappers
    from brax.training import acting
    from brax.training.agents.ppo import checkpoint

    active = config or ImpactRecoveryMJXEvaluationConfig()
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    training_path = training_report_path.expanduser().resolve()
    selected_checkpoint = checkpoint_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if (
        any(not path.is_file() for path in (model_path, curriculum_path, training_path))
        or not selected_checkpoint.is_dir()
    ):
        raise FileNotFoundError("impact-recovery MJX evaluation inputs are incomplete")
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery MJX evaluation output must be new and external")
    training = validate_impact_recovery_mjx_report(training_path)
    manifest, arrays = _load_curriculum_arrays(curriculum_path)
    checkpoint_root = (training_path.parent / "checkpoints").resolve()
    if (
        checkpoint_root not in selected_checkpoint.parents
        or not (selected_checkpoint / "ppo_network_config.json").is_file()
        or training.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
        or training.get("body_hash") != g1_body_hash(root)
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
    ):
        raise ValueError("impact-recovery MJX evaluation binding changed")
    selected_prefix = selected_checkpoint.relative_to(checkpoint_root).as_posix() + "/"
    declared_checkpoint_rows = [
        row
        for row in cast(list[Any], training["checkpoint_files"])
        if isinstance(row, dict) and str(row.get("path", "")).startswith(selected_prefix)
    ]
    selected_hash, selected_rows = _tree_hash(selected_checkpoint)
    normalized_declared_rows = [
        {
            **row,
            "path": str(row["path"])[len(selected_prefix) :],
        }
        for row in declared_checkpoint_rows
    ]
    if normalized_declared_rows != selected_rows:
        raise ValueError("impact-recovery selected checkpoint bytes changed")
    if not jax.devices() or getattr(jax.devices()[0], "platform", "") != "gpu":
        raise RuntimeError("impact-recovery MJX evaluation requires a GPU device")
    training_config = ImpactRecoveryMJXConfig(**cast(dict[str, Any], training["config"]))
    desired_heading = float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"])
    policy = checkpoint.load_policy(
        selected_checkpoint,
        network_factory=_make_recovery_ppo_networks,
        deterministic=True,
    )
    population_reports: dict[str, Any] = {}
    for population in ("ACQUISITION", "RETENTION"):
        environment = ImpactRecoveryMJXEnv(
            model_path=model_path,
            curriculum_arrays=arrays,
            desired_heading_rad=desired_heading,
            reset_population=population,
            config=training_config,
        )
        wrapped = training_wrappers.wrap(
            environment,
            episode_length=training_config.episode_length,
            action_repeat=1,
        )
        evaluator = acting.Evaluator(
            wrapped,
            lambda unused: policy,
            num_eval_envs=active.num_envs,
            episode_length=training_config.episode_length,
            action_repeat=1,
            key=jax.random.PRNGKey(active.seeds[0]),
        )
        repeats: list[dict[str, Any]] = []
        for seed in active.seeds:
            evaluator._key = jax.random.PRNGKey(seed)
            metrics = evaluator.run_evaluation(None, {}, aggregate_episodes=False)
            episode_metrics = {
                name.removeprefix("eval/episode_"): np.asarray(value)
                for name, value in metrics.items()
                if name.startswith("eval/episode_")
            }
            success = np.asarray(episode_metrics["success"], dtype=np.float64)
            lengths = np.asarray(metrics["eval/avg_episode_length"])
            repeats.append(
                {
                    "seed": seed,
                    "success_count": int(np.count_nonzero(success > 0.0)),
                    "success_rate": float(np.mean(success > 0.0)),
                    "episode_metrics": _jsonable(episode_metrics),
                    "mean_episode_length": float(lengths),
                }
            )
        total_success = sum(int(row["success_count"]) for row in repeats)
        population_reports[population.lower()] = {
            "episode_count": active.num_envs * len(active.seeds),
            "success_count": total_success,
            "success_rate": total_success / (active.num_envs * len(active.seeds)),
            "repeats": repeats,
        }
    destination.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_evaluation_report.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "training_report_hash": training["report_hash"],
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "selected_checkpoint_hash": selected_hash,
        "selected_checkpoint_files": selected_rows,
        "training_learning_stage": training_config.learning_stage,
        "populations": population_reports,
        "physics_backend": "MUJOCO_MJX",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Expanded GPU evaluation only; CPU-MuJoCo full-chain exam required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "evaluation-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_mjx_evaluation_report(report_path)


def validate_impact_recovery_mjx_evaluation_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery MJX evaluation report must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        populations = report.get("populations")
        if not isinstance(config_value, dict) or not isinstance(populations, dict):
            raise ValueError("impact-recovery MJX evaluation report is incomplete")
        config = ImpactRecoveryMJXEvaluationConfig(**config_value)
        expected_episodes = config.num_envs * len(config.seeds)
        if (
            report.get("schema_version")
            != "rosclaw_soccer.impact_recovery_mjx_evaluation_report.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != config.config_hash
            or report.get("physics_backend") != "MUJOCO_MJX"
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or set(populations) != {"acquisition", "retention"}
            or any(
                not isinstance(population, dict)
                or population.get("episode_count") != expected_episodes
                or not isinstance(population.get("success_count"), int)
                or not 0 <= int(population["success_count"]) <= expected_episodes
                or population.get("success_rate")
                != int(population["success_count"]) / expected_episodes
                or not isinstance(population.get("repeats"), list)
                or len(population["repeats"]) != len(config.seeds)
                for population in populations.values()
            )
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "training_report_hash",
                    "curriculum_manifest_hash",
                    "selected_checkpoint_hash",
                )
            )
        ):
            raise ValueError("impact-recovery MJX evaluation authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train goal-conditioned G1 impact recovery")
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--curriculum-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--frontier-manifest", type=Path)
    parser.add_argument("--total-timesteps", default=2_097_152, type=int)
    parser.add_argument("--num-envs", default=256, type=int)
    parser.add_argument("--num-evals", default=5, type=int)
    parser.add_argument("--seed", default=5711, type=int)
    parser.add_argument("--learning-stage", choices=("BALANCE", "GOAL_READY"), default="BALANCE")
    args = parser.parse_args()
    result = train_impact_recovery_mjx(
        asset_root=args.asset_root,
        curriculum_manifest_path=args.curriculum_manifest,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        restore_checkpoint_path=args.restore_checkpoint,
        frontier_manifest_path=args.frontier_manifest,
        config=ImpactRecoveryMJXConfig(
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            num_evals=args.num_evals,
            random_seed=args.seed,
            learning_stage=args.learning_stage,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "ImpactRecoveryMJXConfig",
    "ImpactRecoveryMJXEnv",
    "ImpactRecoveryMJXEvaluationConfig",
    "evaluate_impact_recovery_memory_baseline",
    "evaluate_impact_recovery_mjx_checkpoint",
    "train_impact_recovery_mjx",
    "validate_impact_recovery_mjx_evaluation_report",
    "validate_impact_recovery_mjx_report",
]
