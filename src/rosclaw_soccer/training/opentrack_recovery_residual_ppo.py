"""Four-GPU on-policy residual PPO for OpenTrack G1 recovery physics.

This is a simulation-only candidate generator.  It freezes successful
episodic PD-target memories, learns only a bounded recurrent feedback residual,
and never calls ``env.step`` or reads the OpenTrack trajectory handler while
controlling the body.  A separate CPU MuJoCo exam remains the physical truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rosclaw.continual.residual_adaptation import (
    ResidualAdaptationContract,
    write_residual_adaptation_contract,
)

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    _atomic_json,
    _file_hash,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _verified_development_report,
)
from rosclaw_soccer.training.opentrack_recovery_student_collect import (
    _teacher_body_hash,
    _verified_holdout_report,
)
from rosclaw_soccer.training.recovery_residual_ppo import (
    FailurePrioritizedRecoveryCurriculum,
    RecoveryCurriculumSource,
    RecoveryCurriculumState,
    RecoveryResidualObservationSpec,
    RecoveryResidualPPOConfig,
    RecoveryRewardConfig,
    build_recovery_residual_actor_observation,
    compute_recovery_residual_reward,
    generalized_advantage_estimate,
    recovery_successor_potential,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryDistillationCorpus,
    build_recovery_proprioception,
    load_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryPerturbationConfig,
    body_gravity_vector,
    build_recovery_perturbation_holdout,
)


@dataclass(frozen=True)
class _TrainingState:
    snapshot: RecoverySnapshot
    base_snapshot_hash: str
    source: RecoveryCurriculumSource
    difficulty: float
    perturbation_hash: str | None

    @property
    def curriculum_state(self) -> RecoveryCurriculumState:
        return RecoveryCurriculumState(
            state_hash=self.snapshot.snapshot_hash,
            base_snapshot_hash=self.base_snapshot_hash,
            source=self.source,
            difficulty=self.difficulty,
        )


@dataclass(frozen=True)
class RecoveryResidualStep:
    reward: float
    done: bool
    succeeded: bool
    failed: bool
    stable: bool
    pelvis_height_m: float
    upright_projection: float
    root_linear_speed_mps: float
    root_angular_speed_rad_s: float
    torque_saturation_fraction: float
    normalized_residual_rms: float
    normalized_residual_delta_rms: float


def _memory_hash(
    memories: dict[str, NDArray[np.float32]],
    *,
    corpus_hash: str,
) -> str:
    digest = hashlib.sha256(corpus_hash.encode())
    for base_hash, sequence in sorted(memories.items()):
        digest.update(base_hash.encode())
        digest.update(np.ascontiguousarray(sequence, dtype=np.float32).tobytes())
    return "sha256:" + digest.hexdigest()


def _base_memories(
    corpus: RecoveryDistillationCorpus,
) -> dict[str, NDArray[np.float32]]:
    """Return every successful privileged trajectory as frozen muscle memory."""

    memories: dict[str, NDArray[np.float32]] = {}
    for row in corpus.rows:
        if str(row.get("rollout_controller", "PRIVILEGED_TEACHER")) != ("PRIVILEGED_TEACHER"):
            continue
        start = int(row["start_row"])
        count = int(row["row_count"])
        initial_hash = str(row["initial_snapshot_hash"])
        sequence = np.asarray(
            corpus.absolute_motor_targets_rad[start : start + count],
            dtype=np.float32,
        )
        if initial_hash in memories:
            raise ValueError("recovery residual memory has duplicate initial states")
        memories[initial_hash] = sequence
    if not memories:
        raise ValueError("recovery residual memory has no frozen anchor episodes")
    return memories


def _training_states(
    base_snapshots: tuple[RecoverySnapshot, ...],
    *,
    sealed_state_hashes: set[str],
) -> tuple[_TrainingState, ...]:
    rows: list[_TrainingState] = [
        _TrainingState(
            snapshot=snapshot,
            base_snapshot_hash=snapshot.snapshot_hash,
            source="HISTORICAL_ANCHOR",
            difficulty=0.0,
            perturbation_hash=None,
        )
        for snapshot in base_snapshots
    ]
    configs: tuple[tuple[RecoveryCurriculumSource, float, RecoveryPerturbationConfig], ...] = (
        (
            "CAPABILITY_FRONTIER",
            0.35,
            RecoveryPerturbationConfig(
                samples_per_snapshot=4,
                joint_position_half_width_rad=0.015,
                joint_velocity_half_width_rad_s=0.040,
                root_tilt_half_width_rad=0.012,
                root_linear_velocity_half_width_mps=0.025,
                root_angular_velocity_half_width_rad_s=0.040,
                seed_namespace="rosclaw-s53-recovery-frontier-train-v1",
            ),
        ),
        (
            "RECENT_FAILURE",
            0.60,
            RecoveryPerturbationConfig(
                samples_per_snapshot=3,
                joint_position_half_width_rad=0.030,
                joint_velocity_half_width_rad_s=0.080,
                root_tilt_half_width_rad=0.025,
                root_linear_velocity_half_width_mps=0.050,
                root_angular_velocity_half_width_rad_s=0.080,
                seed_namespace="rosclaw-s53-recovery-failure-train-v1",
            ),
        ),
        (
            "NIGHTMARE",
            0.90,
            RecoveryPerturbationConfig(
                samples_per_snapshot=1,
                joint_position_half_width_rad=0.050,
                joint_velocity_half_width_rad_s=0.150,
                root_tilt_half_width_rad=0.045,
                root_linear_velocity_half_width_mps=0.090,
                root_angular_velocity_half_width_rad_s=0.150,
                seed_namespace="rosclaw-s53-recovery-nightmare-train-v1",
            ),
        ),
        (
            "SOCIAL_TEACHER",
            0.50,
            RecoveryPerturbationConfig(
                samples_per_snapshot=2,
                joint_position_half_width_rad=0.030,
                joint_velocity_half_width_rad_s=0.080,
                root_tilt_half_width_rad=0.025,
                root_linear_velocity_half_width_mps=0.050,
                root_angular_velocity_half_width_rad_s=0.080,
                seed_namespace="rosclaw-s52-recovery-student-train-v1",
            ),
        ),
    )
    for source, difficulty, config in configs:
        for snapshot, perturbation in build_recovery_perturbation_holdout(
            base_snapshots, config=config
        ):
            rows.append(
                _TrainingState(
                    snapshot=snapshot,
                    base_snapshot_hash=perturbation.base_snapshot_hash,
                    source=source,
                    difficulty=difficulty,
                    perturbation_hash=perturbation.perturbation_hash,
                )
            )
    hashes = [item.snapshot.snapshot_hash for item in rows]
    overlap = set(hashes) & sealed_state_hashes
    if len(hashes) != len(set(hashes)) or overlap:
        raise ValueError(
            "recovery residual training states are duplicated or overlap sealed holdout"
        )
    return tuple(rows)


def _warmstart_sequences(
    corpus: RecoveryDistillationCorpus,
    *,
    memories: dict[str, NDArray[np.float32]],
    residual_limits_rad: NDArray[np.float32],
) -> tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...]:
    sequences: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    signatures: dict[str, NDArray[np.float32]] = {}
    base_by_initial: dict[str, str] = {}
    for candidate_row in corpus.rows:
        if str(candidate_row.get("rollout_controller", "PRIVILEGED_TEACHER")) != (
            "PRIVILEGED_TEACHER"
        ):
            continue
        candidate_start = int(candidate_row["start_row"])
        candidate_initial = str(candidate_row["initial_snapshot_hash"])
        signatures[candidate_initial] = np.concatenate(
            (
                corpus.proprio[candidate_start, 6:35] / 0.030,
                corpus.proprio[candidate_start, 35:64] / 0.004,
            )
        ).astype(np.float32)
        base_by_initial[candidate_initial] = str(candidate_row["base_snapshot_hash"])
    for row in corpus.rows:
        if str(row.get("rollout_controller", "PRIVILEGED_TEACHER")) != ("PRIVILEGED_TEACHER"):
            continue
        start = int(row["start_row"])
        count = int(row["row_count"])
        initial_hash = str(row["initial_snapshot_hash"])
        base_hash = str(row["base_snapshot_hash"])
        if initial_hash == base_hash:
            selected_hash = initial_hash
        else:
            alternatives = [
                candidate_hash
                for candidate_hash in memories
                if candidate_hash != initial_hash
                and base_by_initial.get(candidate_hash) == base_hash
            ]
            if not alternatives:
                raise ValueError(
                    "recovery residual warmstart has no leave-one-out memory"
                )
            query = signatures[initial_hash]
            selected_hash = min(
                alternatives,
                key=lambda candidate_hash: float(
                    np.mean(np.square(query - signatures[candidate_hash]))
                ),
            )
        memory = memories.get(selected_hash)
        if memory is None:
            continue
        indexes = np.minimum(np.arange(count), memory.shape[0] - 1)
        nominal = np.asarray(memory[indexes], dtype=np.float32)
        proprio = np.asarray(corpus.proprio[start : start + count], dtype=np.float32)
        joint_position = proprio[:, 6:35] + corpus.default_joint_position_rad
        actor_observation = np.concatenate((proprio, nominal - joint_position), axis=1).astype(
            np.float32
        )
        teacher_target = np.asarray(
            corpus.absolute_motor_targets_rad[start : start + count],
            dtype=np.float32,
        )
        normalized_residual = np.clip(
            (teacher_target - nominal) / residual_limits_rad,
            -1.0,
            1.0,
        ).astype(np.float32)
        sequences.append((actor_observation, normalized_residual))
    if not sequences:
        raise ValueError("recovery residual warmstart has no successful teacher sequences")
    return tuple(sequences)


def _build_actor_critic(
    config: RecoveryResidualPPOConfig,
    spec: RecoveryResidualObservationSpec,
) -> Any:
    import torch  # type: ignore[import-not-found]
    from torch import nn

    critic_dim = spec.actor_observation_dim + spec.critic_privileged_dim

    class RecurrentResidualActorCritic(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.actor_encoder = nn.Sequential(
                nn.Linear(spec.actor_observation_dim, config.encoder_size),
                nn.ELU(),
                nn.LayerNorm(config.encoder_size),
            )
            self.actor_gru = nn.GRUCell(config.encoder_size, config.hidden_size)
            self.actor_head = nn.Linear(config.hidden_size, 29)
            self.critic = nn.Sequential(
                nn.Linear(critic_dim, config.encoder_size),
                nn.ELU(),
                nn.Linear(config.encoder_size, config.hidden_size),
                nn.ELU(),
                nn.Linear(config.hidden_size, 1),
            )
            self.log_standard_deviation = nn.Parameter(
                torch.full((29,), config.initial_log_standard_deviation)
            )
            nn.init.zeros_(self.actor_head.weight)
            nn.init.zeros_(self.actor_head.bias)

        def forward(
            self,
            actor_observation: Any,
            critic_observation: Any,
            initial_hidden: Any,
            reset_before_step: Any,
        ) -> tuple[Any, Any, Any, Any]:
            hidden = initial_hidden
            means = []
            for step in range(actor_observation.shape[1]):
                reset = reset_before_step[:, step].unsqueeze(-1)
                hidden = hidden * (1.0 - reset)
                encoded = self.actor_encoder(actor_observation[:, step])
                hidden = self.actor_gru(encoded, hidden)
                means.append(self.actor_head(hidden))
            mean = torch.stack(means, dim=1)
            value = self.critic(critic_observation).squeeze(-1)
            log_standard_deviation = (
                torch.clamp(self.log_standard_deviation, -6.0, -0.5).view(1, 1, 29).expand_as(mean)
            )
            return mean, value, hidden, log_standard_deviation

    return RecurrentResidualActorCritic()


def _tanh_log_probability(normal: Any, raw_action: Any) -> Any:
    import torch

    squashed = torch.tanh(raw_action)
    correction = torch.log(torch.clamp(1.0 - squashed.square(), min=1e-6))
    return (normal.log_prob(raw_action) - correction).sum(dim=-1)


class _RecoveryResidualPhysics:
    def __init__(
        self,
        *,
        environment: Any,
        constants: Any,
        mujoco: Any,
        corpus: RecoveryDistillationCorpus,
        memories: dict[str, NDArray[np.float32]],
        residual_config: RecoveryResidualPPOConfig,
        reward_config: RecoveryRewardConfig,
        observation_spec: RecoveryResidualObservationSpec,
    ) -> None:
        self.environment = environment
        self.constants = constants
        self.mujoco = mujoco
        self.corpus = corpus
        self.memories = memories
        self.config = residual_config
        self.reward_config = reward_config
        self.observation_spec = observation_spec
        self.residual_limits = residual_config.residual_limits_rad
        self.record: _TrainingState | None = None
        self.sequence = np.empty((0, 29), dtype=np.float32)
        self.step_index = 0
        self.stable_streak = 0
        self.settled_failure_streak = 0
        self.last_target = np.zeros(29, dtype=np.float32)
        self.last_residual = np.zeros(29, dtype=np.float32)
        self.previous_potential = 0.0
        self.residual_authority_gate = 0.0
        self.selected_memory_hash = ""
        self.memory_selection_distance = 0.0

    def reset(self, record: _TrainingState) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self.environment.reset()
        qpos = np.asarray(record.snapshot.qpos, dtype=np.float64).copy()
        qpos[:2] = np.asarray(self.environment.mj_data.qpos[:2], dtype=np.float64)
        self.environment.mj_data.qpos[:] = qpos
        self.environment.mj_data.qvel[:] = np.asarray(record.snapshot.qvel, dtype=np.float64)
        self.mujoco.mj_forward(self.environment.mj_model, self.environment.mj_data)
        self.record = record
        self.step_index = 0
        self.stable_streak = 0
        self.settled_failure_streak = 0
        self.last_target = np.asarray(qpos[7:], dtype=np.float32).copy()
        self.last_residual = np.zeros(29, dtype=np.float32)
        query = np.concatenate(
            (
                (
                    np.asarray(record.snapshot.qpos[7:], dtype=np.float32)
                    - self.corpus.default_joint_position_rad
                )
                / 0.030,
                (np.asarray(record.snapshot.qvel[6:], dtype=np.float32) * 0.05) / 0.004,
            )
        ).astype(np.float32)
        candidates: list[tuple[str, NDArray[np.float32], float]] = []
        for row in self.corpus.rows:
            initial_hash = str(row["initial_snapshot_hash"])
            if (
                initial_hash not in self.memories
                or str(row["base_snapshot_hash"]) != record.base_snapshot_hash
                or str(row.get("rollout_controller", "PRIVILEGED_TEACHER")) != "PRIVILEGED_TEACHER"
            ):
                continue
            start = int(row["start_row"])
            signature = np.concatenate(
                (
                    self.corpus.proprio[start, 6:35] / 0.030,
                    self.corpus.proprio[start, 35:64] / 0.004,
                )
            ).astype(np.float32)
            distance = float(np.sqrt(np.mean(np.square(query - signature))))
            candidates.append((initial_hash, signature, distance))
        if not candidates:
            raise ValueError("recovery residual memory has no initial proprio signature")
        exact = [item for item in candidates if item[0] == record.snapshot.snapshot_hash]
        selected = exact[0] if exact else min(candidates, key=lambda item: item[2])
        self.selected_memory_hash = selected[0]
        self.sequence = self.memories[self.selected_memory_hash]
        self.memory_selection_distance = selected[2]
        self.residual_authority_gate = float(
            np.clip(
                (self.memory_selection_distance - self.config.initial_mismatch_deadband)
                / (
                    self.config.initial_mismatch_full_authority
                    - self.config.initial_mismatch_deadband
                ),
                0.0,
                1.0,
            )
        )
        metrics = self._metrics()
        self.previous_potential = recovery_successor_potential(
            pelvis_height_m=metrics[0],
            upright_projection=metrics[1],
            root_linear_speed_mps=metrics[2],
            root_angular_speed_rad_s=metrics[3],
            config=self.reward_config,
        )
        return self.observation()

    def _nominal_target(self) -> NDArray[np.float32]:
        return np.asarray(
            self.sequence[min(self.step_index, self.sequence.shape[0] - 1)],
            dtype=np.float32,
        )

    def _metrics(self) -> tuple[float, float, float, float]:
        qpos = np.asarray(self.environment.mj_data.qpos, dtype=np.float64)
        qvel = np.asarray(self.environment.mj_data.qvel, dtype=np.float64)
        upright = float(-body_gravity_vector(qpos[3:7])[2])
        return (
            max(0.0, float(qpos[2])),
            upright,
            float(np.linalg.norm(qvel[:3])),
            float(np.linalg.norm(qvel[3:6])),
        )

    def observation(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        nominal = self._nominal_target()
        gravity = self.environment.mj_data.site_xmat[self.environment._pelvis_imu_site_id].reshape(
            3, 3
        ).T @ np.asarray((0.0, 0.0, -1.0))
        proprio = build_recovery_proprioception(
            projected_gravity_body=gravity,
            pelvis_gyro_rad_s=self.environment.get_gyro("pelvis"),
            joint_position_rad=self.environment.mj_data.qpos[7:],
            joint_velocity_rad_s=self.environment.mj_data.qvel[6:],
            last_motor_target_rad=self.last_target,
            default_joint_position_rad=self.corpus.default_joint_position_rad,
            spec=self.corpus.proprioception_spec,
        )
        actor = build_recovery_residual_actor_observation(
            proprioception=proprio,
            internal_memory_target_rad=nominal,
            joint_position_rad=self.environment.mj_data.qpos[7:],
            spec=self.observation_spec,
        )
        pelvis, upright, linear, angular = self._metrics()
        privileged = np.asarray(
            (
                self.step_index / self.config.maximum_episode_steps,
                pelvis,
                upright,
                linear,
                angular,
                self.stable_streak / self.reward_config.final_stable_frames,
            ),
            dtype=np.float32,
        )
        return actor, np.concatenate((actor, privileged)).astype(np.float32)

    def step(
        self, normalized_residual: NDArray[np.floating[Any]]
    ) -> tuple[tuple[NDArray[np.float32], NDArray[np.float32]], RecoveryResidualStep]:
        action = np.asarray(normalized_residual, dtype=np.float32)
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise ValueError("recovery residual action must contain 29 finite values")
        action = np.clip(action, -1.0, 1.0)
        desired_residual = action * self.residual_limits * self.residual_authority_gate
        filtered_delta = np.clip(
            self.config.residual_filter_fraction * (desired_residual - self.last_residual),
            -self.config.maximum_residual_step_rad,
            self.config.maximum_residual_step_rad,
        )
        residual = np.clip(
            self.last_residual + filtered_delta,
            -self.residual_limits,
            self.residual_limits,
        ).astype(np.float32)
        nominal = self._nominal_target()
        motor_target = nominal + residual
        saturation_count = 0
        torque_count = 0
        for _ in range(int(self.environment.dt / self.environment.sim_dt)):
            raw_torque = self.constants.KPs * (
                motor_target - self.environment.mj_data.qpos[7:]
            ) + self.constants.KDs * (-self.environment.mj_data.qvel[6:])
            saturation_count += int(
                np.count_nonzero(np.abs(raw_torque) > self.constants.TORQUE_LIMIT)
            )
            torque_count += raw_torque.size
            self.environment.mj_data.ctrl[:] = np.clip(
                raw_torque,
                -self.constants.TORQUE_LIMIT,
                self.constants.TORQUE_LIMIT,
            )
            self.mujoco.mj_step(self.environment.mj_model, self.environment.mj_data)
        self.last_target = np.asarray(motor_target, dtype=np.float32)
        residual_delta = residual - self.last_residual
        self.last_residual = residual
        self.step_index += 1
        qpos = np.asarray(self.environment.mj_data.qpos, dtype=np.float64)
        qvel = np.asarray(self.environment.mj_data.qvel, dtype=np.float64)
        finite = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
        if finite:
            pelvis, upright, linear, angular = self._metrics()
        else:
            pelvis, upright, linear, angular = 0.0, -1.0, 100.0, 100.0
        stable = bool(
            finite
            and pelvis >= self.reward_config.ready_pelvis_height_m
            and upright >= self.reward_config.ready_upright_projection
            and linear <= self.reward_config.maximum_stable_linear_speed_mps
            and angular <= self.reward_config.maximum_stable_angular_speed_rad_s
        )
        self.stable_streak = self.stable_streak + 1 if stable else 0
        succeeded = self.stable_streak >= self.reward_config.final_stable_frames
        settled_failure = bool(
            finite
            and self.step_index >= self.sequence.shape[0] + 100
            and pelvis < 0.35
            and linear <= 0.05
            and angular <= 0.10
        )
        self.settled_failure_streak = self.settled_failure_streak + 1 if settled_failure else 0
        timed_out = self.step_index >= self.config.maximum_episode_steps
        failed = bool(not finite or timed_out or self.settled_failure_streak >= 50)
        current_potential = (
            recovery_successor_potential(
                pelvis_height_m=pelvis,
                upright_projection=upright,
                root_linear_speed_mps=linear,
                root_angular_speed_rad_s=angular,
                config=self.reward_config,
            )
            if finite
            else 0.0
        )
        next_nominal = self._nominal_target()
        tracking_rmse = (
            float(np.sqrt(np.mean(np.square(next_nominal - qpos[7:])))) if finite else 10.0
        )
        normalized_rms = float(np.sqrt(np.mean(np.square(residual / self.residual_limits))))
        normalized_delta_rms = float(
            np.sqrt(np.mean(np.square(residual_delta / self.residual_limits)))
        )
        saturation = saturation_count / max(1, torque_count)
        reward = compute_recovery_residual_reward(
            previous_potential=self.previous_potential,
            current_potential=current_potential,
            nominal_tracking_rmse_rad=tracking_rmse,
            normalized_residual_rms=normalized_rms,
            normalized_residual_delta_rms=normalized_delta_rms,
            torque_saturation_fraction=saturation,
            stable=stable,
            succeeded=succeeded,
            failed=failed,
            config=self.reward_config,
        )
        self.previous_potential = current_potential
        outcome = RecoveryResidualStep(
            reward=reward,
            done=bool(succeeded or failed),
            succeeded=succeeded,
            failed=failed,
            stable=stable,
            pelvis_height_m=pelvis,
            upright_projection=upright,
            root_linear_speed_mps=linear,
            root_angular_speed_rad_s=angular,
            torque_saturation_fraction=saturation,
            normalized_residual_rms=normalized_rms,
            normalized_residual_delta_rms=normalized_delta_rms,
        )
        return self.observation(), outcome


def _create_environment(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    development: dict[str, Any],
    rank: int,
) -> tuple[Any, Any, Any]:
    root = opentrack_root.expanduser().resolve()
    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.play.play_g1_env_tracking_general")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    payload = json.loads(environment_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("env_config"), dict):
        raise ValueError("recovery residual OpenTrack config is invalid")
    selected = development["post_skill_transfer"]["development_schedule"]["selected_trials"]
    route = selected[rank % len(selected)]["match"]
    environment_config = copy.deepcopy(
        tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
    )
    environment_config.update(payload["env_config"])
    environment_config.reference_traj_config.name = {motion_dataset_id: [route["motion_id"]]}
    environment_config.reference_traj_config.random_start = False
    environment_config.reference_traj_config.fixed_start_frame = route["entry_frame"]
    environment_class = tmj.registry.get("G1TrackingGeneral", "tracking_play_env_class")
    previous = Path.cwd()
    try:
        os.chdir(root)
        environment = environment_class(
            config=environment_config,
            play_ref_motion=False,
            use_viewer=False,
            use_renderer=False,
            exp_name=f"rosclaw-s53-recovery-residual-ppo-rank{rank}",
        )
    finally:
        os.chdir(previous)
    return environment, constants, mujoco


def _global_advantage_normalization(advantage: Any, dist: Any) -> Any:
    import torch

    stats = torch.stack(
        (
            advantage.sum(),
            advantage.square().sum(),
            torch.tensor(float(advantage.numel()), device=advantage.device),
        )
    )
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mean = stats[0] / stats[2]
    variance = torch.clamp(stats[1] / stats[2] - mean.square(), min=1e-8)
    return (advantage - mean) / torch.sqrt(variance)


def _cross_rank_parameter_difference(model: Any, dist: Any) -> float:
    import torch

    maximum = torch.zeros((), device=next(model.parameters()).device)
    if not dist.is_initialized():
        return 0.0
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        maximum = torch.maximum(maximum, torch.max(torch.abs(parameter - reference)))
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def run_opentrack_recovery_residual_ppo(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: RecoveryResidualPPOConfig | None = None,
    reward_config: RecoveryRewardConfig | None = None,
) -> dict[str, Any] | None:
    """Run one synchronized four-rank candidate-training job."""

    import torch
    import torch.distributed as dist  # type: ignore[import-not-found]
    from safetensors.torch import save_file  # type: ignore[import-not-found]
    from torch.nn.parallel import (  # type: ignore[import-not-found]
        DistributedDataParallel,
    )

    active = config or RecoveryResidualPPOConfig()
    reward = reward_config or RecoveryRewardConfig()
    spec = RecoveryResidualObservationSpec()
    root = opentrack_root.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    sealed_path = sealed_holdout_report_path.expanduser().resolve()
    corpus_path = corpus_manifest_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if (
        any(
            not path.is_file()
            for path in (
                environment_path,
                snapshot_path,
                development_path,
                sealed_path,
                corpus_path,
            )
        )
        or not root.is_dir()
    ):
        raise FileNotFoundError("recovery residual PPO inputs are incomplete")
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("recovery residual PPO output must be new and external")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4 or not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("recovery residual PPO requires exactly four CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    development = _verified_development_report(development_path)
    sealed = _verified_holdout_report(sealed_path)
    corpus = load_recovery_distillation_corpus(corpus_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if (
        sealed.get("development_report_hash") != development["report_hash"]
        or development.get("snapshot_manifest_hash") != _file_hash(snapshot_path)
        or development.get("teacher_config_hash") != _file_hash(environment_path)
        or corpus_payload.get("development_report_hash") != development["report_hash"]
        or corpus_payload.get("contains_reference_features") is not False
    ):
        raise ValueError("recovery residual PPO evidence bindings differ")
    recorded_holdout = sealed.get("perturbations")
    if not isinstance(recorded_holdout, list):
        raise ValueError("recovery residual PPO sealed identities are missing")
    sealed_hashes = {str(item["perturbed_snapshot_hash"]) for item in recorded_holdout}
    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    memories = _base_memories(corpus)
    if not {item.snapshot_hash for item in base_snapshots}.issubset(memories):
        raise ValueError("frozen recovery memories do not cover every base snapshot")
    states = _training_states(base_snapshots, sealed_state_hashes=sealed_hashes)
    state_by_hash = {item.snapshot.snapshot_hash: item for item in states}
    curriculum = FailurePrioritizedRecoveryCurriculum(
        tuple(item.curriculum_state for item in states),
        seed=active.random_seed + 10_007 * rank,
    )
    memory_hash_before = _memory_hash(memories, corpus_hash=corpus.manifest_hash)
    acquisition_hash = str(
        hash_json(
            {
                "schema_version": "rosclaw_soccer.recovery_residual_training_states.v1",
                "states": [
                    {
                        "state_hash": item.snapshot.snapshot_hash,
                        "base_snapshot_hash": item.base_snapshot_hash,
                        "source": item.source,
                        "difficulty": item.difficulty,
                        "perturbation_hash": item.perturbation_hash,
                    }
                    for item in states
                ],
                "sealed_overlap_count": 0,
            }
        )
    )
    adaptation = ResidualAdaptationContract(
        run_id="s53-recovery-residual-ppo-v1",
        backend_contract_hash=active.config_hash,
        parent_artifact_hash=memory_hash_before,
        body_hash=str(corpus_payload["body_hash"]),
        rehearsal_dataset_hash=_file_hash(snapshot_path),
        acquisition_dataset_hash=acquisition_hash,
        frozen_parameter_selectors=("frozen_skill_memory.*",),
        trainable_parameter_selectors=("recurrent_residual_actor_critic.*",),
        device_ids=(0, 1, 2, 3),
        maximum_world_steps=active.iterations * active.rollout_steps * world_size,
        policy_learning_rate=active.learning_rate,
        rehearsal_fraction=0.15,
        acquisition_fraction=0.85,
        maximum_residual_output_rms=max(
            active.residual_limit_lower_body_rad,
            active.residual_limit_waist_rad,
            active.residual_limit_arm_rad,
        ),
    )
    if rank == 0:
        write_residual_adaptation_contract(adaptation, output / "residual-adaptation-contract.json")
        curriculum_manifest: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.recovery_residual_curriculum.v1",
            "acquisition_dataset_hash": acquisition_hash,
            "base_snapshot_manifest_hash": _file_hash(snapshot_path),
            "sealed_holdout_report_hash": sealed["report_hash"],
            "sealed_state_overlap_count": 0,
            "state_count": len(states),
            "source_counts": {
                source: sum(item.source == source for item in states)
                for source in curriculum.source_weights
            },
            "source_weights": dict(curriculum.source_weights),
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        }
        curriculum_manifest["manifest_hash"] = hash_json(curriculum_manifest)
        _atomic_json(output / "training-curriculum.json", curriculum_manifest)
    dist.barrier()

    random.seed(active.random_seed + rank)
    np.random.seed(active.random_seed + rank)
    torch.manual_seed(active.random_seed + rank)
    torch.cuda.manual_seed(active.random_seed + rank)
    environment, constants, mujoco = _create_environment(
        opentrack_root=root,
        environment_config_path=environment_path,
        motion_dataset_id=motion_dataset_id,
        development=development,
        rank=rank,
    )
    try:
        if (
            _teacher_body_hash(environment, mujoco) != corpus_payload["body_hash"]
            or _file_hash(Path(constants.task_to_xml("flat_terrain")).resolve())
            != corpus_payload["physics_scene_hash"]
        ):
            raise ValueError("recovery residual PPO body or physics scene differs")
        physics = _RecoveryResidualPhysics(
            environment=environment,
            constants=constants,
            mujoco=mujoco,
            corpus=corpus,
            memories=memories,
            residual_config=active,
            reward_config=reward,
            observation_spec=spec,
        )
        model = _build_actor_critic(active, spec).to(device)
        distributed_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        warm_sequences = _warmstart_sequences(
            corpus,
            memories=memories,
            residual_limits_rad=active.residual_limits_rad,
        )
        warm_optimizer = torch.optim.Adam(
            distributed_model.parameters(), lr=active.warmstart_learning_rate
        )
        warm_rng = np.random.default_rng(active.random_seed + 97 * rank)
        warm_losses: list[float] = []
        warm_sequence_indexes = [
            (rank + world_size * slot) % len(warm_sequences)
            for slot in range(active.warmstart_chunks_per_rank)
        ]
        warm_positions = [0 for _ in warm_sequence_indexes]
        warm_hidden = torch.zeros(
            active.warmstart_chunks_per_rank,
            active.hidden_size,
            device=device,
        )
        for _ in range(active.warmstart_optimizer_steps):
            observations = []
            targets = []
            reset_slots = []
            for slot, sequence_index in enumerate(warm_sequence_indexes):
                sequence, target = warm_sequences[sequence_index]
                start = warm_positions[slot]
                stop = start + active.warmstart_chunk_steps
                observation_chunk = sequence[start:stop]
                target_chunk = target[start:stop]
                if observation_chunk.shape[0] < active.warmstart_chunk_steps:
                    pad = active.warmstart_chunk_steps - observation_chunk.shape[0]
                    observation_chunk = np.pad(observation_chunk, ((0, pad), (0, 0)), mode="edge")
                    target_chunk = np.pad(target_chunk, ((0, pad), (0, 0)), mode="edge")
                observations.append(observation_chunk)
                targets.append(target_chunk)
                reset_slots.append(start == 0)
            actor_tensor = torch.as_tensor(np.stack(observations), device=device)
            target_tensor = torch.as_tensor(np.stack(targets), device=device)
            privileged = torch.zeros(
                (*actor_tensor.shape[:2], spec.critic_privileged_dim),
                device=device,
            )
            critic_tensor = torch.cat((actor_tensor, privileged), dim=-1)
            reset = torch.zeros(actor_tensor.shape[:2], device=device)
            reset[:, 0] = torch.as_tensor(reset_slots, dtype=torch.float32, device=device)
            mean, warm_value, warm_hidden_out, warm_log_standard_deviation = distributed_model(
                actor_tensor,
                critic_tensor,
                warm_hidden,
                reset,
            )
            loss = torch.mean(torch.square(torch.tanh(mean) - target_tensor)) + 0.0 * (
                warm_value.sum() + warm_log_standard_deviation.sum()
            )
            warm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                distributed_model.parameters(), active.maximum_gradient_norm
            )
            warm_optimizer.step()
            warm_losses.append(float(loss.detach().item()))
            warm_hidden = warm_hidden_out.detach()
            for slot, sequence_index in enumerate(warm_sequence_indexes):
                warm_positions[slot] += active.warmstart_chunk_steps
                if warm_positions[slot] >= warm_sequences[sequence_index][0].shape[0]:
                    warm_sequence_indexes[slot] = int(warm_rng.integers(0, len(warm_sequences)))
                    warm_positions[slot] = 0
                    warm_hidden[slot].zero_()

        warmstart_reference = _build_actor_critic(active, spec).to(device)
        warmstart_reference.load_state_dict(model.state_dict())
        warmstart_reference.eval()
        for parameter in warmstart_reference.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(distributed_model.parameters(), lr=active.learning_rate)
        selected = curriculum.sample(source="SOCIAL_TEACHER")
        current_record = state_by_hash[selected.state_hash]
        (actor_observation, critic_observation) = physics.reset(current_record)
        hidden = torch.zeros((1, active.hidden_size), device=device)
        episode_start = True
        local_episodes = 0
        local_successes = 0
        local_reward_total = 0.0
        local_residual_square_sum = 0.0
        local_residual_count = 0
        local_saturation_sum = 0.0
        local_environment_steps = 0
        ppo_losses: list[dict[str, float]] = []
        start_time = time.monotonic()
        update_rng = np.random.default_rng(active.random_seed + 811 * rank)
        for iteration in range(active.iterations):
            actor_rows = []
            critic_rows = []
            hidden_rows = []
            reset_rows = []
            raw_action_rows = []
            log_probability_rows = []
            value_rows = []
            reward_rows = []
            done_rows = []
            for _ in range(active.rollout_steps):
                if episode_start:
                    hidden.zero_()
                actor_tensor = torch.as_tensor(actor_observation, device=device).view(1, 1, -1)
                critic_tensor = torch.as_tensor(critic_observation, device=device).view(1, 1, -1)
                reset_tensor = torch.tensor([[float(episode_start)]], device=device)
                hidden_before = hidden.detach().clone()
                with torch.no_grad():
                    mean, value, hidden_after, log_standard_deviation = model(
                        actor_tensor, critic_tensor, hidden, reset_tensor
                    )
                    normal = torch.distributions.Normal(
                        mean[:, 0], torch.exp(log_standard_deviation[:, 0])
                    )
                    raw_action = normal.sample()
                    action = torch.tanh(raw_action)
                    log_probability = _tanh_log_probability(normal, raw_action)
                next_observation, outcome = physics.step(action[0].detach().cpu().numpy())
                actor_rows.append(actor_observation)
                critic_rows.append(critic_observation)
                hidden_rows.append(hidden_before[0].cpu().numpy())
                reset_rows.append(float(episode_start))
                raw_action_rows.append(raw_action[0].cpu().numpy())
                log_probability_rows.append(float(log_probability.item()))
                value_rows.append(float(value[0, 0].item()))
                reward_rows.append(outcome.reward)
                done_rows.append(outcome.done)
                local_reward_total += outcome.reward
                local_residual_square_sum += outcome.normalized_residual_rms**2
                local_residual_count += 1
                local_saturation_sum += outcome.torque_saturation_fraction
                local_environment_steps += 1
                if outcome.done:
                    curriculum.record(
                        current_record.snapshot.snapshot_hash,
                        succeeded=outcome.succeeded,
                    )
                    local_episodes += 1
                    local_successes += int(outcome.succeeded)
                    progress = (iteration + 1) / active.iterations
                    if progress < 0.25:
                        next_source: RecoveryCurriculumSource | None = "SOCIAL_TEACHER"
                    elif progress < 0.50:
                        next_source = (
                            "SOCIAL_TEACHER"
                            if update_rng.random() < 0.50
                            else "CAPABILITY_FRONTIER"
                        )
                    elif progress < 0.75:
                        next_source = (
                            "CAPABILITY_FRONTIER"
                            if update_rng.random() < 0.60
                            else "RECENT_FAILURE"
                        )
                    else:
                        next_source = None
                    selected = curriculum.sample(source=next_source)
                    current_record = state_by_hash[selected.state_hash]
                    actor_observation, critic_observation = physics.reset(current_record)
                    hidden = torch.zeros((1, active.hidden_size), device=device)
                    episode_start = True
                else:
                    actor_observation, critic_observation = next_observation
                    hidden = hidden_after.detach()
                    episode_start = False
            with torch.no_grad():
                bootstrap_actor = torch.as_tensor(actor_observation, device=device).view(1, 1, -1)
                bootstrap_critic = torch.as_tensor(critic_observation, device=device).view(1, 1, -1)
                bootstrap_reset = torch.tensor([[float(episode_start)]], device=device)
                _, bootstrap, _, _ = model(
                    bootstrap_actor, bootstrap_critic, hidden, bootstrap_reset
                )
            advantage_np, return_np = generalized_advantage_estimate(
                rewards=np.asarray(reward_rows, dtype=np.float32),
                values=np.asarray(value_rows, dtype=np.float32),
                dones=np.asarray(done_rows, dtype=np.bool_),
                bootstrap_value=float(bootstrap[0, 0].item()),
                discount=active.discount,
                gae_lambda=active.gae_lambda,
            )
            actor_batch = torch.as_tensor(np.stack(actor_rows), device=device)
            critic_batch = torch.as_tensor(np.stack(critic_rows), device=device)
            hidden_batch = torch.as_tensor(np.stack(hidden_rows), device=device)
            reset_batch = torch.as_tensor(np.asarray(reset_rows, dtype=np.float32), device=device)
            raw_action_batch = torch.as_tensor(np.stack(raw_action_rows), device=device)
            old_log_probability = torch.as_tensor(
                np.asarray(log_probability_rows, dtype=np.float32), device=device
            )
            old_value = torch.as_tensor(np.asarray(value_rows, dtype=np.float32), device=device)
            advantage = _global_advantage_normalization(
                torch.as_tensor(advantage_np, device=device), dist
            )
            returns = torch.as_tensor(return_np, device=device)
            chunk_starts = np.arange(0, active.rollout_steps, active.recurrent_chunk_steps)
            iteration_losses: list[tuple[float, float, float, float, float]] = []
            for _ in range(active.update_epochs):
                update_rng.shuffle(chunk_starts)
                for offset in range(0, len(chunk_starts), active.chunks_per_minibatch):
                    selected_starts = chunk_starts[offset : offset + active.chunks_per_minibatch]
                    index = torch.as_tensor(
                        np.stack(
                            [
                                np.arange(
                                    start,
                                    start + active.recurrent_chunk_steps,
                                )
                                for start in selected_starts
                            ]
                        ),
                        device=device,
                    )
                    actor_chunk = actor_batch[index]
                    critic_chunk = critic_batch[index]
                    reset_chunk = reset_batch[index]
                    hidden_chunk = hidden_batch[torch.as_tensor(selected_starts, device=device)]
                    mean, new_value, _, log_standard_deviation = distributed_model(
                        actor_chunk, critic_chunk, hidden_chunk, reset_chunk
                    )
                    with torch.no_grad():
                        warmstart_mean, _, _, _ = warmstart_reference(
                            actor_chunk,
                            critic_chunk,
                            hidden_chunk,
                            reset_chunk,
                        )
                    normal = torch.distributions.Normal(mean, torch.exp(log_standard_deviation))
                    selected_raw_action = raw_action_batch[index]
                    new_log_probability = _tanh_log_probability(normal, selected_raw_action)
                    selected_old_log_probability = old_log_probability[index]
                    ratio = torch.exp(new_log_probability - selected_old_log_probability)
                    selected_advantage = advantage[index]
                    unclipped = ratio * selected_advantage
                    clipped = (
                        torch.clamp(
                            ratio,
                            1.0 - active.clip_ratio,
                            1.0 + active.clip_ratio,
                        )
                        * selected_advantage
                    )
                    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                    selected_returns = returns[index]
                    selected_old_value = old_value[index]
                    clipped_value = selected_old_value + torch.clamp(
                        new_value - selected_old_value,
                        -active.clip_ratio,
                        active.clip_ratio,
                    )
                    value_loss = 0.5 * torch.mean(
                        torch.maximum(
                            torch.square(new_value - selected_returns),
                            torch.square(clipped_value - selected_returns),
                        )
                    )
                    entropy = normal.entropy().sum(dim=-1).mean()
                    anchor_loss = torch.mean(torch.square(torch.tanh(mean)))
                    retention_loss = torch.mean(
                        torch.square(torch.tanh(mean) - torch.tanh(warmstart_mean))
                    )
                    total_loss = (
                        policy_loss
                        + active.value_coefficient * value_loss
                        - active.entropy_coefficient * entropy
                        + active.zero_residual_anchor_coefficient * anchor_loss
                        + active.warmstart_retention_coefficient * retention_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        distributed_model.parameters(),
                        active.maximum_gradient_norm,
                    )
                    optimizer.step()
                    iteration_losses.append(
                        (
                            float(policy_loss.detach().item()),
                            float(value_loss.detach().item()),
                            float(entropy.detach().item()),
                            float(anchor_loss.detach().item()),
                            float(retention_loss.detach().item()),
                        )
                    )
            means = np.mean(np.asarray(iteration_losses), axis=0)
            ppo_losses.append(
                {
                    "iteration": iteration + 1,
                    "policy_loss": float(means[0]),
                    "value_loss": float(means[1]),
                    "entropy": float(means[2]),
                    "anchor_loss": float(means[3]),
                    "warmstart_retention_loss": float(means[4]),
                }
            )
            if rank == 0 and (
                iteration == 0 or (iteration + 1) % 10 == 0 or iteration + 1 == active.iterations
            ):
                print(
                    json.dumps(
                        {
                            "iteration": iteration + 1,
                            "world_steps": (iteration + 1) * active.rollout_steps * world_size,
                            "rank0_episodes": local_episodes,
                            "rank0_successes": local_successes,
                            "policy_loss": ppo_losses[-1]["policy_loss"],
                            "value_loss": ppo_losses[-1]["value_loss"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        dist.barrier()
        cross_rank_difference = _cross_rank_parameter_difference(model, dist)
        memory_hash_after = _memory_hash(memories, corpus_hash=corpus.manifest_hash)
        aggregate = torch.tensor(
            (
                float(local_episodes),
                float(local_successes),
                local_reward_total,
                local_residual_square_sum,
                float(local_residual_count),
                local_saturation_sum,
            ),
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        if rank != 0:
            return None
        weights_path = output / "recovery-residual-actor-critic.safetensors"
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
            str(weights_path),
        )
        weights_hash = _file_hash(weights_path)
        artifact: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.recovery_residual_actor_critic.v1",
            "architecture": "PROPRIO_INTERNAL_MEMORY_ERROR_GRU_RESIDUAL_ACTOR_CRITIC_V1",
            "weights": weights_path.name,
            "weights_hash": weights_hash,
            "config": asdict(active),
            "config_hash": active.config_hash,
            "reward_config": asdict(reward),
            "reward_config_hash": reward.config_hash,
            "observation_spec": asdict(spec),
            "observation_spec_hash": spec.spec_hash,
            "frozen_skill_memory_hash": memory_hash_before,
            "frozen_skill_memory_corpus_hash": corpus.manifest_hash,
            "frozen_skill_memory_count": len(memories),
            "frozen_skill_memory_selection": (
                "PROPRIO_INITIAL_STATE_NEAREST_NEIGHBOR_WITH_EXACT_MATCH_PRIORITY"
            ),
            "warmstart_supervision": (
                "LEAVE_ONE_TRAJECTORY_OUT_SUCCESSFUL_TEACHER_RESIDUAL"
            ),
            "online_retention": "FROZEN_WARMSTART_ACTOR_OUTPUT_REHEARSAL",
            "body_hash": corpus_payload["body_hash"],
            "physics_scene_hash": corpus_payload["physics_scene_hash"],
            "residual_limits_rad": active.residual_limits_rad.tolist(),
            "residual_authority_gate": ("INITIAL_PROPRIO_MISMATCH_DEADBAND_TO_FULL_AUTHORITY"),
            "external_reference_features": False,
            "teacher_identity_input": False,
            "future_reference_input": False,
            "direct_environment_step_calls": 0,
            "torque_limits_enforced_each_substep": True,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
        }
        artifact["artifact_hash"] = hash_json(artifact)
        _atomic_json(output / "recovery-residual-actor-critic.json", artifact)
        episode_count = int(aggregate[0].item())
        successes = int(aggregate[1].item())
        residual_count = max(1.0, float(aggregate[4].item()))
        report: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.recovery_residual_ppo_training_report.v1",
            "artifact_hash": artifact["artifact_hash"],
            "artifact_weights_hash": weights_hash,
            "adaptation_contract_hash": adaptation.contract_hash,
            "world_size": world_size,
            "gpu_devices": [torch.cuda.get_device_name(index) for index in range(world_size)],
            "synchronized_ddp": cross_rank_difference == 0.0,
            "maximum_cross_rank_parameter_difference": cross_rank_difference,
            "world_steps": active.iterations * active.rollout_steps * world_size,
            "optimizer_steps_per_rank": active.warmstart_optimizer_steps
            + active.iterations
            * active.update_epochs
            * (active.rollout_steps // active.recurrent_chunk_steps // active.chunks_per_minibatch),
            "warmstart_initial_loss_rank0": warm_losses[0] if warm_losses else None,
            "warmstart_final_loss_rank0": warm_losses[-1] if warm_losses else None,
            "warmstart_supervision": (
                "LEAVE_ONE_TRAJECTORY_OUT_SUCCESSFUL_TEACHER_RESIDUAL"
            ),
            "warmstart_retention_coefficient": (
                active.warmstart_retention_coefficient
            ),
            "completed_episode_count": episode_count,
            "training_success_count": successes,
            "training_success_rate": None if episode_count == 0 else successes / episode_count,
            "mean_step_reward": float(aggregate[2].item()) / residual_count,
            "normalized_residual_rms": math.sqrt(float(aggregate[3].item()) / residual_count),
            "mean_torque_saturation_fraction": float(aggregate[5].item()) / residual_count,
            "frozen_skill_memory_hash_before": memory_hash_before,
            "frozen_skill_memory_hash_after": memory_hash_after,
            "maximum_frozen_parameter_drift": 0.0
            if memory_hash_before == memory_hash_after
            else float("inf"),
            "sealed_holdout_state_reads_during_training": 0,
            "sealed_holdout_identity_reads_for_overlap_guard": len(sealed_hashes),
            "sealed_holdout_overlap_count": 0,
            "curriculum_progression": ("SOCIAL_TEACHER_TO_FRONTIER_TO_FAILURE_MIXTURE"),
            "curriculum_rank0": curriculum.metrics(),
            "last_ppo_losses_rank0": ppo_losses[-10:],
            "elapsed_sec": time.monotonic() - start_time,
            "strict_cpu_mujoco_exam_completed": False,
            "promotion_status": "CANDIDATE_PENDING_CPU_MUJOCO_EXAM",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        }
        report["report_hash"] = hash_json(report)
        _atomic_json(output / "training-report.json", report)
        return report
    finally:
        environment.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--environment-config", required=True, type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--sealed-holdout-report", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--warmstart-optimizer-steps", type=int, default=120)
    parser.add_argument("--maximum-episode-steps", type=int, default=2000)
    args = parser.parse_args()
    report = run_opentrack_recovery_residual_ppo(
        opentrack_root=args.opentrack_root,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        corpus_manifest_path=args.corpus_manifest,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        config=RecoveryResidualPPOConfig(
            iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            warmstart_optimizer_steps=args.warmstart_optimizer_steps,
            maximum_episode_steps=args.maximum_episode_steps,
        ),
    )
    if report is not None:
        print(
            json.dumps(
                {
                    "report_hash": report["report_hash"],
                    "world_steps": report["world_steps"],
                    "training_success_rate": report["training_success_rate"],
                    "strict_cpu_mujoco_exam_completed": report["strict_cpu_mujoco_exam_completed"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryResidualStep",
    "run_opentrack_recovery_residual_ppo",
]
