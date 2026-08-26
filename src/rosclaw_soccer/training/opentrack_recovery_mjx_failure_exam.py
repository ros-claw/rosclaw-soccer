"""Paired MJX exam from content-bound recovery failure states.

The full-route exam remains the retention authority, but its averages can hide
whether a candidate actually learned the 200--400 step failure onset.  This
SIM_ONLY diagnostic replays a parent and candidate actor from identical exact
policy-context states, measures local recovery and drift, and writes a
hash-bound report.  It grants neither deployment nor promotion authority.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import training as brax_training

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_teacher_ppo import (
    OpenTrackRecoveryMJXTeacherResidualEnv,
    _tree_hash,
)
from rosclaw_soccer.training.recovery_mjx import (
    RecoveryMJXTeacherResidualPPOConfig,
    _atomic_json,
    compiled_mujoco_model_contract,
    validate_recovery_mjx_failure_state_exam_report,
    validate_recovery_mjx_failure_state_manifest,
    validate_recovery_mjx_teacher_residual_report,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus


@dataclass(frozen=True)
class RecoveryMJXFailureStateExamConfig:
    num_environments: int = 384
    horizon_steps: int = 400
    random_seed: int = 5480
    minimum_state_coverage_fraction: float = 0.95
    minimum_stable_improvement_fraction: float = 0.02
    maximum_streak_regression_fraction: float = 0.02
    minimum_backward_speed_improvement_fraction: float = 0.02
    maximum_lateral_speed_regression_fraction: float = 0.02
    maximum_yaw_speed_regression_fraction: float = 0.02
    candidate_adapter_gain: float = 1.0
    allow_blind_compatible_failure_bank: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_failure_state_exam_config.v1"

    def __post_init__(self) -> None:
        thresholds = (
            self.minimum_state_coverage_fraction,
            self.minimum_stable_improvement_fraction,
            self.maximum_streak_regression_fraction,
            self.minimum_backward_speed_improvement_fraction,
            self.maximum_lateral_speed_regression_fraction,
            self.maximum_yaw_speed_regression_fraction,
        )
        if (
            not 96 <= self.num_environments <= 2_048
            or self.num_environments % 4
            or not 100 <= self.horizon_steps <= 1_200
            or not 0 <= self.random_seed < 2**31
            or any(not np.isfinite(value) for value in thresholds)
            or not 0.90 <= self.minimum_state_coverage_fraction <= 1.0
            or any(not 0.0 <= value <= 0.25 for value in thresholds[1:])
            or not np.isfinite(self.candidate_adapter_gain)
            or not 1.0 <= self.candidate_adapter_gain <= 4.0
            or not isinstance(self.allow_blind_compatible_failure_bank, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX failure-state exam config is invalid")


_BANK_ARRAYS = (
    "qpos",
    "qvel",
    "handoff_frozen",
    "trajectory_step",
    "trajectory_initial_step",
    "last_motor_targets",
    "last_teacher_action",
    "last_residual",
    "proprioception_history",
    "phase_repeat",
)


def _relative_change(candidate: float, parent: float) -> float:
    return (candidate - parent) / max(abs(parent), 1e-12)


def _aggregate_rollout(result: dict[str, np.ndarray[Any, Any]]) -> dict[str, float | int]:
    episode_steps = np.asarray(result["episode_steps"], dtype=np.float64)
    total_steps = float(np.sum(episode_steps))
    if total_steps <= 0.0:
        raise ValueError("failure-state exam rollout has no active steps")
    episode_count = int(episode_steps.shape[0])
    return {
        "episode_count": episode_count,
        "mean_episode_length": float(np.mean(episode_steps)),
        "success_rate": float(np.mean(result["success"] > 0.0)),
        "stable_fraction": float(np.sum(result["stable"]) / total_steps),
        "ready_fraction": float(np.sum(result["ready"]) / total_steps),
        "mean_maximum_stable_streak": float(np.mean(result["maximum_stable_streak"])),
        "root_body_backward_speed_mps": float(np.sum(result["backward"]) / total_steps),
        "root_body_lateral_speed_mps": float(np.sum(result["lateral"]) / total_steps),
        "pelvis_yaw_speed_rad_s": float(np.sum(result["yaw"]) / total_steps),
        "mean_reward_per_step": float(np.sum(result["reward"]) / total_steps),
        "non_success_termination_rate": float(np.mean(result["non_success_termination"] > 0)),
    }


def _aggregate_action_authority(
    result: dict[str, np.ndarray[Any, Any]], *, frozen_parent_baseline: bool
) -> dict[str, float | str | bool]:
    episode_steps = np.asarray(result["episode_steps"], dtype=np.float64)
    total_steps = float(np.sum(episode_steps))
    if total_steps <= 0.0:
        raise ValueError("failure-state action-authority audit has no active steps")
    return {
        "baseline": (
            "FROZEN_PARENT_RESIDUAL_POLICY" if frozen_parent_baseline else "ZERO_RESIDUAL_POLICY"
        ),
        "normalized_action_space": "CANDIDATE_MINUS_BASELINE_RESIDUAL",
        "motor_target_space": "CLIPPED_CANDIDATE_MINUS_CLIPPED_BASELINE_RAD",
        "frozen_parent_baseline": frozen_parent_baseline,
        "mean_normalized_action_increment_rms": float(
            np.sum(result["adapter_residual_rms"]) / total_steps
        ),
        "maximum_normalized_action_increment_rms": float(
            np.max(result["maximum_adapter_residual_rms"])
        ),
        "mean_motor_target_increment_rms_rad": float(
            np.sum(result["adapter_motor_target_delta_rms_rad"]) / total_steps
        ),
        "maximum_motor_target_increment_rms_rad": float(
            np.max(result["maximum_adapter_motor_target_delta_rms_rad"])
        ),
        "residual_active_fraction": float(np.sum(result["residual_active"]) / total_steps),
    }


def _stratified_failure_reset_keys(
    *, rng: jax.Array, environment_count: int, failure_state_count: int
) -> jax.Array:
    """Build deterministic reset keys that cover every failure state at least once."""

    pool_rng, permutation_rng = jax.random.split(rng)
    pool_size = max(4_096, environment_count * 16)
    key_pool = jax.random.split(pool_rng, pool_size)

    def selected_failure_index(reset_key: jax.Array) -> jax.Array:
        _, injection_rng = jax.random.split(reset_key)
        failure_index_rng = jax.random.split(injection_rng, 7)[5]
        return jax.random.randint(failure_index_rng, (), 0, failure_state_count)

    pool_indices = np.asarray(jax.jit(jax.vmap(selected_failure_index))(key_pool), dtype=np.int32)
    selected_positions: list[int] = []
    selected_position_set: set[int] = set()
    for failure_index in range(failure_state_count):
        matches = np.flatnonzero(pool_indices == failure_index)
        if matches.size == 0:
            raise RuntimeError("stratified failure-state reset key pool is incomplete")
        position = int(matches[0])
        selected_positions.append(position)
        selected_position_set.add(position)
    for position in range(pool_size):
        if len(selected_positions) >= environment_count:
            break
        if position not in selected_position_set:
            selected_positions.append(position)
    if len(selected_positions) != environment_count:
        raise RuntimeError("stratified failure-state reset keys are incomplete")
    selected_keys = key_pool[jnp.asarray(selected_positions, dtype=jnp.int32)]
    return jax.random.permutation(permutation_rng, selected_keys, axis=0, independent=False)


def _evaluate_actor(
    *,
    wrapped_environment: Any,
    actor_policy: Any,
    reset_keys: jax.Array,
    rollout_rng: jax.Array,
    horizon_steps: int,
) -> dict[str, np.ndarray[Any, Any]]:
    initial_state = jax.jit(wrapped_environment.reset)(reset_keys)
    selected_indices = initial_state.info["selected_failure_state_index"]
    failure_resets = initial_state.info["failure_state_reset"]
    environment_count = int(reset_keys.shape[0])
    zero = jnp.zeros((environment_count,), dtype=jnp.float32)
    accumulators = {
        "episode_steps": zero,
        "success": zero,
        "stable": zero,
        "ready": zero,
        "maximum_stable_streak": zero,
        "backward": zero,
        "lateral": zero,
        "yaw": zero,
        "reward": zero,
        "adapter_residual_rms": zero,
        "maximum_adapter_residual_rms": zero,
        "adapter_motor_target_delta_rms_rad": zero,
        "maximum_adapter_motor_target_delta_rms_rad": zero,
        "residual_active": zero,
        "non_success_termination": jnp.zeros((environment_count,), dtype=jnp.bool_),
    }

    def rollout_step(
        carry: tuple[Any, jax.Array, jax.Array, dict[str, jax.Array]],
        step: jax.Array,
    ) -> tuple[tuple[Any, jax.Array, jax.Array, dict[str, jax.Array]], None]:
        state, rng, active, totals = carry
        rng, action_rng = jax.random.split(rng)
        action, _ = actor_policy(state.obs, action_rng)
        next_state = wrapped_environment.step(state, action)
        active_float = active.astype(jnp.float32)
        metrics = next_state.metrics
        success = metrics["success"] > 0.0
        episode_done = next_state.info["episode_done"].astype(jnp.bool_)
        finite = jnp.all(jnp.isfinite(next_state.pipeline_state.data.qpos), axis=-1) & jnp.all(
            jnp.isfinite(next_state.pipeline_state.data.qvel), axis=-1
        )
        early_non_success = (
            active & (~success) & ((episode_done & (step + 1 < horizon_steps)) | ~finite)
        )
        updated = dict(totals)
        updated["episode_steps"] = totals["episode_steps"] + active_float
        updated["success"] = jnp.maximum(
            totals["success"], active_float * success.astype(jnp.float32)
        )
        updated["stable"] = totals["stable"] + active_float * metrics["stable"]
        updated["ready"] = totals["ready"] + active_float * metrics["ready"]
        # The environment metric is the increment of its running maximum;
        # summing those increments reconstructs the episode maximum exactly.
        updated["maximum_stable_streak"] = (
            totals["maximum_stable_streak"] + active_float * metrics["maximum_stable_streak"]
        )
        updated["backward"] = (
            totals["backward"] + active_float * metrics["root_body_backward_speed"]
        )
        updated["lateral"] = totals["lateral"] + active_float * metrics["root_body_lateral_speed"]
        updated["yaw"] = totals["yaw"] + active_float * metrics["pelvis_yaw_speed"]
        updated["reward"] = totals["reward"] + active_float * metrics["reward"]
        updated["adapter_residual_rms"] = (
            totals["adapter_residual_rms"] + active_float * metrics["adapter_residual_rms"]
        )
        updated["maximum_adapter_residual_rms"] = jnp.maximum(
            totals["maximum_adapter_residual_rms"],
            active_float * metrics["adapter_residual_rms"],
        )
        updated["adapter_motor_target_delta_rms_rad"] = (
            totals["adapter_motor_target_delta_rms_rad"]
            + active_float * metrics["adapter_motor_target_delta_rms_rad"]
        )
        updated["maximum_adapter_motor_target_delta_rms_rad"] = jnp.maximum(
            totals["maximum_adapter_motor_target_delta_rms_rad"],
            active_float * metrics["adapter_motor_target_delta_rms_rad"],
        )
        updated["residual_active"] = (
            totals["residual_active"] + active_float * metrics["residual_active"]
        )
        updated["non_success_termination"] = totals["non_success_termination"] | early_non_success
        return (
            next_state,
            rng,
            active & (~episode_done) & finite,
            updated,
        ), None

    @jax.jit
    def rollout(state: Any, rng: jax.Array) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
        active = jnp.ones((environment_count,), dtype=jnp.bool_)
        final, _ = jax.lax.scan(
            rollout_step,
            (state, rng, active, accumulators),
            jnp.arange(horizon_steps, dtype=jnp.int32),
        )
        return final[3], selected_indices, failure_resets

    totals, indices, resets = rollout(initial_state, rollout_rng)
    totals["selected_failure_state_index"] = indices
    totals["failure_state_reset"] = resets.astype(jnp.float32)
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), totals)
    return {name: np.asarray(value) for name, value in totals.items()}


def run_opentrack_recovery_mjx_failure_state_exam(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    candidate_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    output_path: Path,
    source_checkout_path: Path,
    config: RecoveryMJXFailureStateExamConfig | None = None,
) -> dict[str, Any]:
    active = config or RecoveryMJXFailureStateExamConfig()
    root = opentrack_root.expanduser().resolve()
    teacher_checkpoint = teacher_checkpoint_path.expanduser().resolve()
    teacher_config = teacher_config_path.expanduser().resolve()
    parent_checkpoint = parent_actor_checkpoint_path.expanduser().resolve()
    candidate_checkpoint = candidate_actor_checkpoint_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    failure_manifest_path = failure_state_manifest_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    paths = (
        root,
        teacher_checkpoint,
        teacher_config,
        parent_checkpoint,
        candidate_checkpoint,
        snapshot_path,
        failure_manifest_path,
    )
    if (
        not all(path.exists() for path in paths)
        or not root.is_dir()
        or not teacher_checkpoint.is_dir()
        or not parent_checkpoint.is_dir()
        or not candidate_checkpoint.is_dir()
    ):
        raise FileNotFoundError("recovery MJX failure-state exam inputs are incomplete")
    if target.exists() or target == checkout or checkout in target.parents:
        raise ValueError("recovery MJX failure-state exam output must be new and external")
    if len(jax.devices()) < 4:
        raise RuntimeError("recovery MJX failure-state exam requires four visible GPUs")

    manifest = validate_recovery_mjx_failure_state_manifest(failure_manifest_path)
    if manifest.get("schema_version") != "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2":
        raise ValueError("failure-state exam requires exact policy-context manifest v2")
    parent_report_path = parent_checkpoint.parent.parent / "training-report.json"
    candidate_report_path = candidate_checkpoint.parent.parent / "training-report.json"
    parent_report = validate_recovery_mjx_teacher_residual_report(parent_report_path)
    candidate_report = validate_recovery_mjx_teacher_residual_report(candidate_report_path)
    parent_hash, parent_files = _tree_hash(parent_checkpoint)
    candidate_hash, candidate_files = _tree_hash(candidate_checkpoint)
    parent_training_tree_hash, _ = _tree_hash(parent_checkpoint.parent)
    candidate_training_tree_hash, _ = _tree_hash(candidate_checkpoint.parent)
    teacher_hash, teacher_files = _tree_hash(teacher_checkpoint)
    snapshot_file_hash = hash_bytes(snapshot_path.read_bytes())
    failure_curriculum = candidate_report.get("failure_state_curriculum")
    observation_migration = candidate_report.get("actor_observation_migration")
    actor_dimensions_compatible = bool(
        parent_report.get("actor_observation_dim") == candidate_report.get("actor_observation_dim")
        or (
            isinstance(observation_migration, dict)
            and observation_migration.get("behavior_preserved") is True
            and observation_migration.get("source_actor_observation_dim")
            == parent_report.get("actor_observation_dim")
            and observation_migration.get("target_actor_observation_dim")
            == candidate_report.get("actor_observation_dim")
        )
    )
    if (
        manifest.get("source_actor_checkpoint_hash") != parent_hash
        or parent_report.get("candidate_checkpoint_hash") != parent_training_tree_hash
        or candidate_report.get("candidate_checkpoint_hash") != candidate_training_tree_hash
        or candidate_report.get("parent_checkpoint_hash") != parent_hash
        or not isinstance(failure_curriculum, dict)
        or (
            not active.allow_blind_compatible_failure_bank
            and failure_curriculum.get("failure_state_manifest_hash") != manifest["report_hash"]
        )
        or parent_report.get("route_manifest_hash") != candidate_report.get("route_manifest_hash")
        or parent_report.get("route_group_hash") != candidate_report.get("route_group_hash")
        or manifest.get("source_route_manifest_hash") != parent_report.get("route_manifest_hash")
        or manifest.get("source_route_group_hash") != parent_report.get("route_group_hash")
        or manifest.get("teacher_checkpoint_hash") != teacher_hash
        or manifest.get("snapshot_manifest_hash") != snapshot_file_hash
        or parent_report.get("snapshot_manifest_hash") != snapshot_file_hash
        or candidate_report.get("snapshot_manifest_hash") != snapshot_file_hash
        or not actor_dimensions_compatible
    ):
        raise ValueError("recovery MJX failure-state exam lineage differs")
    parent_actor_config_payload = parent_report.get("config")
    candidate_actor_config_payload = candidate_report.get("config")
    if not isinstance(parent_actor_config_payload, dict) or not isinstance(
        candidate_actor_config_payload, dict
    ):
        raise ValueError("recovery MJX failure-state exam actor config is absent")

    def evaluation_actor_config(payload: dict[str, Any]) -> RecoveryMJXTeacherResidualPPOConfig:
        return replace(
            RecoveryMJXTeacherResidualPPOConfig(**payload),
            failure_state_reset_fraction=0.0,
            terminate_failure_state_episode_at_target_horizon=False,
            terminal_balance_reset_fraction=0.0,
            failure_state_directional_penalty_scale=0.0,
            failure_state_stable_streak_reward_scale=0.0,
            failure_state_conditioned_critic=False,
        )

    parent_actor_config = evaluation_actor_config(parent_actor_config_payload)
    candidate_actor_config = evaluation_actor_config(candidate_actor_config_payload)

    archive_path = failure_manifest_path.parent / str(manifest["state_archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        failure_bank = {name: np.array(archive[name], copy=True) for name in _BANK_ARRAYS}
        control_steps = np.array(archive["control_step"], copy=True, dtype=np.int32)
    motion_dataset_id = str(candidate_report["motion_dataset_id"])
    motion_id = str(candidate_report["motion_id"])
    entry_frame = int(candidate_report["entry_frame"])
    time_dilation = int(candidate_report["time_dilation"])
    motion_path = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1" / f"{motion_id}.npz"
    )
    if not motion_path.is_file():
        raise FileNotFoundError("recovery MJX failure-state exam motion archive is absent")

    os.environ.setdefault("GLI_PATH", str(root))
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.train.g1_env_tracking_general")
    checkpoint = importlib.import_module("brax.training.agents.ppo.checkpoint")
    teacher_payload = json.loads(teacher_config.read_text(encoding="utf-8"))
    if not isinstance(teacher_payload, dict) or not isinstance(
        teacher_payload.get("env_config"), dict
    ):
        raise ValueError("OpenTrack teacher config has no environment contract")
    environment_config = copy.deepcopy(
        tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
    )
    environment_config.update(teacher_payload["env_config"])
    environment_config.reference_traj_config.name = {motion_dataset_id: [motion_id]}
    environment_config.reference_traj_config.random_start = False
    environment_config.reference_traj_config.fixed_start_frame = entry_frame
    environment_config.noise_config.level = 0.0
    environment_config.push_config.enable = False
    environment_config.episode_length = max(3_000, active.horizon_steps + 100)
    environment_config.termination_config.root_height_threshold = 100.0
    environment_config.termination_config.rigid_body_dif_threshold = 100.0
    environment_config.termination_config.diff_gvec_threshold = 100.0
    environment_class = tmj.registry.get("G1TrackingGeneral", "tracking_train_env_class")
    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        teacher_environment = environment_class(
            terrain_type="flat_terrain", config=environment_config
        )
        trajectory_data = teacher_environment.prepare_trajectory(
            environment_config.reference_traj_config.name
        )
    finally:
        os.chdir(previous_directory)
    teacher_policy = checkpoint.load_policy(teacher_checkpoint, deterministic=True)
    parent_policy = checkpoint.load_policy(
        parent_checkpoint,
        network_factory=_make_recovery_ppo_networks,
        deterministic=True,
    )
    candidate_policy = checkpoint.load_policy(
        candidate_checkpoint,
        network_factory=_make_recovery_ppo_networks,
        deterministic=True,
    )
    all_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    selected_snapshot_indices = tuple(int(value) for value in parent_report["snapshot_indices"])
    snapshots = tuple(all_snapshots[index] for index in selected_snapshot_indices)

    def diagnostic_environment(
        actor_config: RecoveryMJXTeacherResidualPPOConfig,
        *,
        adapter_gain: float = 1.0,
    ) -> OpenTrackRecoveryMJXTeacherResidualEnv:
        return OpenTrackRecoveryMJXTeacherResidualEnv(
            teacher_environment=teacher_environment,
            trajectory_data=trajectory_data,
            teacher_policy=teacher_policy,
            snapshots=snapshots,
            time_dilation=time_dilation,
            terminal_balance_reference_frame=None,
            directional_curriculum=None,
            failure_state_bank=failure_bank,
            config=actor_config,
            parent_residual_policy=(
                parent_policy if actor_config.regularize_velocity_adapter_only else None
            ),
            diagnostic_failure_state_reset_fraction=1.0,
            diagnostic_adapter_gain=adapter_gain,
        )

    parent_environment = diagnostic_environment(parent_actor_config)
    candidate_environment = diagnostic_environment(
        candidate_actor_config,
        adapter_gain=active.candidate_adapter_gain,
    )
    if (
        compiled_mujoco_model_contract(parent_environment.mj_model)
        != manifest["compiled_model_contract"]
        or compiled_mujoco_model_contract(candidate_environment.mj_model)
        != manifest["compiled_model_contract"]
    ):
        raise ValueError("recovery MJX failure-state exam compiled model differs")
    parent_wrapped_environment = brax_training.wrap(
        parent_environment,
        episode_length=active.horizon_steps,
        action_repeat=1,
        randomization_fn=None,
    )
    candidate_wrapped_environment = brax_training.wrap(
        candidate_environment,
        episode_length=active.horizon_steps,
        action_repeat=1,
        randomization_fn=None,
    )
    reset_rng, rollout_rng = jax.random.split(jax.random.PRNGKey(active.random_seed))
    reset_keys = _stratified_failure_reset_keys(
        rng=reset_rng,
        environment_count=active.num_environments,
        failure_state_count=int(control_steps.shape[0]),
    )
    parent_result = _evaluate_actor(
        wrapped_environment=parent_wrapped_environment,
        actor_policy=parent_policy,
        reset_keys=reset_keys,
        rollout_rng=rollout_rng,
        horizon_steps=active.horizon_steps,
    )
    candidate_result = _evaluate_actor(
        wrapped_environment=candidate_wrapped_environment,
        actor_policy=candidate_policy,
        reset_keys=reset_keys,
        rollout_rng=rollout_rng,
        horizon_steps=active.horizon_steps,
    )
    parent_indices = np.asarray(parent_result.pop("selected_failure_state_index"), dtype=np.int32)
    candidate_indices = np.asarray(
        candidate_result.pop("selected_failure_state_index"), dtype=np.int32
    )
    parent_resets = np.asarray(parent_result.pop("failure_state_reset"), dtype=np.float64)
    candidate_resets = np.asarray(candidate_result.pop("failure_state_reset"), dtype=np.float64)
    if not np.array_equal(parent_indices, candidate_indices):
        raise RuntimeError("paired failure-state exam reset indices differ")
    observed_reset_fraction = float(np.mean(np.concatenate((parent_resets, candidate_resets))))
    if observed_reset_fraction != 1.0:
        raise RuntimeError("paired failure-state exam did not reset exclusively from the bank")

    candidate_action_authority = _aggregate_action_authority(
        candidate_result,
        frozen_parent_baseline=candidate_actor_config.regularize_velocity_adapter_only,
    )
    parent_metrics = _aggregate_rollout(parent_result)
    candidate_metrics = _aggregate_rollout(candidate_result)
    unique_indices = np.unique(parent_indices)
    required_steps = sorted({int(value) for value in control_steps.tolist()})
    covered_steps = sorted({int(control_steps[index]) for index in unique_indices.tolist()})
    coverage_fraction = float(unique_indices.size / control_steps.shape[0])
    coverage = {
        "unique_state_count": int(unique_indices.size),
        "total_state_count": int(control_steps.shape[0]),
        "coverage_fraction": coverage_fraction,
        "required_control_steps": required_steps,
        "covered_control_steps": covered_steps,
    }
    per_window: list[dict[str, Any]] = []
    for control_step in required_steps:
        mask = control_steps[parent_indices] == control_step
        if not np.any(mask):
            continue
        per_window.append(
            {
                "control_step": control_step,
                "episode_count": int(np.sum(mask)),
                "parent_metrics": _aggregate_rollout(
                    {name: value[mask] for name, value in parent_result.items()}
                ),
                "candidate_metrics": _aggregate_rollout(
                    {name: value[mask] for name, value in candidate_result.items()}
                ),
            }
        )

    stable_change = _relative_change(
        float(candidate_metrics["stable_fraction"]), float(parent_metrics["stable_fraction"])
    )
    streak_change = _relative_change(
        float(candidate_metrics["mean_maximum_stable_streak"]),
        float(parent_metrics["mean_maximum_stable_streak"]),
    )
    backward_change = _relative_change(
        float(candidate_metrics["root_body_backward_speed_mps"]),
        float(parent_metrics["root_body_backward_speed_mps"]),
    )
    lateral_change = _relative_change(
        float(candidate_metrics["root_body_lateral_speed_mps"]),
        float(parent_metrics["root_body_lateral_speed_mps"]),
    )
    yaw_change = _relative_change(
        float(candidate_metrics["pelvis_yaw_speed_rad_s"]),
        float(parent_metrics["pelvis_yaw_speed_rad_s"]),
    )
    success_improved = float(candidate_metrics["success_rate"]) > float(
        parent_metrics["success_rate"]
    )
    gates = {
        "coverage_passed": coverage_fraction >= active.minimum_state_coverage_fraction
        and covered_steps == required_steps,
        "success_or_stable_passed": (
            float(candidate_metrics["success_rate"]) >= float(parent_metrics["success_rate"])
            and (success_improved or stable_change >= active.minimum_stable_improvement_fraction)
        ),
        "maximum_streak_passed": streak_change >= -active.maximum_streak_regression_fraction,
        "backward_speed_passed": (
            backward_change <= -active.minimum_backward_speed_improvement_fraction
        ),
        "lateral_speed_passed": lateral_change <= active.maximum_lateral_speed_regression_fraction,
        "yaw_speed_passed": yaw_change <= active.maximum_yaw_speed_regression_fraction,
        "termination_safety_passed": (
            float(candidate_metrics["non_success_termination_rate"])
            <= float(parent_metrics["non_success_termination_rate"])
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4",
        "config": asdict(active),
        "failure_state_manifest_hash": manifest["report_hash"],
        "failure_state_manifest_file_hash": hash_bytes(failure_manifest_path.read_bytes()),
        "failure_state_archive_hash": manifest["state_archive_hash"],
        "parent_training_report_hash": parent_report["report_hash"],
        "parent_checkpoint_hash": parent_hash,
        "parent_actor_observation": parent_report["actor_observation"],
        "parent_actor_observation_dim": parent_report["actor_observation_dim"],
        "parent_checkpoint_files": parent_files,
        "parent_training_checkpoint_tree_hash": parent_training_tree_hash,
        "candidate_training_report_hash": candidate_report["report_hash"],
        "candidate_checkpoint_hash": candidate_hash,
        "candidate_actor_observation": candidate_report["actor_observation"],
        "candidate_actor_observation_dim": candidate_report["actor_observation_dim"],
        "actor_observation_migration": observation_migration,
        "candidate_checkpoint_files": candidate_files,
        "candidate_training_checkpoint_tree_hash": candidate_training_tree_hash,
        "route_manifest_hash": parent_report["route_manifest_hash"],
        "route_group_hash": parent_report["route_group_hash"],
        "teacher_checkpoint_hash": teacher_hash,
        "teacher_checkpoint_files": teacher_files,
        "motion_archive_hash": hash_bytes(motion_path.read_bytes()),
        "snapshot_manifest_hash": snapshot_file_hash,
        "parent_snapshot_manifest_hash": parent_report["snapshot_manifest_hash"],
        "candidate_snapshot_manifest_hash": candidate_report["snapshot_manifest_hash"],
        "failure_state_snapshot_manifest_hash": manifest["snapshot_manifest_hash"],
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "paired_identical_reset_keys": True,
        "paired_reset_key_strategy": "DETERMINISTIC_STRATIFIED_FULL_FAILURE_BANK_COVERAGE",
        "candidate_adapter_gain": active.candidate_adapter_gain,
        "blind_compatible_failure_bank": active.allow_blind_compatible_failure_bank,
        "diagnostic_failure_state_reset_fraction": 1.0,
        "observed_failure_state_reset_fraction": observed_reset_fraction,
        "state_coverage": coverage,
        "parent_metrics": parent_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_action_authority": candidate_action_authority,
        "relative_changes": {
            "stable_fraction": stable_change,
            "maximum_stable_streak": streak_change,
            "root_body_backward_speed": backward_change,
            "root_body_lateral_speed": lateral_change,
            "pelvis_yaw_speed": yaw_change,
        },
        "per_failure_window": per_window,
        "retention_gates": gates,
        "local_retention_passed": all(gates.values()),
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return validate_recovery_mjx_failure_state_exam_report(target)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run paired MJX recovery failure-state exam")
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--parent-actor-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-actor-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--failure-state-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--num-environments", default=384, type=int)
    parser.add_argument("--horizon-steps", default=400, type=int)
    parser.add_argument("--seed", default=5480, type=int)
    args = parser.parse_args()
    result = run_opentrack_recovery_mjx_failure_state_exam(
        opentrack_root=args.opentrack_root,
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_config_path=args.teacher_config,
        parent_actor_checkpoint_path=args.parent_actor_checkpoint,
        candidate_actor_checkpoint_path=args.candidate_actor_checkpoint,
        snapshot_manifest_path=args.snapshot_manifest,
        failure_state_manifest_path=args.failure_state_manifest,
        output_path=args.output,
        source_checkout_path=args.source_checkout,
        config=RecoveryMJXFailureStateExamConfig(
            num_environments=args.num_environments,
            horizon_steps=args.horizon_steps,
            random_seed=args.seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "RecoveryMJXFailureStateExamConfig",
    "run_opentrack_recovery_mjx_failure_state_exam",
]
