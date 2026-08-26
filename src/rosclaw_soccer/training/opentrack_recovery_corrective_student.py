"""Distil S55 corrective plans into a conservative four-GPU neural adapter."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs.wrappers import training as brax_wrappers

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.opentrack_recovery_corrective_teacher import (
    _BANK_ARRAYS,
    _pseudo_huber,
    _stratified_subset_failure_reset_keys,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_teacher_ppo import (
    OpenTrackRecoveryMJXTeacherResidualEnv,
    _tree_hash,
)
from rosclaw_soccer.training.recovery_corrective_scale import (
    write_recovery_corrective_frozen_exam_evidence,
)
from rosclaw_soccer.training.recovery_corrective_student import (
    CorrectiveTemporalLeaseConfig,
    RecoveryCorrectiveStudentConfig,
    attach_corrective_temporal_lease,
    attach_corrective_veto_aware_temporal_trigger,
    calibrate_corrective_channel_veto,
    calibrate_corrective_confidence_gate,
    corrective_stability_retention,
    derive_corrective_channel_gain,
    derive_corrective_effect_budget_gain,
    fit_corrective_channel_veto,
    fit_corrective_confidence_gate,
    fit_corrective_historical_veto_gate,
    mine_corrective_temporal_hard_negatives,
    mix_corrective_cross_domain_normal_replay,
    mix_corrective_normal_dagger_replay,
    mix_corrective_training_normal_sources,
    predict_corrective_channel_veto_numpy,
    predict_corrective_confidence_numpy,
    predict_corrective_primary_confidence_numpy,
    predict_corrective_raw_numpy,
    predict_corrective_student_numpy,
    stratified_source_split,
    validate_recovery_corrective_student_evidence,
    write_recovery_corrective_student_evidence,
)
from rosclaw_soccer.training.recovery_corrective_teacher import (
    RecoveryCorrectiveTeacherConfig,
    validate_recovery_corrective_teacher_evidence,
)
from rosclaw_soccer.training.recovery_mjx import (
    RecoveryMJXTeacherResidualPPOConfig,
    compiled_mujoco_model_contract,
    validate_recovery_mjx_failure_state_manifest,
    validate_recovery_mjx_teacher_residual_report,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus

_JOINT_COUNT = 29


@dataclass(frozen=True)
class _OpenTrackStudentContext:
    failure_environment: Any
    normal_environment: Any
    parent_policy: Any
    teacher_config: RecoveryCorrectiveTeacherConfig
    teacher_report: dict[str, Any]
    teacher_corpus: dict[str, np.ndarray[Any, Any]]
    failure_control_steps: np.ndarray[Any, Any]
    lineage: dict[str, str]


def _load_context(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
) -> _OpenTrackStudentContext:
    root = opentrack_root.expanduser().resolve()
    teacher_checkpoint = teacher_checkpoint_path.expanduser().resolve()
    teacher_config_path = teacher_config_path.expanduser().resolve()
    parent_checkpoint = parent_actor_checkpoint_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    failure_manifest_path = failure_state_manifest_path.expanduser().resolve()
    teacher_report_path = corrective_teacher_report_path.expanduser().resolve()
    if (
        not root.is_dir()
        or not teacher_checkpoint.is_dir()
        or not teacher_config_path.is_file()
        or not parent_checkpoint.is_dir()
        or not snapshot_path.is_file()
        or not failure_manifest_path.is_file()
        or not teacher_report_path.is_file()
    ):
        raise FileNotFoundError("OpenTrack corrective student inputs are incomplete")
    teacher_report = validate_recovery_corrective_teacher_evidence(teacher_report_path)
    teacher_config = RecoveryCorrectiveTeacherConfig(**teacher_report["config"])
    teacher_corpus_path = teacher_report_path.parent / str(teacher_report["corpus_archive"])
    with np.load(teacher_corpus_path, allow_pickle=False) as archive:
        teacher_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    if not np.all(teacher_corpus["teacher_accepted"]):
        raise ValueError("corrective student v1 requires accepted teacher traces only")

    manifest = validate_recovery_mjx_failure_state_manifest(failure_manifest_path)
    parent_report_path = parent_checkpoint.parent.parent / "training-report.json"
    parent_report = validate_recovery_mjx_teacher_residual_report(parent_report_path)
    parent_hash, _ = _tree_hash(parent_checkpoint)
    teacher_hash, _ = _tree_hash(teacher_checkpoint)
    snapshot_hash = hash_bytes(snapshot_path.read_bytes())
    if (
        teacher_report.get("failure_state_manifest_hash") != manifest.get("report_hash")
        or teacher_report.get("parent_training_report_hash") != parent_report.get("report_hash")
        or teacher_report.get("parent_checkpoint_hash") != parent_hash
        or teacher_report.get("teacher_checkpoint_hash") != teacher_hash
        or teacher_report.get("snapshot_manifest_hash") != snapshot_hash
        or teacher_report.get("route_manifest_hash") != parent_report.get("route_manifest_hash")
        or teacher_report.get("route_group_hash") != parent_report.get("route_group_hash")
    ):
        raise ValueError("OpenTrack corrective student lineage differs")
    actor_config_payload = parent_report.get("config")
    if not isinstance(actor_config_payload, dict):
        raise ValueError("OpenTrack corrective student parent config is absent")
    actor_config = replace(
        RecoveryMJXTeacherResidualPPOConfig(**actor_config_payload),
        failure_state_reset_fraction=0.0,
        terminate_failure_state_episode_at_target_horizon=False,
        terminal_balance_reset_fraction=0.0,
        failure_state_directional_penalty_scale=0.0,
        failure_state_stable_streak_reward_scale=0.0,
        failure_state_conditioned_critic=False,
    )
    archive_path = failure_manifest_path.parent / str(manifest["state_archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        failure_bank = {name: np.array(archive[name], copy=True) for name in _BANK_ARRAYS}
        failure_control_steps = np.array(archive["control_step"], copy=True, dtype=np.int32)
    motion_dataset_id = str(parent_report["motion_dataset_id"])
    motion_id = str(parent_report["motion_id"])
    entry_frame = int(parent_report["entry_frame"])
    time_dilation = int(parent_report["time_dilation"])
    motion_path = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1" / f"{motion_id}.npz"
    )
    if not motion_path.is_file():
        raise FileNotFoundError("OpenTrack corrective student motion archive is absent")

    os.environ.setdefault("GLI_PATH", str(root))
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.train.g1_env_tracking_general")
    checkpoint = importlib.import_module("brax.training.agents.ppo.checkpoint")
    teacher_payload = json.loads(teacher_config_path.read_text(encoding="utf-8"))
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
    environment_config.episode_length = 3_000
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
    all_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    selected_indices = tuple(int(value) for value in parent_report["snapshot_indices"])
    snapshots = tuple(all_snapshots[index] for index in selected_indices)
    common = {
        "teacher_environment": teacher_environment,
        "trajectory_data": trajectory_data,
        "teacher_policy": teacher_policy,
        "snapshots": snapshots,
        "time_dilation": time_dilation,
        "terminal_balance_reference_frame": None,
        "directional_curriculum": None,
        "config": actor_config,
        "parent_residual_policy": None,
    }
    failure_environment = OpenTrackRecoveryMJXTeacherResidualEnv(
        **common,
        failure_state_bank=failure_bank,
        diagnostic_failure_state_reset_fraction=1.0,
    )
    normal_environment = OpenTrackRecoveryMJXTeacherResidualEnv(
        **common,
        failure_state_bank=None,
        diagnostic_failure_state_reset_fraction=None,
    )
    if (
        compiled_mujoco_model_contract(failure_environment.mj_model)
        != manifest["compiled_model_contract"]
    ):
        raise ValueError("OpenTrack corrective student compiled model differs")
    lineage = {
        "teacher_report_hash": str(teacher_report["report_hash"]),
        "teacher_report_file_hash": hash_bytes(teacher_report_path.read_bytes()),
        "teacher_corpus_hash": str(teacher_report["corpus_archive_hash"]),
        "failure_state_manifest_hash": str(manifest["report_hash"]),
        "parent_checkpoint_hash": parent_hash,
        "teacher_checkpoint_hash": teacher_hash,
        "snapshot_manifest_hash": snapshot_hash,
        "route_manifest_hash": str(parent_report["route_manifest_hash"]),
        "route_group_hash": str(parent_report["route_group_hash"]),
    }
    return _OpenTrackStudentContext(
        failure_environment=failure_environment,
        normal_environment=normal_environment,
        parent_policy=parent_policy,
        teacher_config=teacher_config,
        teacher_report=teacher_report,
        teacher_corpus=teacher_corpus,
        failure_control_steps=failure_control_steps,
        lineage=lineage,
    )


def _reshape_trace(value: Any, *, state_count: int) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    # pmap + scan produces [device, step, state_per_device, ...].
    array = np.swapaxes(array, 1, 2)
    return array.reshape((state_count,) + array.shape[2:])


def _collect_failure_trace(
    *, context: _OpenTrackStudentContext, reset_keys: jax.Array
) -> tuple[dict[str, np.ndarray[Any, Any]], Any]:
    teacher_config = context.teacher_config
    state_count = teacher_config.state_count
    per_device = state_count // teacher_config.required_gpu_count
    wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=teacher_config.horizon_steps + 1,
        action_repeat=1,
    )
    initial_state = jax.pmap(wrapped.reset)(
        reset_keys.reshape((teacher_config.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        initial_state.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((state_count,))
    expected = np.asarray(context.teacher_corpus["failure_state_index"], dtype=np.int32)
    if not np.array_equal(selected, expected):
        raise RuntimeError("corrective teacher exact reset order could not be reproduced")
    plan = jnp.asarray(context.teacher_corpus["teacher_plan"], dtype=jnp.float32).reshape(
        (
            teacher_config.required_gpu_count,
            per_device,
            teacher_config.action_chunk_count,
            _JOINT_COUNT,
        )
    )
    rollout_rng = jax.random.split(
        jax.random.PRNGKey(teacher_config.random_seed + 17),
        teacher_config.required_gpu_count,
    )

    @jax.pmap
    def trace_device(
        state: Any, teacher_plan: jax.Array, rng: jax.Array
    ) -> tuple[Any, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        def step(
            carry: tuple[Any, jax.Array], step_index: jax.Array
        ) -> tuple[tuple[Any, jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            current, current_rng = carry
            current_rng, action_rng = jax.random.split(current_rng)
            parent_action, _ = context.parent_policy(current.obs, action_rng)
            delta = teacher_plan[:, step_index // teacher_config.action_chunk_steps]
            action = jnp.clip(parent_action + delta, -1.0, 1.0)
            applied = action - parent_action
            next_state = wrapped.step(current, action)
            finite = jnp.all(jnp.isfinite(next_state.pipeline_state.data.qpos), axis=-1) & jnp.all(
                jnp.isfinite(next_state.pipeline_state.data.qvel), axis=-1
            )
            output = (current.obs["state"], parent_action, applied, finite)
            return (next_state, current_rng), output

        return cast(
            tuple[Any, tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
            jax.lax.scan(
                step,
                (state, rng),
                jnp.arange(teacher_config.horizon_steps, dtype=jnp.int32),
            ),
        )

    _, outputs = trace_device(initial_state, plan, rollout_rng)
    observation, parent_action, target_increment, finite = outputs
    trace = {
        "failure_observation": _reshape_trace(observation, state_count=state_count).astype(
            np.float32
        ),
        "failure_parent_action": _reshape_trace(parent_action, state_count=state_count).astype(
            np.float32
        ),
        "failure_target_increment": _reshape_trace(
            target_increment, state_count=state_count
        ).astype(np.float32),
        "failure_state_index": selected,
        "failure_control_step": context.failure_control_steps[selected],
    }
    if not np.all(_reshape_trace(finite, state_count=state_count)):
        raise RuntimeError("corrective teacher trace became non-finite")
    return trace, wrapped


def _collect_normal_trace(
    *,
    context: _OpenTrackStudentContext,
    config: RecoveryCorrectiveStudentConfig,
    reset_keys: jax.Array,
    sample_all_steps: bool = False,
) -> tuple[dict[str, np.ndarray[Any, Any]], Any]:
    state_count = int(reset_keys.shape[0])
    if (
        state_count < config.required_gpu_count
        or state_count > context.teacher_config.state_count
        or state_count % config.required_gpu_count
    ):
        raise ValueError("normal trace source count is not four-GPU shardable")
    per_device = state_count // config.required_gpu_count
    wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=config.normal_rollout_steps + 1,
        action_repeat=1,
    )
    state = jax.pmap(wrapped.reset)(reset_keys.reshape((config.required_gpu_count, per_device, 2)))
    rng = jax.random.split(jax.random.PRNGKey(config.random_seed + 19), config.required_gpu_count)

    chunk_steps = (
        50
        if sample_all_steps and config.normal_rollout_steps % 50 == 0
        else 10
        if config.normal_rollout_steps % 10 == 0
        else 1
    )

    def chunk_device(
        state: Any, rng: jax.Array
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array]:
        def step(
            carry: tuple[Any, jax.Array], _: Any
        ) -> tuple[tuple[Any, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
            current, current_rng = carry
            current_rng, action_rng = jax.random.split(current_rng)
            parent_action, _ = context.parent_policy(current.obs, action_rng)
            next_state = wrapped.step(current, parent_action)
            finite = jnp.all(jnp.isfinite(next_state.pipeline_state.data.qpos), axis=-1) & jnp.all(
                jnp.isfinite(next_state.pipeline_state.data.qvel), axis=-1
            )
            return (next_state, current_rng), (
                current.obs["state"],
                parent_action,
                finite,
            )

        (next_state, next_rng), outputs = jax.lax.scan(step, (state, rng), None, length=chunk_steps)
        return next_state, next_rng, *outputs

    chunk = jax.pmap(chunk_device, axis_name="corrective_student_normal_trace_devices")
    sample_steps = (
        np.arange(config.normal_rollout_steps, dtype=np.int32)
        if sample_all_steps
        else np.linspace(
            0,
            config.normal_rollout_steps - 1,
            num=config.normal_sample_count_per_route,
            dtype=np.int32,
        )
    )
    sample_set = {int(value) for value in sample_steps}
    observations: list[np.ndarray[Any, Any]] = []
    parent_actions: list[np.ndarray[Any, Any]] = []
    finite = True
    for chunk_start in range(0, config.normal_rollout_steps, chunk_steps):
        state, rng, observation, parent_action, step_finite = chunk(state, rng)
        observation_trace = _reshape_trace(observation, state_count=state_count)
        parent_action_trace = _reshape_trace(parent_action, state_count=state_count)
        finite_trace = _reshape_trace(step_finite, state_count=state_count)
        for local_step in range(chunk_steps):
            if chunk_start + local_step in sample_set:
                observations.append(observation_trace[:, local_step].astype(np.float32))
                parent_actions.append(parent_action_trace[:, local_step].astype(np.float32))
        finite = finite and bool(np.all(finite_trace))
    expected_sample_count = (
        config.normal_rollout_steps if sample_all_steps else config.normal_sample_count_per_route
    )
    if (
        not finite
        or len(observations) != expected_sample_count
        or len(parent_actions) != expected_sample_count
    ):
        raise RuntimeError("normal parent trace became non-finite")
    return {
        "normal_observation": np.stack(observations, axis=1).astype(np.float32),
        "normal_parent_action": np.stack(parent_actions, axis=1).astype(np.float32),
    }, wrapped


def _collect_candidate_normal_trace(
    *,
    context: _OpenTrackStudentContext,
    config: RecoveryCorrectiveStudentConfig,
    reset_keys: jax.Array,
    model: dict[str, np.ndarray[Any, Any]],
    sample_all_steps: bool = False,
) -> tuple[dict[str, np.ndarray[Any, Any]], Any]:
    """Collect zero-correction labels on states visited by a prior student."""

    state_count = int(reset_keys.shape[0])
    if (
        state_count < config.required_gpu_count
        or state_count > context.teacher_config.state_count
        or state_count % config.required_gpu_count
    ):
        raise ValueError("candidate normal trace source count is not four-GPU shardable")
    per_device = state_count // config.required_gpu_count
    wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=config.normal_rollout_steps + 1,
        action_repeat=1,
    )
    state = jax.pmap(wrapped.reset)(reset_keys.reshape((config.required_gpu_count, per_device, 2)))
    rng = jax.random.split(jax.random.PRNGKey(config.random_seed + 19), config.required_gpu_count)
    model_jax = {name: jnp.asarray(value) for name, value in model.items()}
    temporal_state = _initial_temporal_gate_state(state.obs["state"])

    chunk_steps = (
        50
        if sample_all_steps and config.normal_rollout_steps % 50 == 0
        else 10
        if config.normal_rollout_steps % 10 == 0
        else 1
    )

    def chunk_device(
        active_state: Any, active_rng: jax.Array, active_temporal: jax.Array
    ) -> tuple[Any, ...]:
        def step(
            carry: tuple[Any, jax.Array, jax.Array], _: Any
        ) -> tuple[
            tuple[Any, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            current_state, current_rng, current_temporal = carry
            current_rng, action_rng = jax.random.split(current_rng)
            parent_action, _ = context.parent_policy(current_state.obs, action_rng)
            delta, next_temporal = _student_step(
                model_jax,
                current_state.obs["state"],
                current_temporal,
                config.maximum_action_increment,
            )
            action = jnp.clip(parent_action + delta, -1.0, 1.0)
            next_state = wrapped.step(current_state, action)
            finite = jnp.all(jnp.isfinite(next_state.pipeline_state.data.qpos), axis=-1) & jnp.all(
                jnp.isfinite(next_state.pipeline_state.data.qvel), axis=-1
            )
            return (next_state, current_rng, next_temporal), (
                current_state.obs["state"],
                parent_action,
                action - parent_action,
                finite,
            )

        (next_state, next_rng, next_temporal), outputs = jax.lax.scan(
            step, (active_state, active_rng, active_temporal), None, length=chunk_steps
        )
        return next_state, next_rng, next_temporal, *outputs

    chunk = jax.pmap(chunk_device, axis_name="corrective_student_dagger_devices")
    sample_steps = (
        np.arange(config.normal_rollout_steps, dtype=np.int32)
        if sample_all_steps
        else np.linspace(
            0,
            config.normal_rollout_steps - 1,
            num=config.normal_sample_count_per_route,
            dtype=np.int32,
        )
    )
    sample_set = {int(value) for value in sample_steps}
    observations: list[np.ndarray[Any, Any]] = []
    parent_actions: list[np.ndarray[Any, Any]] = []
    applied_increments: list[np.ndarray[Any, Any]] = []
    finite = True
    for chunk_start in range(0, config.normal_rollout_steps, chunk_steps):
        (
            state,
            rng,
            temporal_state,
            observation,
            parent_action,
            applied_increment,
            step_finite,
        ) = chunk(state, rng, temporal_state)
        observation_trace = _reshape_trace(observation, state_count=state_count)
        parent_action_trace = _reshape_trace(parent_action, state_count=state_count)
        applied_increment_trace = _reshape_trace(applied_increment, state_count=state_count)
        finite_trace = _reshape_trace(step_finite, state_count=state_count)
        for local_step in range(chunk_steps):
            if chunk_start + local_step in sample_set:
                observations.append(observation_trace[:, local_step].astype(np.float32))
                parent_actions.append(parent_action_trace[:, local_step].astype(np.float32))
                applied_increments.append(applied_increment_trace[:, local_step].astype(np.float32))
        finite = finite and bool(np.all(finite_trace))
    expected_sample_count = (
        config.normal_rollout_steps if sample_all_steps else config.normal_sample_count_per_route
    )
    if (
        not finite
        or len(observations) != expected_sample_count
        or len(parent_actions) != expected_sample_count
        or len(applied_increments) != expected_sample_count
    ):
        raise RuntimeError("candidate-visited normal DAgger trace became non-finite")
    return {
        "normal_observation": np.stack(observations, axis=1).astype(np.float32),
        "normal_parent_action": np.stack(parent_actions, axis=1).astype(np.float32),
        "normal_applied_increment": np.stack(applied_increments, axis=1).astype(np.float32),
    }, wrapped


def _student_raw_and_confidence(
    params: dict[str, jax.Array], observation: jax.Array, maximum_increment: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    value = (observation - params["observation_mean"]) / params["observation_scale"]
    value = jnp.tanh(value @ params["hidden_0_weight"] + params["hidden_0_bias"])
    value = jnp.tanh(value @ params["hidden_1_weight"] + params["hidden_1_bias"])
    veto_authority: jax.Array | float = 1.0
    if "gate_weight" in params:
        confidence = jax.nn.sigmoid(value @ params["gate_weight"] + params["gate_bias"])
        distance = jnp.sqrt(
            jnp.mean(
                jnp.square((value - params["gate_ood_center"]) / params["gate_ood_scale"]),
                axis=-1,
            )
        )
        confidence = jnp.where(
            distance[..., None] <= params["gate_ood_radius"], confidence[..., None], 0.0
        )
        if "veto_gate_weight" in params:
            veto = jax.nn.sigmoid(value @ params["veto_gate_weight"] + params["veto_gate_bias"])
            veto_distance = jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        (value - params["veto_gate_ood_center"]) / params["veto_gate_ood_scale"]
                    ),
                    axis=-1,
                )
            )
            veto = jnp.where(
                veto_distance[..., None] <= params["veto_gate_ood_radius"],
                veto[..., None],
                0.0,
            )
            if "veto_gate_minimum_authority" in params:
                floor = params["veto_gate_minimum_authority"][0]
                veto = floor + (1.0 - floor) * veto
            if "veto_gate_primary_trigger_amplitude_only" in params:
                veto_authority = veto
            else:
                confidence = confidence * veto
    else:
        confidence = 1.0
    output_gain = params.get("output_gain", jnp.ones((1,), dtype=jnp.float32))
    raw = (
        veto_authority
        * output_gain
        * maximum_increment
        * jnp.tanh(value @ params["output_weight"] + params["output_bias"])
    )
    channel_veto_mean: jax.Array | float = 1.0
    if "channel_veto_weight" in params:
        channel_veto = jax.nn.sigmoid(
            value @ params["channel_veto_weight"] + params["channel_veto_bias"]
        )
        channel_veto_distance = jnp.sqrt(
            jnp.mean(
                jnp.square(
                    (value - params["channel_veto_ood_center"]) / params["channel_veto_ood_scale"]
                ),
                axis=-1,
            )
        )
        channel_veto = jnp.where(
            channel_veto_distance[..., None] <= params["channel_veto_ood_radius"],
            channel_veto,
            0.0,
        )
        channel_veto_mean = jnp.mean(channel_veto, axis=-1, keepdims=True)
        raw = raw * channel_veto
    confidence = jnp.broadcast_to(confidence, raw.shape[:-1] + (1,))
    trigger_confidence = confidence
    if "channel_veto_temporal_trigger_mean_authority" in params:
        marker = params["channel_veto_temporal_trigger_mean_authority"]
        consensus = jnp.where(marker[0] == 2.0, jnp.square(channel_veto_mean), channel_veto_mean)
        trigger_confidence = jnp.where(
            (marker.shape == (1,)) & ((marker[0] == 1.0) | (marker[0] == 2.0)),
            confidence * consensus,
            0.0,
        )
    return raw, confidence, trigger_confidence


def _student_apply(
    params: dict[str, jax.Array], observation: jax.Array, maximum_increment: float
) -> jax.Array:
    raw, confidence, _ = _student_raw_and_confidence(params, observation, maximum_increment)
    return confidence * raw


def _initial_temporal_gate_state(observation: jax.Array) -> jax.Array:
    return jnp.zeros(observation.shape[:-1] + (4,), dtype=jnp.float32)


def _student_step(
    params: dict[str, jax.Array],
    observation: jax.Array,
    temporal_state: jax.Array,
    maximum_increment: float,
) -> tuple[jax.Array, jax.Array]:
    """Apply one corrective action while advancing optional lease state."""

    if "temporal_gate_open_threshold" not in params:
        return _student_apply(params, observation, maximum_increment), temporal_state
    raw, confidence, trigger_confidence = _student_raw_and_confidence(
        params, observation, maximum_increment
    )
    if "channel_veto_temporal_trigger_mean_authority" in params:
        marker = params["channel_veto_temporal_trigger_mean_authority"]
        confidence = jnp.where(marker[0] == 2.0, trigger_confidence, confidence)
    confidence_scalar = jnp.clip(confidence[..., 0], 0.0, 1.0)
    trigger_scalar = jnp.clip(trigger_confidence[..., 0], 0.0, 1.0)
    gate_value, open_streak, lease_remaining, cooldown_remaining = jnp.moveaxis(
        temporal_state, -1, 0
    )
    cooldown_remaining = jnp.maximum(cooldown_remaining - 1.0, 0.0)
    eligible = (lease_remaining <= 0.0) & (cooldown_remaining <= 0.0) & (gate_value <= 1.0e-6)
    qualifying = trigger_scalar >= params["temporal_gate_open_threshold"][0]
    open_streak = jnp.where(eligible & qualifying, open_streak + 1.0, 0.0)
    starting = eligible & (open_streak >= params["temporal_gate_required_open_steps"][0])
    lease_remaining = jnp.where(
        starting, params["temporal_gate_maximum_lease_steps"][0], lease_remaining
    )
    open_streak = jnp.where(starting, 0.0, open_streak)
    active = lease_remaining > 0.0
    target = jnp.where(
        active & (trigger_scalar >= params["temporal_gate_exit_threshold"][0]),
        confidence_scalar,
        0.0,
    )
    maximum_slew = params["temporal_gate_maximum_slew"][0]
    gate_value = gate_value + jnp.clip(target - gate_value, -maximum_slew, maximum_slew)
    next_lease = jnp.maximum(lease_remaining - active.astype(jnp.float32), 0.0)
    ended = active & (next_lease <= 0.0)
    cooldown_remaining = jnp.where(
        ended, params["temporal_gate_cooldown_steps"][0], cooldown_remaining
    )
    next_state = jnp.stack((gate_value, open_streak, next_lease, cooldown_remaining), axis=-1)
    return gate_value[..., None] * raw, next_state


def _train_student(
    *,
    corpus: dict[str, np.ndarray[Any, Any]],
    config: RecoveryCorrectiveStudentConfig,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
    failure_x = corpus["failure_observation"][train].reshape((-1, 384))
    failure_y = corpus["failure_target_increment"][train].reshape((-1, _JOINT_COUNT))
    normal_x = corpus["normal_observation"][train].reshape((-1, 384))
    normal_y = np.zeros((normal_x.shape[0], _JOINT_COUNT), dtype=np.float32)
    if failure_x.shape[0] != normal_x.shape[0]:
        raise ValueError("corrective student training streams are not exactly balanced")
    observation = np.concatenate((failure_x, normal_x), axis=0).astype(np.float32)
    target = np.concatenate((failure_y, normal_y), axis=0).astype(np.float32)
    random = np.random.default_rng(config.random_seed)
    order = random.permutation(observation.shape[0])
    observation = observation[order]
    target = target[order]
    mean = np.mean(observation, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(np.std(observation, axis=0, dtype=np.float64), 1.0e-3).astype(np.float32)
    key_0, key_1 = jax.random.split(jax.random.PRNGKey(config.random_seed))
    hidden_0, hidden_1 = config.hidden_sizes
    params = {
        "observation_mean": jnp.asarray(mean),
        "observation_scale": jnp.asarray(scale),
        "hidden_0_weight": jax.random.normal(key_0, (384, hidden_0), dtype=jnp.float32)
        * np.sqrt(2.0 / (384 + hidden_0)),
        "hidden_0_bias": jnp.zeros((hidden_0,), dtype=jnp.float32),
        "hidden_1_weight": jax.random.normal(key_1, (hidden_0, hidden_1), dtype=jnp.float32)
        * np.sqrt(2.0 / (hidden_0 + hidden_1)),
        "hidden_1_bias": jnp.zeros((hidden_1,), dtype=jnp.float32),
        "output_weight": jnp.zeros((hidden_1, _JOINT_COUNT), dtype=jnp.float32),
        "output_bias": jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32),
    }
    initial_prediction = _student_apply(
        params, jnp.asarray(observation[:32]), config.maximum_action_increment
    )
    if not np.array_equal(np.asarray(initial_prediction), np.zeros((32, _JOINT_COUNT))):
        raise RuntimeError("corrective student output head was not exactly zero initialized")
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    optimizer_state = optimizer.init(params)
    device_count = config.required_gpu_count
    sharded_x = jnp.asarray(observation.reshape((device_count, -1, 384)))
    sharded_y = jnp.asarray(target.reshape((device_count, -1, _JOINT_COUNT)))
    replicated_params = jax.device_put_replicated(params, jax.devices())
    replicated_optimizer_state = jax.device_put_replicated(optimizer_state, jax.devices())

    training_chunk_steps = (
        100 if config.training_steps % 100 == 0 else 10 if config.training_steps % 10 == 0 else 1
    )

    @partial(jax.pmap, axis_name="corrective_student_devices")
    def training_chunk(
        active_params: dict[str, jax.Array],
        active_optimizer_state: Any,
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> tuple[dict[str, jax.Array], Any, jax.Array]:
        def step(
            carry: tuple[dict[str, jax.Array], Any], _: Any
        ) -> tuple[tuple[dict[str, jax.Array], Any], jax.Array]:
            current_params, current_optimizer_state = carry

            def loss_fn(candidate: dict[str, jax.Array]) -> jax.Array:
                prediction = _student_apply(candidate, batch_x, config.maximum_action_increment)
                error = prediction - batch_y
                return jnp.mean(jnp.square(error))

            loss, gradient = jax.value_and_grad(loss_fn)(current_params)
            gradient = jax.lax.pmean(gradient, axis_name="corrective_student_devices")
            loss = jax.lax.pmean(loss, axis_name="corrective_student_devices")
            updates, next_optimizer_state = optimizer.update(
                gradient, current_optimizer_state, current_params
            )
            next_params = optax.apply_updates(current_params, updates)
            return (next_params, next_optimizer_state), loss

        (next_params, next_optimizer_state), chunk_loss = jax.lax.scan(
            step,
            (active_params, active_optimizer_state),
            None,
            length=training_chunk_steps,
        )
        return next_params, next_optimizer_state, chunk_loss

    losses: list[float] = []
    trace_indices = {
        value
        for value in (0, 9, 49, 99, 249, 499, 999, config.training_steps - 1)
        if value < config.training_steps
    }
    for chunk_start in range(0, config.training_steps, training_chunk_steps):
        replicated_params, replicated_optimizer_state, chunk_loss = training_chunk(
            replicated_params, replicated_optimizer_state, sharded_x, sharded_y
        )
        chunk_loss_array = np.asarray(chunk_loss)[0]
        for local_step in range(training_chunk_steps):
            if chunk_start + local_step in trace_indices:
                losses.append(float(chunk_loss_array[local_step]))
    trained = jax.tree_util.tree_map(lambda value: np.asarray(value[0]), replicated_params)
    prediction = np.asarray(
        _student_apply(
            {name: jnp.asarray(value) for name, value in trained.items()},
            jnp.asarray(observation),
            config.maximum_action_increment,
        )
    )
    return trained, {
        "algorithm": "FOUR_GPU_FULL_BATCH_ADAMW_SUPERVISED_DISTILLATION",
        "steps": config.training_steps,
        "loss_trace": losses,
        "initial_output_was_exact_zero": True,
        "balanced_failure_normal_samples": True,
        "total_training_sample_count": int(observation.shape[0]),
        "final_training_increment_rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "model_parameter_count": int(
            sum(np.asarray(value).size for value in trained.values()) - 2 * observation.shape[1]
        ),
    }


def _make_paired_exam(
    *,
    wrapped_environment: Any,
    parent_policy: Any,
    model: dict[str, np.ndarray[Any, Any]],
    horizon_steps: int,
    teacher_config: RecoveryCorrectiveTeacherConfig,
    student_config: RecoveryCorrectiveStudentConfig,
) -> Any:
    model_jax = {name: jnp.asarray(value) for name, value in model.items()}

    def exam_device(
        initial_state: Any, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        state_count = initial_state.reward.shape[0]
        zero_effect = jnp.zeros((state_count, 4), dtype=jnp.float32)
        zero_scalar = jnp.zeros((state_count,), dtype=jnp.float32)
        finite = jnp.ones((state_count,), dtype=jnp.bool_)
        zero_delta = jnp.zeros((state_count, _JOINT_COUNT), dtype=jnp.float32)

        def step(
            carry: tuple[
                Any,
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            _: jax.Array,
        ) -> tuple[
            tuple[
                Any,
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            None,
        ]:
            (
                parent_state,
                candidate_state,
                current_rng,
                parent_effect,
                candidate_effect,
                action_cost,
                slew_cost,
                prior_delta,
                finite_so_far,
            ) = carry
            current_rng, action_rng = jax.random.split(current_rng)
            parent_action, _ = parent_policy(parent_state.obs, action_rng)
            candidate_parent_action, _ = parent_policy(candidate_state.obs, action_rng)
            delta = _student_apply(
                model_jax,
                candidate_state.obs["state"],
                student_config.maximum_action_increment,
            )
            candidate_action = jnp.clip(candidate_parent_action + delta, -1.0, 1.0)
            applied_delta = candidate_action - candidate_parent_action
            next_parent = wrapped_environment.step(parent_state, parent_action)
            next_candidate = wrapped_environment.step(candidate_state, candidate_action)

            def effect(state: Any) -> jax.Array:
                metrics = state.metrics
                return jnp.stack(
                    (
                        _pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
                        _pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
                        _pseudo_huber(metrics["pelvis_yaw_speed"] / 1.5),
                        1.0 - metrics["stable"],
                    ),
                    axis=-1,
                )

            finite_parent = jnp.all(
                jnp.isfinite(next_parent.pipeline_state.data.qpos), axis=-1
            ) & jnp.all(jnp.isfinite(next_parent.pipeline_state.data.qvel), axis=-1)
            finite_candidate = jnp.all(
                jnp.isfinite(next_candidate.pipeline_state.data.qpos), axis=-1
            ) & jnp.all(jnp.isfinite(next_candidate.pipeline_state.data.qvel), axis=-1)
            parent_step_effect = jnp.nan_to_num(effect(next_parent), nan=100.0, posinf=100.0)
            candidate_step_effect = jnp.nan_to_num(effect(next_candidate), nan=100.0, posinf=100.0)
            return (
                next_parent,
                next_candidate,
                current_rng,
                parent_effect + parent_step_effect,
                candidate_effect + candidate_step_effect,
                action_cost + jnp.mean(jnp.square(applied_delta), axis=-1),
                slew_cost + jnp.mean(jnp.square(applied_delta - prior_delta), axis=-1),
                applied_delta,
                finite_so_far & finite_parent & finite_candidate,
            ), None

        final, _ = jax.lax.scan(
            step,
            (
                initial_state,
                initial_state,
                rng,
                zero_effect,
                zero_effect,
                zero_scalar,
                zero_scalar,
                zero_delta,
                finite,
            ),
            jnp.arange(horizon_steps, dtype=jnp.int32),
        )
        parent_effect = final[3] / horizon_steps
        candidate_effect = final[4] / horizon_steps
        action_cost = final[5] / horizon_steps
        slew_cost = final[6] / horizon_steps
        weights = jnp.asarray(
            (
                teacher_config.backward_cost_weight,
                teacher_config.lateral_cost_weight,
                teacher_config.yaw_cost_weight,
                teacher_config.stability_deficit_weight,
            ),
            dtype=jnp.float32,
        )
        parent_cost = jnp.sum(parent_effect * weights, axis=-1)
        candidate_cost = (
            jnp.sum(candidate_effect * weights, axis=-1)
            + teacher_config.action_magnitude_cost_weight * action_cost
            + teacher_config.action_slew_cost_weight * slew_cost
        )
        candidate_finite = jnp.all(
            jnp.isfinite(final[1].pipeline_state.data.qpos), axis=-1
        ) & jnp.all(jnp.isfinite(final[1].pipeline_state.data.qvel), axis=-1)
        parent_finite = jnp.all(jnp.isfinite(final[0].pipeline_state.data.qpos), axis=-1) & jnp.all(
            jnp.isfinite(final[0].pipeline_state.data.qvel), axis=-1
        )
        return (
            parent_cost,
            candidate_cost,
            parent_effect,
            candidate_effect,
            jnp.sqrt(action_cost),
            final[8] & candidate_finite & parent_finite,
        )

    return jax.pmap(exam_device, axis_name="corrective_student_exam_devices")


def _run_chunked_paired_exam(
    *,
    wrapped_environment: Any,
    parent_policy: Any,
    model: dict[str, np.ndarray[Any, Any]],
    initial_state: Any,
    initial_rng: jax.Array,
    horizon_steps: int,
    chunk_steps: int,
    teacher_config: RecoveryCorrectiveTeacherConfig,
    student_config: RecoveryCorrectiveStudentConfig,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Run a long paired exam through a reusable bounded-size XLA program."""

    if horizon_steps % chunk_steps:
        raise ValueError("corrective student paired exam chunks do not cover the horizon")
    model_jax = {name: jnp.asarray(value) for name, value in model.items()}

    def chunk_device(
        parent_state: Any,
        candidate_state: Any,
        rng: jax.Array,
        parent_effect: jax.Array,
        candidate_effect: jax.Array,
        action_cost: jax.Array,
        slew_cost: jax.Array,
        prior_delta: jax.Array,
        finite: jax.Array,
    ) -> tuple[
        Any, Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ]:
        def step(
            carry: tuple[
                Any,
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            _: jax.Array,
        ) -> tuple[
            tuple[
                Any,
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            None,
        ]:
            (
                active_parent,
                active_candidate,
                active_rng,
                parent_effect_sum,
                candidate_effect_sum,
                action_cost_sum,
                slew_cost_sum,
                previous_delta,
                finite_so_far,
            ) = carry
            active_rng, action_rng = jax.random.split(active_rng)
            parent_action, _ = parent_policy(active_parent.obs, action_rng)
            candidate_parent_action, _ = parent_policy(active_candidate.obs, action_rng)
            delta = _student_apply(
                model_jax,
                active_candidate.obs["state"],
                student_config.maximum_action_increment,
            )
            candidate_action = jnp.clip(candidate_parent_action + delta, -1.0, 1.0)
            applied_delta = candidate_action - candidate_parent_action
            next_parent = wrapped_environment.step(active_parent, parent_action)
            next_candidate = wrapped_environment.step(active_candidate, candidate_action)

            def effect(state: Any) -> jax.Array:
                metrics = state.metrics
                return jnp.stack(
                    (
                        _pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
                        _pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
                        _pseudo_huber(metrics["pelvis_yaw_speed"] / 1.5),
                        1.0 - metrics["stable"],
                    ),
                    axis=-1,
                )

            finite_parent = jnp.all(
                jnp.isfinite(next_parent.pipeline_state.data.qpos), axis=-1
            ) & jnp.all(jnp.isfinite(next_parent.pipeline_state.data.qvel), axis=-1)
            finite_candidate = jnp.all(
                jnp.isfinite(next_candidate.pipeline_state.data.qpos), axis=-1
            ) & jnp.all(jnp.isfinite(next_candidate.pipeline_state.data.qvel), axis=-1)
            return (
                next_parent,
                next_candidate,
                active_rng,
                parent_effect_sum + jnp.nan_to_num(effect(next_parent), nan=100.0, posinf=100.0),
                candidate_effect_sum
                + jnp.nan_to_num(effect(next_candidate), nan=100.0, posinf=100.0),
                action_cost_sum + jnp.mean(jnp.square(applied_delta), axis=-1),
                slew_cost_sum + jnp.mean(jnp.square(applied_delta - previous_delta), axis=-1),
                applied_delta,
                finite_so_far & finite_parent & finite_candidate,
            ), None

        initial = (
            parent_state,
            candidate_state,
            rng,
            parent_effect,
            candidate_effect,
            action_cost,
            slew_cost,
            prior_delta,
            finite,
        )
        if chunk_steps == 1:
            final, _ = step(initial, jnp.asarray(0, dtype=jnp.int32))
        else:
            final, _ = jax.lax.scan(
                step,
                initial,
                jnp.arange(chunk_steps, dtype=jnp.int32),
            )
        return final

    chunk = jax.pmap(chunk_device, axis_name="corrective_student_chunked_exam_devices")
    state_shape = initial_state.reward.shape
    parent_state = initial_state
    candidate_state = initial_state
    rng = initial_rng
    parent_effect = jnp.zeros(state_shape + (4,), dtype=jnp.float32)
    candidate_effect = jnp.zeros_like(parent_effect)
    action_cost = jnp.zeros(state_shape, dtype=jnp.float32)
    slew_cost = jnp.zeros_like(action_cost)
    prior_delta = jnp.zeros(state_shape + (_JOINT_COUNT,), dtype=jnp.float32)
    finite = jnp.ones(state_shape, dtype=jnp.bool_)
    for _ in range(horizon_steps // chunk_steps):
        (
            parent_state,
            candidate_state,
            rng,
            parent_effect,
            candidate_effect,
            action_cost,
            slew_cost,
            prior_delta,
            finite,
        ) = chunk(
            parent_state,
            candidate_state,
            rng,
            parent_effect,
            candidate_effect,
            action_cost,
            slew_cost,
            prior_delta,
            finite,
        )
    parent_effect = parent_effect / horizon_steps
    candidate_effect = candidate_effect / horizon_steps
    action_cost = action_cost / horizon_steps
    slew_cost = slew_cost / horizon_steps
    weights = jnp.asarray(
        (
            teacher_config.backward_cost_weight,
            teacher_config.lateral_cost_weight,
            teacher_config.yaw_cost_weight,
            teacher_config.stability_deficit_weight,
        ),
        dtype=jnp.float32,
    )
    parent_cost = jnp.sum(parent_effect * weights, axis=-1)
    candidate_cost = (
        jnp.sum(candidate_effect * weights, axis=-1)
        + teacher_config.action_magnitude_cost_weight * action_cost
        + teacher_config.action_slew_cost_weight * slew_cost
    )
    return (
        parent_cost,
        candidate_cost,
        parent_effect,
        candidate_effect,
        jnp.sqrt(action_cost),
        finite,
    )


def _run_single_policy_exam(
    *,
    wrapped_environment: Any,
    parent_policy: Any,
    model: dict[str, np.ndarray[Any, Any]],
    initial_state: Any,
    initial_rng: jax.Array,
    horizon_steps: int,
    use_student: bool,
    student_config: RecoveryCorrectiveStudentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run one policy state per graph; host looping avoids doubled MJX HLOs."""

    model_jax = {name: jnp.asarray(value) for name, value in model.items()}

    def step_device(
        state: Any,
        rng: jax.Array,
        effect_sum: jax.Array,
        action_cost_sum: jax.Array,
        slew_cost_sum: jax.Array,
        prior_delta: jax.Array,
        finite: jax.Array,
        temporal_state: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        rng, action_rng = jax.random.split(rng)
        parent_action, _ = parent_policy(state.obs, action_rng)
        if use_student:
            delta, temporal_state = _student_step(
                model_jax,
                state.obs["state"],
                temporal_state,
                student_config.maximum_action_increment,
            )
            action = jnp.clip(parent_action + delta, -1.0, 1.0)
            applied_delta = action - parent_action
        else:
            action = parent_action
            applied_delta = jnp.zeros_like(parent_action)
        next_state = wrapped_environment.step(state, action)
        metrics = next_state.metrics
        effect = jnp.stack(
            (
                _pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
                _pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
                _pseudo_huber(metrics["pelvis_yaw_speed"] / 1.5),
                1.0 - metrics["stable"],
            ),
            axis=-1,
        )
        step_finite = jnp.all(jnp.isfinite(next_state.pipeline_state.data.qpos), axis=-1) & jnp.all(
            jnp.isfinite(next_state.pipeline_state.data.qvel), axis=-1
        )
        return (
            next_state,
            rng,
            effect_sum + jnp.nan_to_num(effect, nan=100.0, posinf=100.0),
            action_cost_sum + jnp.mean(jnp.square(applied_delta), axis=-1),
            slew_cost_sum + jnp.mean(jnp.square(applied_delta - prior_delta), axis=-1),
            applied_delta,
            finite & step_finite,
            temporal_state,
        )

    chunk_steps = 10 if horizon_steps % 10 == 0 else 1

    def chunk_device(
        state: Any,
        rng: jax.Array,
        effect_sum: jax.Array,
        action_cost_sum: jax.Array,
        slew_cost_sum: jax.Array,
        prior_delta: jax.Array,
        finite: jax.Array,
        temporal_state: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def step(
            carry: tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            _: Any,
        ) -> tuple[
            tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            None,
        ]:
            return step_device(*carry), None

        final, _ = jax.lax.scan(
            step,
            (
                state,
                rng,
                effect_sum,
                action_cost_sum,
                slew_cost_sum,
                prior_delta,
                finite,
                temporal_state,
            ),
            None,
            length=chunk_steps,
        )
        return cast(
            tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            final,
        )

    chunk = jax.pmap(chunk_device, axis_name="corrective_student_single_exam_devices")
    state_shape = initial_state.reward.shape
    state = initial_state
    rng = initial_rng
    effect_sum = jnp.zeros(state_shape + (4,), dtype=jnp.float32)
    action_cost_sum = jnp.zeros(state_shape, dtype=jnp.float32)
    slew_cost_sum = jnp.zeros(state_shape, dtype=jnp.float32)
    prior_delta = jnp.zeros(state_shape + (_JOINT_COUNT,), dtype=jnp.float32)
    finite = jnp.ones(state_shape, dtype=jnp.bool_)
    temporal_state = _initial_temporal_gate_state(initial_state.obs["state"])
    for _ in range(horizon_steps // chunk_steps):
        (
            state,
            rng,
            effect_sum,
            action_cost_sum,
            slew_cost_sum,
            prior_delta,
            finite,
            temporal_state,
        ) = chunk(
            state,
            rng,
            effect_sum,
            action_cost_sum,
            slew_cost_sum,
            prior_delta,
            finite,
            temporal_state,
        )
    return (
        effect_sum / horizon_steps,
        jnp.sqrt(action_cost_sum / horizon_steps),
        slew_cost_sum / horizon_steps,
        finite,
    )


def _run_lockstep_paired_exam(
    *,
    wrapped_environment: Any,
    parent_policy: Any,
    model: dict[str, np.ndarray[Any, Any]],
    initial_state: Any,
    initial_rng: jax.Array,
    horizon_steps: int,
    teacher_config: RecoveryCorrectiveTeacherConfig,
    student_config: RecoveryCorrectiveStudentConfig,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Advance parent and candidate lanes in one compiled MJX batch.

    The previous implementation compiled the two lanes separately. Long chaotic
    trajectories could then diverge even when the archived candidate applied
    exactly zero action. Both lanes now share one Vmap batch, and a lane that
    has never intervened reuses the parent action and next state exactly. The
    coupling is released permanently after its first applied residual.
    """

    model_jax = {name: jnp.asarray(value) for name, value in model.items()}
    per_device = int(initial_state.reward.shape[1])
    paired_initial = jax.tree_util.tree_map(
        lambda value: jnp.concatenate((value, value), axis=1), initial_state
    )

    def split_lane(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        return value[:per_device], value[per_device:]

    def state_lane(state: Any, lane: int) -> Any:
        start = lane * per_device
        stop = start + per_device
        return jax.tree_util.tree_map(lambda value: value[start:stop], state)

    def effect_from_metrics(metrics: Mapping[str, jax.Array]) -> jax.Array:
        return jnp.stack(
            (
                _pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
                _pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
                _pseudo_huber(metrics["pelvis_yaw_speed"] / 1.5),
                1.0 - metrics["stable"],
            ),
            axis=-1,
        )

    def step_device(
        paired_state: Any,
        rng: jax.Array,
        parent_effect_sum: jax.Array,
        candidate_effect_sum: jax.Array,
        action_cost_sum: jax.Array,
        slew_cost_sum: jax.Array,
        prior_delta: jax.Array,
        finite: jax.Array,
        temporal_state: jax.Array,
        zero_intervention: jax.Array,
        causal_identity: jax.Array,
    ) -> tuple[
        Any,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        rng, action_rng = jax.random.split(rng)
        parent_state = state_lane(paired_state, 0)
        candidate_state = state_lane(paired_state, 1)
        parent_action, _ = parent_policy(parent_state.obs, action_rng)
        independent_candidate_action, _ = parent_policy(candidate_state.obs, action_rng)
        candidate_parent_action = jnp.where(
            zero_intervention[..., None], parent_action, independent_candidate_action
        )
        delta, temporal_state = _student_step(
            model_jax,
            candidate_state.obs["state"],
            temporal_state,
            student_config.maximum_action_increment,
        )
        candidate_action = jnp.clip(candidate_parent_action + delta, -1.0, 1.0)
        applied_delta = candidate_action - candidate_parent_action
        next_zero_intervention = zero_intervention & jnp.all(applied_delta == 0.0, axis=-1)
        paired_action = jnp.concatenate((parent_action, candidate_action), axis=0)
        next_paired_state = wrapped_environment.step(paired_state, paired_action)

        def causally_couple_zero_lane(value: jax.Array) -> jax.Array:
            parent_value, candidate_value = split_lane(value)
            mask = next_zero_intervention.reshape(
                next_zero_intervention.shape + (1,) * (candidate_value.ndim - 1)
            )
            coupled_candidate = jnp.where(mask, parent_value, candidate_value)
            return jnp.concatenate((parent_value, coupled_candidate), axis=0)

        next_paired_state = jax.tree_util.tree_map(causally_couple_zero_lane, next_paired_state)
        parent_metrics = {
            name: split_lane(value)[0] for name, value in next_paired_state.metrics.items()
        }
        candidate_metrics = {
            name: split_lane(value)[1] for name, value in next_paired_state.metrics.items()
        }
        parent_effect = effect_from_metrics(parent_metrics)
        candidate_effect = effect_from_metrics(candidate_metrics)
        parent_qpos, candidate_qpos = split_lane(next_paired_state.pipeline_state.data.qpos)
        parent_qvel, candidate_qvel = split_lane(next_paired_state.pipeline_state.data.qvel)
        step_finite = (
            jnp.all(jnp.isfinite(parent_qpos), axis=-1)
            & jnp.all(jnp.isfinite(parent_qvel), axis=-1)
            & jnp.all(jnp.isfinite(candidate_qpos), axis=-1)
            & jnp.all(jnp.isfinite(candidate_qvel), axis=-1)
        )
        exact_state_identity = jnp.all(parent_qpos == candidate_qpos, axis=-1) & jnp.all(
            parent_qvel == candidate_qvel, axis=-1
        )
        causal_identity = causal_identity & (~next_zero_intervention | exact_state_identity)
        return (
            next_paired_state,
            rng,
            parent_effect_sum + jnp.nan_to_num(parent_effect, nan=100.0, posinf=100.0),
            candidate_effect_sum + jnp.nan_to_num(candidate_effect, nan=100.0, posinf=100.0),
            action_cost_sum + jnp.mean(jnp.square(applied_delta), axis=-1),
            slew_cost_sum + jnp.mean(jnp.square(applied_delta - prior_delta), axis=-1),
            applied_delta,
            finite & step_finite,
            temporal_state,
            next_zero_intervention,
            causal_identity,
        )

    chunk_steps = 10 if horizon_steps % 10 == 0 else 1

    def chunk_device(*carry: Any) -> tuple[Any, ...]:
        final, _ = jax.lax.scan(
            lambda current, _: (step_device(*current), None),
            carry,
            None,
            length=chunk_steps,
        )
        return cast(tuple[Any, ...], final)

    chunk = jax.pmap(chunk_device, axis_name="corrective_student_lockstep_exam_devices")
    state_shape = initial_state.reward.shape
    paired_state = paired_initial
    rng = initial_rng
    parent_effect_sum = jnp.zeros(state_shape + (4,), dtype=jnp.float32)
    candidate_effect_sum = jnp.zeros(state_shape + (4,), dtype=jnp.float32)
    action_cost_sum = jnp.zeros(state_shape, dtype=jnp.float32)
    slew_cost_sum = jnp.zeros(state_shape, dtype=jnp.float32)
    prior_delta = jnp.zeros(state_shape + (_JOINT_COUNT,), dtype=jnp.float32)
    finite = jnp.ones(state_shape, dtype=jnp.bool_)
    temporal_state = _initial_temporal_gate_state(initial_state.obs["state"])
    zero_intervention = jnp.ones(state_shape, dtype=jnp.bool_)
    causal_identity = jnp.ones(state_shape, dtype=jnp.bool_)
    for _ in range(horizon_steps // chunk_steps):
        (
            paired_state,
            rng,
            parent_effect_sum,
            candidate_effect_sum,
            action_cost_sum,
            slew_cost_sum,
            prior_delta,
            finite,
            temporal_state,
            zero_intervention,
            causal_identity,
        ) = chunk(
            paired_state,
            rng,
            parent_effect_sum,
            candidate_effect_sum,
            action_cost_sum,
            slew_cost_sum,
            prior_delta,
            finite,
            temporal_state,
            zero_intervention,
            causal_identity,
        )
    zero_array = np.asarray(zero_intervention)
    identity_array = np.asarray(causal_identity)
    if np.any(zero_array & ~identity_array):
        raise RuntimeError("lockstep paired exam violated exact-zero causal identity")
    parent_effect = parent_effect_sum / horizon_steps
    candidate_effect = candidate_effect_sum / horizon_steps
    action_rms = jnp.sqrt(action_cost_sum / horizon_steps)
    slew_cost = slew_cost_sum / horizon_steps
    weights = jnp.asarray(
        (
            teacher_config.backward_cost_weight,
            teacher_config.lateral_cost_weight,
            teacher_config.yaw_cost_weight,
            teacher_config.stability_deficit_weight,
        ),
        dtype=jnp.float32,
    )
    parent_cost = jnp.sum(parent_effect * weights, axis=-1)
    candidate_cost = (
        jnp.sum(candidate_effect * weights, axis=-1)
        + teacher_config.action_magnitude_cost_weight * jnp.square(action_rms)
        + teacher_config.action_slew_cost_weight * slew_cost
    )
    return (
        parent_cost,
        candidate_cost,
        parent_effect,
        candidate_effect,
        action_rms,
        finite,
    )


def _run_separate_paired_exam(
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Compatibility name for the now lockstep paired implementation."""

    return _run_lockstep_paired_exam(**kwargs)


def _summarize_exam(
    *,
    outputs: tuple[Any, Any, Any, Any, Any, Any],
    config: RecoveryCorrectiveStudentConfig,
    normal_route: bool,
) -> dict[str, Any]:
    parent_cost, candidate_cost, parent_effect, candidate_effect, action_rms, finite = (
        np.asarray(value).reshape((-1,) + np.asarray(value).shape[2:]) for value in outputs
    )
    improvement = (parent_cost - candidate_cost) / np.maximum(np.abs(parent_cost), 1.0e-12)
    mean_parent_effect = np.mean(parent_effect, axis=0)
    mean_candidate_effect = np.mean(candidate_effect, axis=0)
    directional_tolerance = np.maximum(
        np.abs(mean_parent_effect[:3]) * config.maximum_holdout_directional_regression_fraction,
        config.maximum_holdout_directional_regression_absolute,
    )
    directional_passed = bool(
        np.all(mean_candidate_effect[:3] <= mean_parent_effect[:3] + directional_tolerance)
    )
    stability_passed, stability_tolerance = corrective_stability_retention(
        parent_effect=parent_effect,
        candidate_effect=candidate_effect,
        config=config,
        allow_configured_tolerance=normal_route,
    )
    finite_passed = bool(np.all(finite))
    mean_action_rms = float(np.sqrt(np.mean(np.square(action_rms))))
    if normal_route:
        normal_regression = float(
            (np.mean(candidate_cost) - np.mean(parent_cost))
            / max(abs(float(np.mean(parent_cost))), 1.0e-12)
        )
        passed = bool(
            finite_passed
            and normal_regression <= config.maximum_normal_cost_regression_fraction
            and mean_action_rms <= config.maximum_normal_increment_rms
            and directional_passed
            and stability_passed
        )
    else:
        normal_regression = None
        passed = bool(
            finite_passed
            and float(np.mean(improvement)) >= config.minimum_holdout_cost_improvement_fraction
            and directional_passed
            and stability_passed
        )
    return {
        "passed": passed,
        "route_kind": "NORMAL_PARENT_ROUTE" if normal_route else "UNSEEN_EXACT_FAILURE_STATES",
        "state_count": int(parent_cost.size),
        "horizon_steps": config.normal_rollout_steps if normal_route else config.trace_steps,
        "finite_fraction": float(np.mean(finite)),
        "mean_parent_cost": float(np.mean(parent_cost)),
        "mean_candidate_cost": float(np.mean(candidate_cost)),
        "mean_cost_improvement_fraction": float(np.mean(improvement)),
        "median_cost_improvement_fraction": float(np.median(improvement)),
        "minimum_cost_improvement_fraction": float(np.min(improvement)),
        "mean_action_increment_rms": mean_action_rms,
        "mean_parent_effect_metrics": [float(value) for value in mean_parent_effect],
        "mean_candidate_effect_metrics": [float(value) for value in mean_candidate_effect],
        "directional_retention_passed": directional_passed,
        "stability_retention_passed": stability_passed,
        "stability_retention_tolerance": stability_tolerance,
        "normal_cost_regression_fraction": normal_regression,
        "physics_backend": "MUJOCO_MJX",
        "paired_identical_resets": True,
        "paired_execution_semantics": (
            "LOCKSTEP_SINGLE_GRAPH_EXACT_ZERO_CAUSAL_COUPLING_SHARED_RESET_AND_ACTION_RNG"
        ),
        "exact_zero_intervention_causal_identity_enforced": True,
        "source_diagnostics": {
            "parent_cost": [float(value) for value in parent_cost],
            "candidate_cost": [float(value) for value in candidate_cost],
            "parent_effect_metrics": [
                [float(metric) for metric in source] for source in parent_effect
            ],
            "candidate_effect_metrics": [
                [float(metric) for metric in source] for source in candidate_effect
            ],
            "action_increment_rms": [float(value) for value in action_rms],
            "finite": [bool(value) for value in finite],
        },
    }


def run_opentrack_recovery_corrective_student(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: RecoveryCorrectiveStudentConfig | None = None,
) -> dict[str, Any]:
    """Collect, distil, and physically re-examine an S55 corrective student."""

    active = config or RecoveryCorrectiveStudentConfig()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective student output must be new and external")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective student requires exactly four visible GPUs")
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if (
        active.trace_steps != context.teacher_config.horizon_steps
        or active.maximum_action_increment != context.teacher_config.maximum_action_increment
    ):
        raise ValueError("corrective student and teacher action horizons differ")
    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    failure_trace, failure_wrapped = _collect_failure_trace(
        context=context, reset_keys=failure_reset_keys
    )
    normal_trace, normal_wrapped = _collect_normal_trace(
        context=context, config=active, reset_keys=normal_reset_keys
    )
    train_mask, holdout_mask = stratified_source_split(
        failure_trace["failure_control_step"],
        holdout_per_window=active.holdout_states_per_window,
        random_seed=active.random_seed,
    )
    corpus = {
        **failure_trace,
        **normal_trace,
        "train_source_mask": train_mask,
        "holdout_source_mask": holdout_mask,
    }
    model, training = _train_student(corpus=corpus, config=active)

    holdout_failure_keys = failure_reset_keys[holdout_mask]
    holdout_normal_keys = normal_reset_keys[holdout_mask]
    holdout_count = int(np.sum(holdout_mask))
    if holdout_count % active.required_gpu_count:
        raise ValueError("corrective student holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_initial = jax.pmap(failure_wrapped.reset)(
        holdout_failure_keys.reshape((active.required_gpu_count, per_device, 2))
    )
    selected_holdout = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected_holdout, failure_trace["failure_state_index"][holdout_mask]):
        raise RuntimeError("corrective student holdout reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        holdout_normal_keys.reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(jax.random.PRNGKey(active.random_seed + 31), 4)
    failure_exam_runner = _make_paired_exam(
        wrapped_environment=failure_wrapped,
        parent_policy=context.parent_policy,
        model=model,
        horizon_steps=active.trace_steps,
        teacher_config=context.teacher_config,
        student_config=active,
    )
    failure_exam = _summarize_exam(
        outputs=failure_exam_runner(failure_initial, exam_rng),
        config=active,
        normal_route=False,
    )
    normal_exam_outputs = _run_chunked_paired_exam(
        wrapped_environment=normal_wrapped,
        parent_policy=context.parent_policy,
        model=model,
        initial_state=normal_initial,
        initial_rng=exam_rng,
        horizon_steps=active.normal_rollout_steps,
        chunk_steps=1,
        teacher_config=context.teacher_config,
        student_config=active,
    )
    normal_exam = _summarize_exam(
        outputs=normal_exam_outputs,
        config=active,
        normal_route=True,
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def recheck_opentrack_recovery_corrective_student_gain(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    source_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    output_gain: float | np.ndarray[Any, Any],
    effect_budget_training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-examine a rejected student under a hash-bound conservative trust region."""

    gain_array = np.atleast_1d(np.asarray(output_gain, dtype=np.float32))
    if (
        gain_array.shape not in {(1,), (_JOINT_COUNT,)}
        or np.any(gain_array < 0.0)
        or np.any(gain_array > 1.0)
        or not np.any(gain_array > 0.0)
    ):
        raise ValueError("corrective student output gain is invalid")
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective student recheck output must be new and external")
    devices = tuple(jax.devices())
    source_path = source_student_report_path.expanduser().resolve()
    source_report = validate_recovery_corrective_student_evidence(source_path)
    active = RecoveryCorrectiveStudentConfig(**source_report["config"])
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective student recheck requires exactly four GPUs")
    with np.load(
        source_path.parent / str(source_report["corpus_archive"]), allow_pickle=False
    ) as a:
        corpus = {name: np.array(a[name], copy=True) for name in a.files}
    with np.load(source_path.parent / str(source_report["model_archive"]), allow_pickle=False) as a:
        model = {name: np.array(a[name], copy=True) for name in a.files}
    model["output_gain"] = gain_array
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(source_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("corrective student recheck lineage differs from source")
    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=active.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective student recheck failure reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(jax.random.PRNGKey(active.random_seed + 31), 4)
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    training = dict(source_report["training"])
    training.update(
        {
            "trust_region_recheck": True,
            "output_gain": (
                float(gain_array[0])
                if gain_array.shape == (1,)
                else [float(value) for value in gain_array]
            ),
            "source_student_report_hash": source_report["report_hash"],
            "source_student_report_file_hash": hash_bytes(source_path.read_bytes()),
            "source_student_development_retained": source_report["student_development_retained"],
        }
    )
    if effect_budget_training is not None:
        derived = np.asarray(effect_budget_training.get("derived_gain"), dtype=np.float32)
        if derived.shape != gain_array.shape or not np.array_equal(derived, gain_array):
            raise ValueError("corrective effect-channel budget does not bind output gain")
        training["effect_channel_budget"] = dict(effect_budget_training)
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


_G1_MIRRORED_CORRECTIVE_CHANNEL_PAIRS = (
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (4, 10),
    (5, 11),
    (15, 22),
    (16, 23),
    (17, 24),
    (18, 25),
    (19, 26),
    (20, 27),
    (21, 28),
)


def recheck_opentrack_recovery_corrective_student_effect_budget(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    source_student_report_path: Path,
    frozen_normal_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    minimum_gain_fraction: float = 0.50,
    maximum_gain_fraction: float | None = None,
) -> dict[str, Any]:
    """Derive a source-monotone channel budget without consulting old holdout labels."""

    source_path = source_student_report_path.expanduser().resolve()
    frozen_path = frozen_normal_student_report_path.expanduser().resolve()
    source_report = validate_recovery_corrective_student_evidence(source_path)
    frozen_report = validate_recovery_corrective_student_evidence(frozen_path)
    source_config = RecoveryCorrectiveStudentConfig(**source_report["config"])
    frozen_config = RecoveryCorrectiveStudentConfig(**frozen_report["config"])
    physical_config_names = (
        "maximum_action_increment",
        "trace_steps",
        "normal_rollout_steps",
        "minimum_holdout_cost_improvement_fraction",
        "maximum_holdout_directional_regression_fraction",
        "maximum_holdout_directional_regression_absolute",
        "maximum_normal_increment_rms",
        "maximum_normal_cost_regression_fraction",
        "required_gpu_count",
    )
    if any(
        getattr(source_config, name) != getattr(frozen_config, name)
        for name in physical_config_names
    ):
        raise ValueError("corrective effect-channel budget physical contracts differ")
    shared_lineage = (
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    if any(source_report.get(name) != frozen_report.get(name) for name in shared_lineage):
        raise ValueError("corrective effect-channel budget lineage differs")
    teacher_path = corrective_teacher_report_path.expanduser().resolve()
    teacher_report = validate_recovery_corrective_teacher_evidence(teacher_path)
    if (
        source_report.get("teacher_report_hash") != teacher_report.get("report_hash")
        or source_report.get("teacher_report_file_hash") != hash_bytes(teacher_path.read_bytes())
        or source_report.get("teacher_corpus_hash") != teacher_report.get("corpus_archive_hash")
    ):
        raise ValueError("corrective effect-channel teacher evidence differs")
    with np.load(
        source_path.parent / str(source_report["model_archive"]), allow_pickle=False
    ) as archive:
        model = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        source_path.parent / str(source_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        source_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        frozen_path.parent / str(frozen_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        frozen_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    teacher_corpus_path = teacher_path.parent / str(teacher_report["corpus_archive"])
    with np.load(teacher_corpus_path, allow_pickle=False) as archive:
        action_effect_jacobian = np.array(archive["action_effect_jacobian"], copy=True)
    source_train = np.asarray(source_corpus["train_source_mask"], dtype=np.bool_)
    frozen_train = np.asarray(frozen_corpus["train_source_mask"], dtype=np.bool_)
    source_failure_observation = np.asarray(source_corpus["failure_observation"])[source_train]
    historical_normal_observation = np.asarray(frozen_corpus["normal_observation"])[frozen_train]
    if (
        action_effect_jacobian.shape[0] != source_train.size
        or source_failure_observation.shape[0] != int(np.sum(source_train))
        or historical_normal_observation.shape[0] != int(np.sum(frozen_train))
    ):
        raise ValueError("corrective effect-channel training split is invalid")
    failure_prediction = predict_corrective_raw_numpy(
        model,
        source_failure_observation[:, 0],
        maximum_increment=source_config.maximum_action_increment,
    )
    historical_normal_prediction = predict_corrective_raw_numpy(
        model,
        historical_normal_observation,
        maximum_increment=source_config.maximum_action_increment,
    )
    output_gain, budget = derive_corrective_effect_budget_gain(
        action_effect_jacobian=action_effect_jacobian[source_train],
        failure_prediction=failure_prediction,
        historical_normal_prediction=historical_normal_prediction,
        base_gain=np.asarray(model.get("output_gain", np.ones((1,), dtype=np.float32))),
        mirrored_channel_pairs=_G1_MIRRORED_CORRECTIVE_CHANNEL_PAIRS,
        minimum_gain_fraction=minimum_gain_fraction,
        maximum_gain_fraction=maximum_gain_fraction,
    )
    budget.update(
        {
            "selection_split": "CURRENT_FAILURE_TRAIN_AND_FROZEN_NORMAL_TRAIN_ONLY",
            "frozen_holdout_consumed_for_selection": False,
            "failure_training_source_count": int(np.sum(source_train)),
            "historical_normal_training_source_count": int(np.sum(frozen_train)),
            "historical_normal_holdout_source_count": int(np.sum(~frozen_train)),
            "failure_prediction_content_hash": hash_bytes(failure_prediction.tobytes()),
            "historical_normal_prediction_content_hash": hash_bytes(
                historical_normal_prediction.tobytes()
            ),
            "source_student_report_hash": source_report["report_hash"],
            "source_student_report_file_hash": hash_bytes(source_path.read_bytes()),
            "frozen_normal_student_report_hash": frozen_report["report_hash"],
            "frozen_normal_student_report_file_hash": hash_bytes(frozen_path.read_bytes()),
            "frozen_normal_corpus_hash": frozen_report["corpus_archive_hash"],
        }
    )
    return recheck_opentrack_recovery_corrective_student_gain(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
        source_student_report_path=source_student_report_path,
        output_dir=output_dir,
        source_checkout_path=source_checkout_path,
        output_gain=output_gain,
        effect_budget_training=budget,
    )


def run_opentrack_recovery_corrective_channel_veto(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    source_student_report_path: Path,
    current_normal_student_report_path: Path,
    frozen_normal_student_report_path: Path,
    frozen_failure_state_manifest_path: Path,
    frozen_corrective_teacher_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    training_steps: int = 1_500,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Learn a monotone vector veto from candidate-visited frozen normal routes."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective channel-veto output must be new and external")
    source_path = source_student_report_path.expanduser().resolve()
    current_normal_path = current_normal_student_report_path.expanduser().resolve()
    frozen_path = frozen_normal_student_report_path.expanduser().resolve()
    source_report = validate_recovery_corrective_student_evidence(source_path)
    current_normal_report = validate_recovery_corrective_student_evidence(current_normal_path)
    frozen_report = validate_recovery_corrective_student_evidence(frozen_path)
    active = RecoveryCorrectiveStudentConfig(**source_report["config"])
    current_normal_config = RecoveryCorrectiveStudentConfig(**current_normal_report["config"])
    frozen_config = RecoveryCorrectiveStudentConfig(**frozen_report["config"])
    physical_config_names = (
        "maximum_action_increment",
        "trace_steps",
        "normal_rollout_steps",
        "normal_sample_count_per_route",
        "minimum_holdout_cost_improvement_fraction",
        "maximum_holdout_directional_regression_fraction",
        "maximum_holdout_directional_regression_absolute",
        "maximum_normal_increment_rms",
        "maximum_normal_cost_regression_fraction",
        "required_gpu_count",
    )
    if any(
        getattr(active, name) != getattr(candidate, name)
        for candidate in (current_normal_config, frozen_config)
        for name in physical_config_names
    ):
        raise ValueError("corrective channel-veto physical contracts differ")
    source_retained = source_report.get("student_development_retained") is True
    source_failure_exam = source_report.get("failure_state_paired_physics_exam")
    source_normal_exam = source_report.get("normal_route_paired_physics_exam")
    current_normal_training = current_normal_report.get("training")
    repair_mode = not source_retained
    if repair_mode and not (
        isinstance(source_failure_exam, dict)
        and source_failure_exam.get("passed") is True
        and source_failure_exam.get("finite_fraction") == 1.0
        and isinstance(source_normal_exam, dict)
        and source_normal_exam.get("finite_fraction") == 1.0
        and isinstance(current_normal_training, dict)
        and current_normal_training.get("dagger_source_student_report_hash")
        == source_report["report_hash"]
    ):
        raise ValueError(
            "corrective channel-veto repair requires a failure-passing source and its "
            "on-policy normal child"
        )
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective channel veto requires exactly four GPUs")
    with np.load(
        source_path.parent / str(source_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        source_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(source_path.parent / str(source_report["model_archive"]), allow_pickle=False) as a:
        model = {name: np.array(a[name], copy=True) for name in a.files}
    if any(name.startswith("channel_veto_") for name in model):
        raise ValueError("corrective channel-veto source already contains a channel veto")
    with np.load(
        current_normal_path.parent / str(current_normal_report["corpus_archive"]),
        allow_pickle=False,
    ) as archive:
        current_normal_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        frozen_path.parent / str(frozen_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        frozen_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    current_context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    frozen_context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=frozen_failure_state_manifest_path,
        corrective_teacher_report_path=frozen_corrective_teacher_report_path,
    )
    if any(source_report.get(name) != value for name, value in current_context.lineage.items()):
        raise ValueError("corrective channel-veto current runtime lineage differs")
    if any(frozen_report.get(name) != value for name, value in frozen_context.lineage.items()):
        raise ValueError("corrective channel-veto frozen runtime lineage differs")
    shared_lineage = (
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    if any(source_report.get(name) != frozen_report.get(name) for name in shared_lineage):
        raise ValueError("corrective channel-veto source and frozen lineage differ")
    # The reports intentionally have different models/training; only runtime lineage is shared.
    if any(
        source_report.get(name) != current_normal_report.get(name)
        for name in current_context.lineage
    ):
        raise ValueError("corrective channel-veto current normal lineage differs")
    identity_arrays = (
        "failure_state_index",
        "failure_control_step",
        "train_source_mask",
        "holdout_source_mask",
    )
    if any(
        not np.array_equal(source_corpus[name], current_normal_corpus[name])
        for name in identity_arrays
    ):
        raise ValueError("corrective channel-veto current normal source split differs")
    frozen_train = np.asarray(frozen_corpus["train_source_mask"], dtype=np.bool_)
    frozen_all_reset_keys = jax.random.split(
        jax.random.PRNGKey(frozen_config.random_seed + 1),
        frozen_context.teacher_config.state_count,
    )
    frozen_training_trace, _ = _collect_candidate_normal_trace(
        context=frozen_context,
        config=frozen_config,
        reset_keys=frozen_all_reset_keys[frozen_train],
        model=model,
        sample_all_steps=True,
    )
    if "temporal_gate_required_open_steps" in model:
        required_open_steps = int(np.asarray(model["temporal_gate_required_open_steps"])[0])
        primary_confidence = predict_corrective_primary_confidence_numpy(
            model, frozen_training_trace["normal_observation"]
        )
        hard_negative_score_semantics = "PRIMARY_TEMPORAL_TRIGGER_CONFIDENCE"
    else:
        # A repair source has no temporal trigger to inherit.  Preserve the
        # miner's conservative consecutive-evidence contract rather than
        # treating a single raw-action spike as a reusable normal negative.
        required_open_steps = 2
        raw_increment = predict_corrective_raw_numpy(
            model,
            frozen_training_trace["normal_observation"],
            maximum_increment=active.maximum_action_increment,
        )
        primary_confidence = np.clip(
            np.sqrt(np.mean(np.square(raw_increment), axis=-1)) / active.maximum_action_increment,
            0.0,
            1.0,
        )
        hard_negative_score_semantics = "SOURCE_RAW_INCREMENT_RMS_DIVIDED_BY_MAXIMUM_INCREMENT"
    frozen_normal_observation, frozen_normal_action, selected_index, mining = (
        mine_corrective_temporal_hard_negatives(
            observation=frozen_training_trace["normal_observation"],
            parent_action=frozen_training_trace["normal_parent_action"],
            confidence=primary_confidence,
            sample_count_per_source=active.normal_sample_count_per_route,
            consecutive_window_steps=required_open_steps,
        )
    )
    mining["score_semantics"] = hard_negative_score_semantics
    current_train = np.asarray(source_corpus["train_source_mask"], dtype=np.bool_)
    mixed_observation, mixed_action, frozen_source_mask = mix_corrective_training_normal_sources(
        current_observation=current_normal_corpus["normal_observation"],
        current_parent_action=current_normal_corpus["normal_parent_action"],
        current_train_source_mask=current_train,
        frozen_training_observation=frozen_normal_observation,
        frozen_training_parent_action=frozen_normal_action,
    )
    corpus = {
        **source_corpus,
        "normal_observation": mixed_observation,
        "normal_parent_action": mixed_action,
    }
    channel_veto, veto_training = fit_corrective_channel_veto(
        model=model,
        failure_observation=corpus["failure_observation"][current_train],
        normal_observation=corpus["normal_observation"][current_train],
        mirrored_channel_pairs=_G1_MIRRORED_CORRECTIVE_CHANNEL_PAIRS,
        training_steps=training_steps,
        batch_size=batch_size,
        random_seed=active.random_seed + 8,
    )
    model.update(channel_veto)
    training = dict(source_report["training"])
    training.update(
        {
            "channel_veto": {
                **veto_training,
                "selection_split": (
                    "CURRENT_FAILURE_TRAIN_CURRENT_NORMAL_TRAIN_"
                    "AND_FROZEN_CLOSED_LOOP_NORMAL_TRAIN_ONLY"
                ),
                "source_repair_mode": repair_mode,
                "source_failure_gate_passed": bool(
                    isinstance(source_failure_exam, dict)
                    and source_failure_exam.get("passed") is True
                ),
                "current_normal_on_policy_child_bound": bool(
                    isinstance(current_normal_training, dict)
                    and current_normal_training.get("dagger_source_student_report_hash")
                    == source_report["report_hash"]
                ),
                "current_holdout_consumed_for_selection": False,
                "frozen_holdout_consumed_for_selection": False,
                "current_training_source_count": int(np.sum(current_train)),
                "frozen_training_source_count": int(np.sum(frozen_train)),
                "frozen_closed_loop_rollout_steps": active.normal_rollout_steps,
                "frozen_closed_loop_observation_content_hash": hash_bytes(
                    np.ascontiguousarray(frozen_training_trace["normal_observation"]).tobytes()
                ),
                "frozen_closed_loop_parent_action_content_hash": hash_bytes(
                    np.ascontiguousarray(frozen_training_trace["normal_parent_action"]).tobytes()
                ),
                "frozen_selected_index_content_hash": hash_bytes(
                    np.ascontiguousarray(selected_index).tobytes()
                ),
                "frozen_replay_source_mask_content_hash": hash_bytes(
                    np.ascontiguousarray(frozen_source_mask).tobytes()
                ),
                "source_student_report_hash": source_report["report_hash"],
                "source_student_report_file_hash": hash_bytes(source_path.read_bytes()),
                "current_normal_student_report_hash": current_normal_report["report_hash"],
                "current_normal_student_report_file_hash": hash_bytes(
                    current_normal_path.read_bytes()
                ),
                "current_normal_corpus_hash": current_normal_report["corpus_archive_hash"],
                "frozen_normal_student_report_hash": frozen_report["report_hash"],
                "frozen_normal_student_report_file_hash": hash_bytes(frozen_path.read_bytes()),
                "frozen_normal_corpus_hash": frozen_report["corpus_archive_hash"],
                "frozen_failure_state_manifest_hash": frozen_report["failure_state_manifest_hash"],
                "hard_negative_mining": mining,
                "static_output_gain": [
                    float(value)
                    for value in np.broadcast_to(np.asarray(model["output_gain"]), (_JOINT_COUNT,))
                ],
            }
        }
    )
    teacher_reset_rng, _ = jax.random.split(
        jax.random.PRNGKey(current_context.teacher_config.random_seed)
    )
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=current_context.teacher_config.state_count,
        control_steps=current_context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), current_context.teacher_config.state_count
    )
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    if holdout_count % active.required_gpu_count:
        raise ValueError("corrective channel-veto holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(current_context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(current_context.normal_environment),
        episode_length=active.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective channel-veto failure reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 31), active.required_gpu_count
    )
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=current_context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=current_context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=current_context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=current_context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=current_context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def recheck_opentrack_recovery_corrective_channel_veto_calibration(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    source_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    logit_temperature: float = 2.0,
    failure_recall_logit_margin: float = 0.0,
    veto_aware_temporal_trigger: bool = False,
    scale_temporal_amplitude_by_consensus: bool = False,
) -> dict[str, Any]:
    """Recalibrate an archived vector veto and rerun the sealed current exam."""

    if scale_temporal_amplitude_by_consensus and not veto_aware_temporal_trigger:
        raise ValueError("consensus amplitude scaling requires a veto-aware trigger")

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack channel-veto calibration output must be new and external")
    devices = tuple(jax.devices())
    source_path = source_student_report_path.expanduser().resolve()
    source_report = validate_recovery_corrective_student_evidence(source_path)
    active = RecoveryCorrectiveStudentConfig(**source_report["config"])
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack channel-veto calibration requires exactly four GPUs")
    with np.load(
        source_path.parent / str(source_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        source_path.parent / str(source_report["model_archive"]), allow_pickle=False
    ) as archive:
        model = {name: np.array(archive[name], copy=True) for name in archive.files}
    source_training = source_report.get("training")
    if not isinstance(source_training, dict):
        raise ValueError("channel-veto calibration source training is absent")
    source_channel_veto = source_training.get("channel_veto")
    if not isinstance(source_channel_veto, dict):
        raise ValueError("channel-veto calibration source training is invalid")
    source_calibration = source_channel_veto.get("calibration")
    source_has_uncalibrated_arrays = all(
        name in model
        for name in (
            "channel_veto_uncalibrated_weight",
            "channel_veto_uncalibrated_bias",
        )
    )
    if failure_recall_logit_margin > 0.0 and not source_has_uncalibrated_arrays:
        raise ValueError("failure-recall margin requires archived absolute veto logits")
    if source_calibration is not None and not (
        source_calibration
        in {
            "IN_PROCESS_LOGIT_TEMPERATURE_AROUND_UNCHANGED_HALF_AUTHORITY_BOUNDARY",
            "IN_PROCESS_LOGIT_TEMPERATURE_WITH_FAILURE_RECALL_MARGIN",
        }
        and source_has_uncalibrated_arrays
    ):
        raise ValueError("channel-veto calibration source cannot be safely recalibrated")
    model.update(
        calibrate_corrective_channel_veto(
            model,
            logit_temperature=logit_temperature,
            failure_recall_logit_margin=failure_recall_logit_margin,
        )
    )
    if veto_aware_temporal_trigger:
        model.update(
            attach_corrective_veto_aware_temporal_trigger(
                model,
                scale_amplitude_by_consensus=scale_temporal_amplitude_by_consensus,
            )
        )
    train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
    failure_authority = predict_corrective_channel_veto_numpy(
        model, corpus["failure_observation"][train]
    )
    normal_authority = predict_corrective_channel_veto_numpy(
        model, corpus["normal_observation"][train]
    )
    reduction_axes = tuple(range(failure_authority.ndim - 1))
    channel_veto_training = dict(source_channel_veto)
    channel_veto_training.pop("calibration_failure_recall_logit_margin", None)
    channel_veto_training.update(
        {
            "calibration": (
                "IN_PROCESS_LOGIT_TEMPERATURE_WITH_FAILURE_RECALL_MARGIN"
                if failure_recall_logit_margin > 0.0
                else (
                    "IN_PROCESS_LOGIT_TEMPERATURE_AROUND_UNCHANGED_HALF_AUTHORITY_BOUNDARY"
                    if source_has_uncalibrated_arrays
                    else "LOGIT_TEMPERATURE_AROUND_UNCHANGED_HALF_AUTHORITY_BOUNDARY"
                )
            ),
            "calibration_logit_temperature": float(logit_temperature),
            "calibration_source_student_report_hash": source_report["report_hash"],
            "calibration_source_student_report_file_hash": hash_bytes(source_path.read_bytes()),
            "failure_mean_authority": [
                float(value) for value in np.mean(failure_authority, axis=reduction_axes)
            ],
            "normal_mean_authority": [
                float(value) for value in np.mean(normal_authority, axis=reduction_axes)
            ],
            "failure_ood_fraction": float(np.mean(np.all(failure_authority == 0.0, axis=-1))),
            "normal_ood_fraction": float(np.mean(np.all(normal_authority == 0.0, axis=-1))),
        }
    )
    if failure_recall_logit_margin > 0.0:
        channel_veto_training["calibration_failure_recall_logit_margin"] = float(
            failure_recall_logit_margin
        )
    if veto_aware_temporal_trigger:
        channel_veto_training.update(
            {
                "temporal_trigger": (
                    "PRIMARY_CONFIDENCE_TIMES_SQUARED_MEAN_CHANNEL_AUTHORITY"
                    if scale_temporal_amplitude_by_consensus
                    else "PRIMARY_CONFIDENCE_TIMES_MEAN_CHANNEL_AUTHORITY"
                ),
                "temporal_trigger_amplitude_semantics": (
                    "TRIGGER_CONFIDENCE_SETS_LEASE_AMPLITUDE"
                    if scale_temporal_amplitude_by_consensus
                    else "PRIMARY_CONFIDENCE_UNCHANGED_AFTER_TRIGGER_QUALIFIES"
                ),
                "temporal_trigger_source_student_report_hash": source_report["report_hash"],
                "temporal_trigger_source_student_report_file_hash": hash_bytes(
                    source_path.read_bytes()
                ),
            }
        )
    training = dict(source_training)
    training["channel_veto"] = channel_veto_training
    training["channel_veto_calibration_recheck"] = True
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(source_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("channel-veto calibration lineage differs from source")
    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    if holdout_count % active.required_gpu_count:
        raise ValueError("channel-veto calibration holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=active.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("channel-veto calibration failure reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 31), active.required_gpu_count
    )
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def run_opentrack_recovery_corrective_channel_veto_dagger(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    base_student_report_path: Path,
    candidate_student_report_path: Path,
    frozen_normal_student_report_path: Path,
    frozen_failure_state_manifest_path: Path,
    frozen_corrective_teacher_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    training_steps: int = 1_500,
    batch_size: int = 512,
    logit_temperature: float = 2.0,
    failure_recall_logit_margin: float = 0.0,
) -> dict[str, Any]:
    """Retrain a vector veto on its own current and frozen closed-loop mistakes."""

    if (
        not np.isfinite(failure_recall_logit_margin)
        or not 0.0 <= failure_recall_logit_margin <= 8.0
        or (failure_recall_logit_margin > 0.0 and logit_temperature <= 1.0)
    ):
        raise ValueError("corrective channel-veto DAgger recall margin is invalid")
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack channel-veto DAgger output must be new and external")
    base_path = base_student_report_path.expanduser().resolve()
    candidate_path = candidate_student_report_path.expanduser().resolve()
    frozen_path = frozen_normal_student_report_path.expanduser().resolve()
    base_report = validate_recovery_corrective_student_evidence(base_path)
    candidate_report = validate_recovery_corrective_student_evidence(candidate_path)
    frozen_report = validate_recovery_corrective_student_evidence(frozen_path)
    active = RecoveryCorrectiveStudentConfig(**base_report["config"])
    candidate_config = RecoveryCorrectiveStudentConfig(**candidate_report["config"])
    frozen_config = RecoveryCorrectiveStudentConfig(**frozen_report["config"])
    if active != candidate_config or any(
        getattr(active, name) != getattr(frozen_config, name)
        for name in (
            "maximum_action_increment",
            "trace_steps",
            "normal_rollout_steps",
            "normal_sample_count_per_route",
            "minimum_holdout_cost_improvement_fraction",
            "maximum_holdout_directional_regression_fraction",
            "maximum_holdout_directional_regression_absolute",
            "maximum_normal_increment_rms",
            "maximum_normal_cost_regression_fraction",
            "required_gpu_count",
        )
    ):
        raise ValueError("corrective channel-veto DAgger physical contracts differ")
    if base_report.get("student_development_retained") is not True:
        raise ValueError("corrective channel-veto DAgger base did not pass its development gate")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack channel-veto DAgger requires exactly four GPUs")
    with np.load(
        base_path.parent / str(base_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        base_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        base_path.parent / str(base_report["model_archive"]), allow_pickle=False
    ) as archive:
        base_model = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        candidate_path.parent / str(candidate_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        candidate_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        candidate_path.parent / str(candidate_report["model_archive"]), allow_pickle=False
    ) as archive:
        candidate_model = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        frozen_path.parent / str(frozen_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        frozen_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    if any(name.startswith("channel_veto_") for name in base_model) or not all(
        name in candidate_model
        for name in (
            "channel_veto_weight",
            "channel_veto_bias",
            "channel_veto_ood_center",
            "channel_veto_ood_scale",
            "channel_veto_ood_radius",
        )
    ):
        raise ValueError("corrective channel-veto DAgger model roles are invalid")
    identity_arrays = (
        "failure_state_index",
        "failure_control_step",
        "train_source_mask",
        "holdout_source_mask",
    )
    if any(
        not np.array_equal(base_corpus[name], candidate_corpus[name]) for name in identity_arrays
    ):
        raise ValueError("corrective channel-veto DAgger current source split differs")
    current_context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    frozen_context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=frozen_failure_state_manifest_path,
        corrective_teacher_report_path=frozen_corrective_teacher_report_path,
    )
    if any(base_report.get(name) != value for name, value in current_context.lineage.items()):
        raise ValueError("corrective channel-veto DAgger current lineage differs")
    if any(candidate_report.get(name) != value for name, value in current_context.lineage.items()):
        raise ValueError("corrective channel-veto DAgger candidate lineage differs")
    if any(frozen_report.get(name) != value for name, value in frozen_context.lineage.items()):
        raise ValueError("corrective channel-veto DAgger frozen lineage differs")
    shared_lineage = (
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    if any(base_report.get(name) != frozen_report.get(name) for name in shared_lineage):
        raise ValueError("corrective channel-veto DAgger domains are not comparable")

    current_train = np.asarray(base_corpus["train_source_mask"], dtype=np.bool_)
    frozen_train = np.asarray(frozen_corpus["train_source_mask"], dtype=np.bool_)
    current_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1),
        current_context.teacher_config.state_count,
    )[current_train]
    frozen_reset_keys = jax.random.split(
        jax.random.PRNGKey(frozen_config.random_seed + 1),
        frozen_context.teacher_config.state_count,
    )[frozen_train]
    current_trace, _ = _collect_candidate_normal_trace(
        context=current_context,
        config=active,
        reset_keys=current_reset_keys,
        model=candidate_model,
        sample_all_steps=True,
    )
    frozen_trace, _ = _collect_candidate_normal_trace(
        context=frozen_context,
        config=frozen_config,
        reset_keys=frozen_reset_keys,
        model=candidate_model,
        sample_all_steps=True,
    )
    required_open_steps = int(np.asarray(candidate_model["temporal_gate_required_open_steps"])[0])

    def mine_trace(
        trace: Mapping[str, np.ndarray[Any, Any]],
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        dict[str, Any],
    ]:
        applied = np.asarray(trace["normal_applied_increment"], dtype=np.float32)
        intervention_score = np.clip(
            np.sqrt(np.mean(np.square(applied), axis=-1)) / active.maximum_action_increment,
            0.0,
            1.0,
        )
        observation, parent_action, index, mining = mine_corrective_temporal_hard_negatives(
            observation=trace["normal_observation"],
            parent_action=trace["normal_parent_action"],
            confidence=intervention_score,
            sample_count_per_source=active.normal_sample_count_per_route,
            consecutive_window_steps=required_open_steps,
        )
        mining["score_semantics"] = "ACTUAL_APPLIED_INCREMENT_RMS_DIVIDED_BY_MAXIMUM_INCREMENT"
        mining["full_trace_applied_increment_rms"] = float(np.sqrt(np.mean(np.square(applied))))
        return observation, parent_action, index, mining

    current_observation, current_action, current_index, current_mining = mine_trace(current_trace)
    frozen_observation, frozen_action, frozen_index, frozen_mining = mine_trace(frozen_trace)
    current_normal_observation = np.asarray(
        candidate_corpus["normal_observation"], dtype=np.float32
    ).copy()
    current_normal_action = np.asarray(
        candidate_corpus["normal_parent_action"], dtype=np.float32
    ).copy()
    current_normal_observation[current_train] = current_observation
    current_normal_action[current_train] = current_action
    mixed_observation, mixed_action, frozen_source_mask = mix_corrective_training_normal_sources(
        current_observation=current_normal_observation,
        current_parent_action=current_normal_action,
        current_train_source_mask=current_train,
        frozen_training_observation=frozen_observation,
        frozen_training_parent_action=frozen_action,
    )
    corpus = {
        **base_corpus,
        "normal_observation": mixed_observation,
        "normal_parent_action": mixed_action,
        "dagger_current_applied_increment": current_trace["normal_applied_increment"],
        "dagger_current_selected_index": current_index,
        "dagger_frozen_applied_increment": frozen_trace["normal_applied_increment"],
        "dagger_frozen_selected_index": frozen_index,
        "dagger_frozen_replay_source_mask": frozen_source_mask,
    }
    channel_veto, veto_training = fit_corrective_channel_veto(
        model=base_model,
        failure_observation=corpus["failure_observation"][current_train],
        normal_observation=corpus["normal_observation"][current_train],
        mirrored_channel_pairs=_G1_MIRRORED_CORRECTIVE_CHANNEL_PAIRS,
        training_steps=training_steps,
        batch_size=batch_size,
        logit_temperature=logit_temperature,
        random_seed=active.random_seed + 9,
    )
    if failure_recall_logit_margin > 0.0:
        channel_veto.update(
            calibrate_corrective_channel_veto(
                channel_veto,
                logit_temperature=logit_temperature,
                failure_recall_logit_margin=failure_recall_logit_margin,
            )
        )
        calibrated_model = {**base_model, **channel_veto}
        failure_authority = predict_corrective_channel_veto_numpy(
            calibrated_model, corpus["failure_observation"][current_train]
        )
        normal_authority = predict_corrective_channel_veto_numpy(
            calibrated_model, corpus["normal_observation"][current_train]
        )
        reduction_axes = tuple(range(failure_authority.ndim - 1))
        veto_training.update(
            {
                "calibration": "IN_PROCESS_LOGIT_TEMPERATURE_WITH_FAILURE_RECALL_MARGIN",
                "calibration_failure_recall_logit_margin": float(failure_recall_logit_margin),
                "failure_mean_authority": [
                    float(value) for value in np.mean(failure_authority, axis=reduction_axes)
                ],
                "normal_mean_authority": [
                    float(value) for value in np.mean(normal_authority, axis=reduction_axes)
                ],
                "failure_ood_fraction": float(np.mean(np.all(failure_authority == 0.0, axis=-1))),
                "normal_ood_fraction": float(np.mean(np.all(normal_authority == 0.0, axis=-1))),
            }
        )
    model = {**base_model, **channel_veto}
    training = dict(base_report["training"])
    training["channel_veto"] = {
        **veto_training,
        "selection_split": "CURRENT_AND_FROZEN_CANDIDATE_CLOSED_LOOP_NORMAL_TRAIN_ONLY",
        "current_holdout_consumed_for_selection": False,
        "frozen_holdout_consumed_for_selection": False,
        "current_training_source_count": int(np.sum(current_train)),
        "frozen_training_source_count": int(np.sum(frozen_train)),
        "current_closed_loop_rollout_steps": active.normal_rollout_steps,
        "frozen_closed_loop_rollout_steps": frozen_config.normal_rollout_steps,
        "current_closed_loop_observation_content_hash": hash_bytes(
            np.ascontiguousarray(current_trace["normal_observation"]).tobytes()
        ),
        "current_closed_loop_parent_action_content_hash": hash_bytes(
            np.ascontiguousarray(current_trace["normal_parent_action"]).tobytes()
        ),
        "current_closed_loop_applied_increment_content_hash": hash_bytes(
            np.ascontiguousarray(current_trace["normal_applied_increment"]).tobytes()
        ),
        "current_selected_index_content_hash": hash_bytes(
            np.ascontiguousarray(current_index).tobytes()
        ),
        "frozen_closed_loop_observation_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_trace["normal_observation"]).tobytes()
        ),
        "frozen_closed_loop_parent_action_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_trace["normal_parent_action"]).tobytes()
        ),
        "frozen_closed_loop_applied_increment_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_trace["normal_applied_increment"]).tobytes()
        ),
        "frozen_selected_index_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_index).tobytes()
        ),
        "frozen_replay_source_mask_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_source_mask).tobytes()
        ),
        "source_student_report_hash": base_report["report_hash"],
        "source_student_report_file_hash": hash_bytes(base_path.read_bytes()),
        "current_normal_student_report_hash": candidate_report["report_hash"],
        "current_normal_student_report_file_hash": hash_bytes(candidate_path.read_bytes()),
        "current_normal_corpus_hash": candidate_report["corpus_archive_hash"],
        "dagger_candidate_student_report_hash": candidate_report["report_hash"],
        "dagger_candidate_student_report_file_hash": hash_bytes(candidate_path.read_bytes()),
        "frozen_normal_student_report_hash": frozen_report["report_hash"],
        "frozen_normal_student_report_file_hash": hash_bytes(frozen_path.read_bytes()),
        "frozen_normal_corpus_hash": frozen_report["corpus_archive_hash"],
        "frozen_failure_state_manifest_hash": frozen_report["failure_state_manifest_hash"],
        "hard_negative_mining": {
            "current": current_mining,
            "frozen": frozen_mining,
        },
        "static_output_gain": [
            float(value)
            for value in np.broadcast_to(
                np.asarray(model["output_gain"], dtype=np.float32), (_JOINT_COUNT,)
            )
        ],
    }

    teacher_reset_rng, _ = jax.random.split(
        jax.random.PRNGKey(current_context.teacher_config.random_seed)
    )
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=current_context.teacher_config.state_count,
        control_steps=current_context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), current_context.teacher_config.state_count
    )
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    if holdout_count % active.required_gpu_count:
        raise ValueError("channel-veto DAgger holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(current_context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(current_context.normal_environment),
        episode_length=active.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("channel-veto DAgger failure reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 31), active.required_gpu_count
    )
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=current_context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=current_context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=current_context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=current_context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=current_context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def recheck_opentrack_recovery_corrective_student_temporal_lease(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    source_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    temporal_config: CorrectiveTemporalLeaseConfig | None = None,
    authority_scale: float = 1.0,
    frozen_normal_veto_student_report_path: Path | None = None,
    historical_veto_minimum_authority: float = 0.0,
) -> dict[str, Any]:
    """Re-examine a learned gate under a stateful, finite intervention lease."""

    temporal = temporal_config or CorrectiveTemporalLeaseConfig()
    if not np.isfinite(authority_scale) or not 0.05 <= authority_scale <= 1.0:
        raise ValueError("corrective temporal-lease authority scale is invalid")
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective temporal-lease output must be new and external")
    devices = tuple(jax.devices())
    source_path = source_student_report_path.expanduser().resolve()
    source_report = validate_recovery_corrective_student_evidence(source_path)
    active = RecoveryCorrectiveStudentConfig(**source_report["config"])
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective temporal lease requires exactly four GPUs")
    with np.load(
        source_path.parent / str(source_report["corpus_archive"]), allow_pickle=False
    ) as a:
        corpus = {name: np.array(a[name], copy=True) for name in a.files}
    with np.load(source_path.parent / str(source_report["model_archive"]), allow_pickle=False) as a:
        model = {name: np.array(a[name], copy=True) for name in a.files}
    historical_veto_training: dict[str, Any] | None = None
    if frozen_normal_veto_student_report_path is not None:
        frozen_path = frozen_normal_veto_student_report_path.expanduser().resolve()
        frozen_report = validate_recovery_corrective_student_evidence(frozen_path)
        frozen_config = RecoveryCorrectiveStudentConfig(**frozen_report["config"])
        physical_config_names = (
            "maximum_action_increment",
            "trace_steps",
            "normal_sample_count_per_route",
            "normal_rollout_steps",
            "required_gpu_count",
        )
        if any(
            getattr(active, name) != getattr(frozen_config, name) for name in physical_config_names
        ):
            raise ValueError("corrective historical veto physical contracts differ")
        shared_lineage = (
            "parent_checkpoint_hash",
            "teacher_checkpoint_hash",
            "snapshot_manifest_hash",
            "route_manifest_hash",
            "route_group_hash",
        )
        if any(source_report.get(name) != frozen_report.get(name) for name in shared_lineage):
            raise ValueError("corrective historical veto lineage differs")
        with np.load(
            frozen_path.parent / str(frozen_report["corpus_archive"]), allow_pickle=False
        ) as archive:
            frozen_normal = np.array(archive["normal_observation"], copy=True)
            frozen_train = np.array(archive["train_source_mask"], copy=True, dtype=np.bool_)
        source_train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
        failure_train = np.asarray(corpus["failure_observation"])[source_train]
        frozen_normal_train = frozen_normal[frozen_train]
        if failure_train.shape[0] < frozen_normal_train.shape[0]:
            raise ValueError("corrective historical veto has too few failure sources")
        veto, historical_veto_training = fit_corrective_historical_veto_gate(
            model=model,
            failure_observation=failure_train[: frozen_normal_train.shape[0]],
            frozen_normal_observation=frozen_normal_train,
            minimum_authority=historical_veto_minimum_authority,
        )
        model.update(veto)
        historical_veto_training.update(
            {
                "failure_training_source_count": int(frozen_normal_train.shape[0]),
                "frozen_normal_training_source_count": int(frozen_normal_train.shape[0]),
                "frozen_holdout_source_count": int(np.sum(~frozen_train)),
                "frozen_student_report_hash": frozen_report["report_hash"],
                "frozen_student_report_file_hash": hash_bytes(frozen_path.read_bytes()),
                "frozen_corpus_archive_hash": frozen_report["corpus_archive_hash"],
            }
        )
    model.update(attach_corrective_temporal_lease(model, temporal))
    source_output_gain = np.asarray(
        model.get("output_gain", np.ones((1,), dtype=np.float32)), dtype=np.float32
    )
    model["output_gain"] = np.asarray(source_output_gain * authority_scale, dtype=np.float32)
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(source_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("corrective temporal-lease lineage differs from source")
    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    if holdout_count % active.required_gpu_count:
        raise ValueError("corrective temporal-lease holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=active.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective temporal-lease holdout reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(jax.random.PRNGKey(active.random_seed + 31), 4)
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    training = dict(source_report["training"])
    training.update(
        {
            "temporal_intervention_lease": asdict(temporal),
            "temporal_intervention_lease_config_hash": temporal.config_hash,
            "temporal_intervention_semantics": "EVIDENCE_ACCUMULATION_FINITE_LEASE_COOLDOWN",
            "temporal_authority_scale": float(authority_scale),
            "source_output_gain": [float(value) for value in source_output_gain],
            "scaled_output_gain": [float(value) for value in model["output_gain"]],
            "source_student_report_hash": source_report["report_hash"],
            "source_student_report_file_hash": hash_bytes(source_path.read_bytes()),
            "source_student_development_retained": source_report["student_development_retained"],
        }
    )
    if historical_veto_training is not None:
        training["historical_normal_veto"] = historical_veto_training
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def run_opentrack_recovery_corrective_temporal_hard_negative_gate(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    base_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    confidence_threshold: float = 0.75,
    confidence_logit_temperature: float = 8.0,
    hard_negative_refinement_rounds: int = 1,
    failure_prefix_steps: int = 0,
    failure_prefix_weight: float = 1.0,
    temporal_config: CorrectiveTemporalLeaseConfig | None = None,
    authority_scale: float = 1.0,
    frozen_normal_replay_student_report_path: Path | None = None,
) -> dict[str, Any]:
    """Refit silence authority from full-route consecutive false positives.

    Unlike sparse normal replay, this round scores every frame of the frozen
    parent's 600-step route.  It mines source-local non-overlapping windows
    whose *minimum* confidence is highest, then refits the gate with the
    original failure traces while preserving the source-disjoint holdout.
    """

    temporal = temporal_config or CorrectiveTemporalLeaseConfig(
        open_threshold=0.95,
        exit_threshold=0.02,
        required_open_steps=2,
        maximum_lease_steps=20,
        cooldown_steps=600,
        maximum_slew=1.0,
    )
    if isinstance(hard_negative_refinement_rounds, bool) or hard_negative_refinement_rounds not in {
        1,
        2,
    }:
        raise ValueError("corrective hard-negative refinement rounds must be one or two")
    if (
        isinstance(failure_prefix_steps, bool)
        or not 0 <= failure_prefix_steps <= 80
        or not np.isfinite(failure_prefix_weight)
        or not 1.0 <= failure_prefix_weight <= 100.0
        or (failure_prefix_steps == 0) != (failure_prefix_weight == 1.0)
    ):
        raise ValueError("corrective hard-negative failure-prefix weighting is invalid")
    if not np.isfinite(authority_scale) or not 0.05 <= authority_scale <= 1.0:
        raise ValueError("corrective hard-negative authority scale is invalid")
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective hard-negative output must be new and external")
    base_path = base_student_report_path.expanduser().resolve()
    base_report = validate_recovery_corrective_student_evidence(base_path)
    active = RecoveryCorrectiveStudentConfig(**base_report["config"])
    if failure_prefix_steps > active.trace_steps:
        raise ValueError("corrective hard-negative failure prefix exceeds the trace")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective hard-negative gate requires exactly four GPUs")
    with np.load(base_path.parent / str(base_report["corpus_archive"]), allow_pickle=False) as a:
        corpus = {name: np.array(a[name], copy=True) for name in a.files}
    with np.load(base_path.parent / str(base_report["model_archive"]), allow_pickle=False) as a:
        model = {name: np.array(a[name], copy=True) for name in a.files}
    temporal_model_names = {
        "temporal_gate_open_threshold",
        "temporal_gate_exit_threshold",
        "temporal_gate_required_open_steps",
        "temporal_gate_maximum_lease_steps",
        "temporal_gate_cooldown_steps",
        "temporal_gate_maximum_slew",
    }
    present_temporal_names = frozenset(name for name in model if name.startswith("temporal_gate_"))
    if "gate_weight" not in model or present_temporal_names not in {
        frozenset(),
        frozenset(temporal_model_names),
    }:
        raise ValueError("corrective hard-negative base has an incomplete confidence lease")
    for name in temporal_model_names:
        model.pop(name, None)

    frozen_report: dict[str, Any] | None = None
    frozen_path: Path | None = None
    frozen_corpus: dict[str, np.ndarray[Any, Any]] | None = None
    if frozen_normal_replay_student_report_path is not None:
        frozen_path = frozen_normal_replay_student_report_path.expanduser().resolve()
        frozen_report = validate_recovery_corrective_student_evidence(frozen_path)
        frozen_config = RecoveryCorrectiveStudentConfig(**frozen_report["config"])
        physical_config_names = (
            "maximum_action_increment",
            "trace_steps",
            "normal_sample_count_per_route",
            "normal_rollout_steps",
            "required_gpu_count",
        )
        if any(
            getattr(active, name) != getattr(frozen_config, name) for name in physical_config_names
        ):
            raise ValueError("corrective frozen normal replay physical contracts differ")
        shared_lineage = (
            "parent_checkpoint_hash",
            "teacher_checkpoint_hash",
            "snapshot_manifest_hash",
            "route_manifest_hash",
            "route_group_hash",
        )
        if any(base_report.get(name) != frozen_report.get(name) for name in shared_lineage):
            raise ValueError("corrective frozen normal replay lineage differs")
        with np.load(
            frozen_path.parent / str(frozen_report["corpus_archive"]), allow_pickle=False
        ) as archive:
            frozen_corpus = {
                "normal_observation": np.array(archive["normal_observation"], copy=True),
                "normal_parent_action": np.array(archive["normal_parent_action"], copy=True),
            }

    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(base_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("corrective hard-negative runtime lineage differs")
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    full_normal_trace, normal_wrapped = _collect_normal_trace(
        context=context,
        config=active,
        reset_keys=normal_reset_keys,
        sample_all_steps=True,
    )
    source_confidence = predict_corrective_confidence_numpy(
        model, full_normal_trace["normal_observation"]
    )
    if hard_negative_refinement_rounds == 1:
        normal_observation, normal_parent_action, selected_index, mining = (
            mine_corrective_temporal_hard_negatives(
                observation=full_normal_trace["normal_observation"],
                parent_action=full_normal_trace["normal_parent_action"],
                confidence=source_confidence,
                sample_count_per_source=active.normal_sample_count_per_route,
                consecutive_window_steps=temporal.required_open_steps,
            )
        )
        mining["refinement_rounds"] = 1
    else:
        half_sample_count = active.normal_sample_count_per_route // 2
        if (
            active.normal_sample_count_per_route % 2
            or half_sample_count % temporal.required_open_steps
        ):
            raise ValueError("corrective hard-negative replay cannot be evenly refined")
        first_observation, _, first_index, first_mining = mine_corrective_temporal_hard_negatives(
            observation=full_normal_trace["normal_observation"],
            parent_action=full_normal_trace["normal_parent_action"],
            confidence=source_confidence,
            sample_count_per_source=half_sample_count,
            consecutive_window_steps=temporal.required_open_steps,
        )
        train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
        interim_normal = np.concatenate((first_observation, first_observation), axis=1)
        interim_failure_weight = np.ones(
            corpus["failure_observation"][train].shape[:-1], dtype=np.float32
        )
        if failure_prefix_steps:
            interim_failure_weight[:, :failure_prefix_steps] = failure_prefix_weight
        interim_gate, interim_training = fit_corrective_confidence_gate(
            model=model,
            failure_observation=corpus["failure_observation"][train],
            normal_observation=interim_normal[train],
            failure_sample_weight=interim_failure_weight,
        )
        interim_model = {**model, **interim_gate}
        interim_model.update(
            calibrate_corrective_confidence_gate(
                interim_model,
                threshold=confidence_threshold,
                logit_temperature=confidence_logit_temperature,
            )
        )
        refreshed_confidence = predict_corrective_confidence_numpy(
            interim_model, full_normal_trace["normal_observation"]
        )
        excluded = np.zeros(refreshed_confidence.shape[:2], dtype=np.bool_)
        source_index = np.arange(excluded.shape[0], dtype=np.int32)[:, None]
        excluded[source_index, first_index] = True
        _, _, second_index, second_mining = mine_corrective_temporal_hard_negatives(
            observation=full_normal_trace["normal_observation"],
            parent_action=full_normal_trace["normal_parent_action"],
            confidence=refreshed_confidence,
            sample_count_per_source=half_sample_count,
            consecutive_window_steps=temporal.required_open_steps,
            excluded_index_mask=excluded,
        )
        selected_index = np.sort(np.concatenate((first_index, second_index), axis=1), axis=1)
        if np.any(np.diff(selected_index, axis=1) == 0):
            raise RuntimeError("corrective hard-negative refinement reused a frame")
        normal_observation = full_normal_trace["normal_observation"][
            source_index, selected_index
        ].astype(np.float32)
        normal_parent_action = full_normal_trace["normal_parent_action"][
            source_index, selected_index
        ].astype(np.float32)
        mining = {
            "algorithm": "TWO_ROUND_SOURCE_LOCAL_BOUNDARY_REFRESH",
            "refinement_rounds": 2,
            "source_count": int(selected_index.shape[0]),
            "rollout_steps": int(full_normal_trace["normal_observation"].shape[1]),
            "sample_count_per_source": int(active.normal_sample_count_per_route),
            "consecutive_window_steps": int(temporal.required_open_steps),
            "first_round": first_mining,
            "second_round": second_mining,
            "interim_gate": interim_training,
        }
    frozen_domain_mask = np.zeros((normal_observation.shape[0],), dtype=np.bool_)
    if frozen_corpus is not None:
        normal_observation, normal_parent_action, frozen_domain_mask = (
            mix_corrective_cross_domain_normal_replay(
                current_observation=normal_observation,
                current_parent_action=normal_parent_action,
                frozen_observation=frozen_corpus["normal_observation"],
                frozen_parent_action=frozen_corpus["normal_parent_action"],
            )
        )
        if frozen_report is None or frozen_path is None:
            raise RuntimeError("corrective frozen normal replay provenance is absent")
        mining["cross_domain_replay"] = {
            "algorithm": "ALTERNATING_CURRENT_HARD_NEGATIVE_AND_FROZEN_NORMAL_SOURCE",
            "current_source_count": int(np.sum(~frozen_domain_mask)),
            "frozen_source_count": int(np.sum(frozen_domain_mask)),
            "frozen_domain_mask_content_hash": hash_bytes(
                np.ascontiguousarray(frozen_domain_mask).tobytes()
            ),
            "frozen_student_report_hash": frozen_report["report_hash"],
            "frozen_student_report_file_hash": hash_bytes(frozen_path.read_bytes()),
            "frozen_corpus_archive_hash": frozen_report["corpus_archive_hash"],
        }
    corpus["normal_observation"] = normal_observation
    corpus["normal_parent_action"] = normal_parent_action
    train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    failure_sample_weight = np.ones(
        corpus["failure_observation"][train].shape[:-1], dtype=np.float32
    )
    if failure_prefix_steps:
        failure_sample_weight[:, :failure_prefix_steps] = failure_prefix_weight
    gate, gate_training = fit_corrective_confidence_gate(
        model=model,
        failure_observation=corpus["failure_observation"][train],
        normal_observation=corpus["normal_observation"][train],
        failure_sample_weight=failure_sample_weight,
    )
    model.update(gate)
    uncalibrated_failure = predict_corrective_confidence_numpy(
        model, corpus["failure_observation"][train]
    )
    uncalibrated_normal = predict_corrective_confidence_numpy(
        model, corpus["normal_observation"][train]
    )
    model.update(
        calibrate_corrective_confidence_gate(
            model,
            threshold=confidence_threshold,
            logit_temperature=confidence_logit_temperature,
        )
    )
    calibrated_failure = predict_corrective_confidence_numpy(
        model, corpus["failure_observation"][train]
    )
    calibrated_normal = predict_corrective_confidence_numpy(
        model, corpus["normal_observation"][train]
    )
    model.update(attach_corrective_temporal_lease(model, temporal))
    source_output_gain = np.asarray(
        model.get("output_gain", np.ones((1,), dtype=np.float32)), dtype=np.float32
    )
    model["output_gain"] = np.asarray(source_output_gain * authority_scale, dtype=np.float32)
    mining.update(
        {
            "selection_score_model_report_hash": base_report["report_hash"],
            "selection_score_model_report_file_hash": hash_bytes(base_path.read_bytes()),
            "full_normal_observation_content_hash": hash_bytes(
                np.ascontiguousarray(full_normal_trace["normal_observation"]).tobytes()
            ),
            "full_normal_parent_action_content_hash": hash_bytes(
                np.ascontiguousarray(full_normal_trace["normal_parent_action"]).tobytes()
            ),
            "selected_time_index_content_hash": hash_bytes(
                np.ascontiguousarray(selected_index).tobytes()
            ),
            "selected_time_indices": selected_index.tolist(),
        }
    )
    gate_training.update(
        {
            "calibration": "SMOOTH_CONSERVATIVE_INTERVENTION_DEADBAND",
            "calibration_threshold": float(confidence_threshold),
            "calibration_logit_temperature": float(confidence_logit_temperature),
            "failure_prefix_weighting_semantics": (
                "EARLY_FAILURE_RECALL_WITHIN_CLASS_UNIT_MEAN_BALANCED"
            ),
            "failure_prefix_steps": int(failure_prefix_steps),
            "failure_prefix_weight": float(failure_prefix_weight),
            "uncalibrated_failure_mean_confidence": float(np.mean(uncalibrated_failure)),
            "uncalibrated_normal_mean_confidence": float(np.mean(uncalibrated_normal)),
            "failure_mean_confidence": float(np.mean(calibrated_failure)),
            "failure_confidence_rms": float(np.sqrt(np.mean(np.square(calibrated_failure)))),
            "normal_mean_confidence": float(np.mean(calibrated_normal)),
            "normal_confidence_rms": float(np.sqrt(np.mean(np.square(calibrated_normal)))),
            "holdout_failure_mean_confidence": float(
                np.mean(
                    predict_corrective_confidence_numpy(
                        model, corpus["failure_observation"][holdout]
                    )
                )
            ),
            "holdout_normal_mean_confidence": float(
                np.mean(
                    predict_corrective_confidence_numpy(
                        model, corpus["normal_observation"][holdout]
                    )
                )
            ),
        }
    )

    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    holdout_count = int(np.sum(holdout))
    if holdout_count % active.required_gpu_count:
        raise ValueError("corrective hard-negative holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective hard-negative holdout reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 31), active.required_gpu_count
    )
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    training = dict(base_report["training"])
    training.update(
        {
            "hard_negative_mining": mining,
            "confidence_gate": gate_training,
            "temporal_intervention_lease": asdict(temporal),
            "temporal_intervention_lease_config_hash": temporal.config_hash,
            "temporal_intervention_semantics": "EVIDENCE_ACCUMULATION_FINITE_LEASE_COOLDOWN",
            "temporal_authority_scale": float(authority_scale),
            "source_output_gain": [float(value) for value in source_output_gain],
            "scaled_output_gain": [float(value) for value in model["output_gain"]],
            "base_student_report_hash": base_report["report_hash"],
            "base_student_report_file_hash": hash_bytes(base_path.read_bytes()),
            "base_student_development_retained": base_report["student_development_retained"],
        }
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def run_opentrack_recovery_corrective_student_dagger(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    source_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    channel_count: int = 16,
    active_gain: float = 0.16,
) -> dict[str, Any]:
    """Run one normal-route DAgger round and re-examine the learned student."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective DAgger output must be new and external")
    source_path = source_student_report_path.expanduser().resolve()
    source_report = validate_recovery_corrective_student_evidence(source_path)
    active = RecoveryCorrectiveStudentConfig(**source_report["config"])
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective DAgger requires exactly four GPUs")
    with np.load(
        source_path.parent / str(source_report["corpus_archive"]), allow_pickle=False
    ) as a:
        source_corpus = {name: np.array(a[name], copy=True) for name in a.files}
    with np.load(source_path.parent / str(source_report["model_archive"]), allow_pickle=False) as a:
        source_model = {name: np.array(a[name], copy=True) for name in a.files}
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(source_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("corrective DAgger lineage differs from source")
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    candidate_trace, normal_wrapped = _collect_candidate_normal_trace(
        context=context,
        config=active,
        reset_keys=normal_reset_keys,
        model=source_model,
    )
    mixed_observation, mixed_action, candidate_sample_mask = mix_corrective_normal_dagger_replay(
        parent_observation=source_corpus["normal_observation"],
        parent_action=source_corpus["normal_parent_action"],
        candidate_observation=candidate_trace["normal_observation"],
        candidate_parent_action=candidate_trace["normal_parent_action"],
    )
    corpus = {
        **source_corpus,
        "normal_observation": mixed_observation,
        "normal_parent_action": mixed_action,
    }
    model, training = _train_student(corpus=corpus, config=active)
    train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
    failure_prediction = predict_corrective_student_numpy(
        model,
        corpus["failure_observation"][train, 0],
        maximum_increment=active.maximum_action_increment,
    )
    failure_trace_prediction = predict_corrective_student_numpy(
        model,
        corpus["failure_observation"][train],
        maximum_increment=active.maximum_action_increment,
    )
    normal_trace_prediction = predict_corrective_student_numpy(
        model,
        corpus["normal_observation"][train],
        maximum_increment=active.maximum_action_increment,
    )
    output_gain, channel_selection = derive_corrective_channel_gain(
        action_effect_jacobian=context.teacher_corpus["action_effect_jacobian"][train],
        failure_prediction=failure_prediction,
        failure_trace_prediction=failure_trace_prediction,
        normal_trace_prediction=normal_trace_prediction,
        channel_count=channel_count,
        active_gain=active_gain,
        allow_fewer_beneficial=True,
    )
    model["output_gain"] = output_gain
    prior_training = source_report.get("training")
    prior_dagger_round = (
        int(prior_training.get("dagger_round", 0)) if isinstance(prior_training, dict) else 0
    )
    training.update(
        {
            "dagger_round": prior_dagger_round + 1,
            "dagger_rollout_controller": "SOURCE_STUDENT_PLUS_FROZEN_PARENT",
            "normal_replay_parent_sample_count_per_route": int(np.sum(~candidate_sample_mask)),
            "normal_replay_candidate_sample_count_per_route": int(np.sum(candidate_sample_mask)),
            "dagger_source_student_report_hash": source_report["report_hash"],
            "dagger_source_student_report_file_hash": hash_bytes(source_path.read_bytes()),
            "dagger_source_student_development_retained": bool(
                source_report["student_development_retained"]
            ),
            "channel_selection": channel_selection,
            "output_gain": [float(value) for value in output_gain],
        }
    )

    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    holdout = np.asarray(corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    if holdout_count % active.required_gpu_count:
        raise ValueError("corrective DAgger holdout is not evenly shardable")
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective DAgger holdout reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(jax.random.PRNGKey(active.random_seed + 31), 4)
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def run_opentrack_recovery_corrective_student_confidence_gate(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    base_student_report_path: Path,
    negative_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    confidence_threshold: float = 0.84,
    confidence_logit_temperature: float = 8.0,
) -> dict[str, Any]:
    """Combine a plastic base student with a learned, OOD-closed silence gate."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective confidence-gate output must be new and external")
    base_path = base_student_report_path.expanduser().resolve()
    negative_path = negative_student_report_path.expanduser().resolve()
    base_report = validate_recovery_corrective_student_evidence(base_path)
    negative_report = validate_recovery_corrective_student_evidence(negative_path)
    active = RecoveryCorrectiveStudentConfig(**base_report["config"])
    if negative_report.get("config_hash") != active.config_hash:
        raise ValueError("corrective confidence-gate configs differ")
    negative_training = negative_report.get("training")
    if (
        not isinstance(negative_training, dict)
        or negative_training.get("dagger_source_student_report_hash") != base_report["report_hash"]
    ):
        raise ValueError("corrective confidence-gate negative replay is not base on-policy")
    lineage_names = (
        "teacher_report_hash",
        "teacher_report_file_hash",
        "teacher_corpus_hash",
        "failure_state_manifest_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    if any(base_report.get(name) != negative_report.get(name) for name in lineage_names):
        raise ValueError("corrective confidence-gate evidence lineages differ")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective confidence gate requires exactly four GPUs")
    with np.load(base_path.parent / str(base_report["corpus_archive"]), allow_pickle=False) as a:
        corpus = {name: np.array(a[name], copy=True) for name in a.files}
    with np.load(base_path.parent / str(base_report["model_archive"]), allow_pickle=False) as a:
        model = {name: np.array(a[name], copy=True) for name in a.files}
    with np.load(
        negative_path.parent / str(negative_report["corpus_archive"]), allow_pickle=False
    ) as a:
        negative_corpus = {name: np.array(a[name], copy=True) for name in a.files}
    if not np.array_equal(corpus["train_source_mask"], negative_corpus["train_source_mask"]):
        raise ValueError("corrective confidence-gate source split differs")
    train = np.asarray(corpus["train_source_mask"], dtype=np.bool_)
    gate, gate_training = fit_corrective_confidence_gate(
        model=model,
        failure_observation=corpus["failure_observation"][train],
        normal_observation=negative_corpus["normal_observation"][train],
    )
    model.update(gate)
    uncalibrated_failure_confidence = predict_corrective_confidence_numpy(
        model, corpus["failure_observation"][train]
    )
    uncalibrated_normal_confidence = predict_corrective_confidence_numpy(
        model, negative_corpus["normal_observation"][train]
    )
    model.update(
        calibrate_corrective_confidence_gate(
            model,
            threshold=confidence_threshold,
            logit_temperature=confidence_logit_temperature,
        )
    )
    calibrated_failure_confidence = predict_corrective_confidence_numpy(
        model, corpus["failure_observation"][train]
    )
    calibrated_normal_confidence = predict_corrective_confidence_numpy(
        model, negative_corpus["normal_observation"][train]
    )
    gate_training.update(
        {
            "calibration": "SMOOTH_CONSERVATIVE_INTERVENTION_DEADBAND",
            "calibration_threshold": float(confidence_threshold),
            "calibration_logit_temperature": float(confidence_logit_temperature),
            "uncalibrated_failure_mean_confidence": float(np.mean(uncalibrated_failure_confidence)),
            "uncalibrated_normal_mean_confidence": float(np.mean(uncalibrated_normal_confidence)),
            "failure_mean_confidence": float(np.mean(calibrated_failure_confidence)),
            "failure_confidence_rms": float(
                np.sqrt(np.mean(np.square(calibrated_failure_confidence)))
            ),
            "normal_mean_confidence": float(np.mean(calibrated_normal_confidence)),
            "normal_confidence_rms": float(
                np.sqrt(np.mean(np.square(calibrated_normal_confidence)))
            ),
        }
    )
    corpus["normal_observation"] = negative_corpus["normal_observation"]
    corpus["normal_parent_action"] = negative_corpus["normal_parent_action"]
    holdout = ~train
    gate_training.update(
        {
            "holdout_failure_mean_confidence": float(
                np.mean(
                    predict_corrective_confidence_numpy(
                        model, corpus["failure_observation"][holdout]
                    )
                )
            ),
            "holdout_normal_mean_confidence": float(
                np.mean(
                    predict_corrective_confidence_numpy(
                        model, corpus["normal_observation"][holdout]
                    )
                )
            ),
        }
    )
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(base_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("corrective confidence-gate runtime lineage differs")
    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(active.random_seed + 1), context.teacher_config.state_count
    )
    holdout_count = int(np.sum(holdout))
    per_device = holdout_count // active.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=active.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=active.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective confidence-gate holdout reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((active.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(jax.random.PRNGKey(active.random_seed + 31), 4)
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=active.trace_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=active.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=active,
        ),
        config=active,
        normal_route=True,
    )
    training = dict(base_report["training"])
    training.update(
        {
            "confidence_gate": gate_training,
            "confidence_gate_base_student_report_hash": base_report["report_hash"],
            "confidence_gate_base_student_report_file_hash": hash_bytes(base_path.read_bytes()),
            "confidence_gate_negative_student_report_hash": negative_report["report_hash"],
            "confidence_gate_negative_student_report_file_hash": hash_bytes(
                negative_path.read_bytes()
            ),
        }
    )
    return write_recovery_corrective_student_evidence(
        output_dir=destination,
        config=active,
        corpus=corpus,
        model=model,
        lineage=context.lineage,
        devices=tuple(str(device) for device in devices),
        training=training,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )


def evaluate_opentrack_recovery_corrective_student_frozen_bank(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    corrective_teacher_report_path: Path,
    candidate_student_report_path: Path,
    frozen_benchmark_student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
) -> dict[str, Any]:
    """Re-examine a scale candidate on an older immutable source split."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective frozen exam output must be new and external")
    candidate_path = candidate_student_report_path.expanduser().resolve()
    frozen_path = frozen_benchmark_student_report_path.expanduser().resolve()
    candidate_report = validate_recovery_corrective_student_evidence(candidate_path)
    frozen_report = validate_recovery_corrective_student_evidence(frozen_path)
    candidate_config = RecoveryCorrectiveStudentConfig(**candidate_report["config"])
    frozen_config = RecoveryCorrectiveStudentConfig(**frozen_report["config"])
    physical_config_names = (
        "maximum_action_increment",
        "trace_steps",
        "normal_rollout_steps",
        "minimum_holdout_cost_improvement_fraction",
        "maximum_holdout_directional_regression_fraction",
        "maximum_holdout_directional_regression_absolute",
        "maximum_normal_increment_rms",
        "maximum_normal_cost_regression_fraction",
        "required_gpu_count",
    )
    if any(
        getattr(candidate_config, name) != getattr(frozen_config, name)
        for name in physical_config_names
    ):
        raise ValueError("corrective frozen exam physical contracts differ")
    devices = tuple(jax.devices())
    if len(devices) != frozen_config.required_gpu_count:
        raise RuntimeError("OpenTrack corrective frozen exam requires exactly four GPUs")
    with np.load(
        candidate_path.parent / str(candidate_report["model_archive"]), allow_pickle=False
    ) as archive:
        candidate_model = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(
        frozen_path.parent / str(frozen_report["corpus_archive"]), allow_pickle=False
    ) as archive:
        frozen_corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
    context = _load_context(
        opentrack_root=opentrack_root,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_config_path=teacher_config_path,
        parent_actor_checkpoint_path=parent_actor_checkpoint_path,
        snapshot_manifest_path=snapshot_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        corrective_teacher_report_path=corrective_teacher_report_path,
    )
    if any(frozen_report.get(name) != value for name, value in context.lineage.items()):
        raise ValueError("corrective frozen benchmark runtime lineage differs")
    shared_lineage = (
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    if any(candidate_report.get(name) != frozen_report.get(name) for name in shared_lineage):
        raise ValueError("corrective scale candidate is not comparable to frozen benchmark")
    teacher_reset_rng, _ = jax.random.split(jax.random.PRNGKey(context.teacher_config.random_seed))
    failure_reset_keys = _stratified_subset_failure_reset_keys(
        rng=teacher_reset_rng,
        environment_count=context.teacher_config.state_count,
        control_steps=context.failure_control_steps,
    )
    normal_reset_keys = jax.random.split(
        jax.random.PRNGKey(frozen_config.random_seed + 1), context.teacher_config.state_count
    )
    holdout = np.asarray(frozen_corpus["holdout_source_mask"], dtype=np.bool_)
    holdout_count = int(np.sum(holdout))
    if holdout_count % frozen_config.required_gpu_count:
        raise ValueError("corrective frozen benchmark holdout is not evenly shardable")
    per_device = holdout_count // frozen_config.required_gpu_count
    failure_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.failure_environment),
        episode_length=frozen_config.trace_steps + 1,
        action_repeat=1,
    )
    normal_wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(context.normal_environment),
        episode_length=frozen_config.normal_rollout_steps + 1,
        action_repeat=1,
    )
    failure_initial = jax.pmap(failure_wrapped.reset)(
        failure_reset_keys[holdout].reshape((frozen_config.required_gpu_count, per_device, 2))
    )
    selected = np.asarray(
        failure_initial.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((-1,))
    if not np.array_equal(selected, frozen_corpus["failure_state_index"][holdout]):
        raise RuntimeError("corrective frozen benchmark reset identity differs")
    normal_initial = jax.pmap(normal_wrapped.reset)(
        normal_reset_keys[holdout].reshape((frozen_config.required_gpu_count, per_device, 2))
    )
    exam_rng = jax.random.split(
        jax.random.PRNGKey(frozen_config.random_seed + 31),
        frozen_config.required_gpu_count,
    )
    failure_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=failure_wrapped,
            parent_policy=context.parent_policy,
            model=candidate_model,
            initial_state=failure_initial,
            initial_rng=exam_rng,
            horizon_steps=frozen_config.trace_steps,
            teacher_config=context.teacher_config,
            student_config=frozen_config,
        ),
        config=frozen_config,
        normal_route=False,
    )
    normal_exam = _summarize_exam(
        outputs=_run_separate_paired_exam(
            wrapped_environment=normal_wrapped,
            parent_policy=context.parent_policy,
            model=candidate_model,
            initial_state=normal_initial,
            initial_rng=exam_rng,
            horizon_steps=frozen_config.normal_rollout_steps,
            teacher_config=context.teacher_config,
            student_config=frozen_config,
        ),
        config=frozen_config,
        normal_route=True,
    )
    return write_recovery_corrective_frozen_exam_evidence(
        candidate_report=candidate_report,
        candidate_report_path=candidate_path,
        frozen_report=frozen_report,
        frozen_report_path=frozen_path,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
        output_path=destination / "frozen-benchmark-report.json",
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Distil an S55 corrective teacher into a four-GPU neural adapter"
    )
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--parent-actor-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--failure-state-manifest", required=True, type=Path)
    parser.add_argument("--corrective-teacher-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--dagger-source-student-report", type=Path)
    parser.add_argument("--confidence-base-student-report", type=Path)
    parser.add_argument("--confidence-negative-student-report", type=Path)
    parser.add_argument("--confidence-threshold", default=0.84, type=float)
    parser.add_argument("--confidence-logit-temperature", default=8.0, type=float)
    parser.add_argument("--temporal-source-student-report", type=Path)
    parser.add_argument("--effect-budget-source-student-report", type=Path)
    parser.add_argument("--effect-budget-frozen-normal-student-report", type=Path)
    parser.add_argument("--effect-budget-minimum-gain-fraction", default=0.50, type=float)
    parser.add_argument("--effect-budget-maximum-gain-fraction", type=float)
    parser.add_argument("--channel-veto-source-student-report", type=Path)
    parser.add_argument("--channel-veto-current-normal-student-report", type=Path)
    parser.add_argument("--channel-veto-frozen-normal-student-report", type=Path)
    parser.add_argument("--channel-veto-frozen-failure-state-manifest", type=Path)
    parser.add_argument("--channel-veto-frozen-corrective-teacher-report", type=Path)
    parser.add_argument("--channel-veto-training-steps", default=1_500, type=int)
    parser.add_argument("--channel-veto-batch-size", default=512, type=int)
    parser.add_argument("--channel-veto-calibration-source-student-report", type=Path)
    parser.add_argument("--channel-veto-logit-temperature", default=2.0, type=float)
    parser.add_argument("--channel-veto-failure-recall-logit-margin", default=0.0, type=float)
    parser.add_argument("--channel-veto-aware-temporal-trigger", action="store_true")
    parser.add_argument("--channel-veto-scale-temporal-amplitude", action="store_true")
    parser.add_argument("--channel-veto-dagger-base-student-report", type=Path)
    parser.add_argument("--channel-veto-dagger-candidate-student-report", type=Path)
    parser.add_argument("--channel-veto-dagger-frozen-normal-student-report", type=Path)
    parser.add_argument("--channel-veto-dagger-frozen-failure-state-manifest", type=Path)
    parser.add_argument("--channel-veto-dagger-frozen-corrective-teacher-report", type=Path)
    parser.add_argument("--temporal-open-threshold", default=0.50, type=float)
    parser.add_argument("--temporal-exit-threshold", default=0.05, type=float)
    parser.add_argument("--temporal-required-open-steps", default=3, type=int)
    parser.add_argument("--temporal-maximum-lease-steps", default=20, type=int)
    parser.add_argument("--temporal-cooldown-steps", default=600, type=int)
    parser.add_argument("--temporal-maximum-slew", default=0.50, type=float)
    parser.add_argument("--temporal-authority-scale", default=1.0, type=float)
    parser.add_argument("--frozen-normal-veto-student-report", type=Path)
    parser.add_argument("--historical-veto-minimum-authority", default=0.0, type=float)
    parser.add_argument("--hard-negative-base-student-report", type=Path)
    parser.add_argument("--hard-negative-refinement-rounds", default=1, type=int)
    parser.add_argument("--hard-negative-failure-prefix-steps", default=0, type=int)
    parser.add_argument("--hard-negative-failure-prefix-weight", default=1.0, type=float)
    parser.add_argument("--frozen-normal-replay-student-report", type=Path)
    parser.add_argument("--frozen-candidate-student-report", type=Path)
    parser.add_argument("--frozen-benchmark-student-report", type=Path)
    parser.add_argument("--channel-count", default=16, type=int)
    parser.add_argument("--active-gain", default=0.16, type=float)
    parser.add_argument("--training-steps", default=1_500, type=int)
    parser.add_argument("--seed", default=5_600, type=int)
    args = parser.parse_args()
    if not 0.0 <= args.channel_veto_failure_recall_logit_margin <= 8.0:
        parser.error("--channel-veto-failure-recall-logit-margin must be in [0, 8]")
    if (
        args.channel_veto_failure_recall_logit_margin > 0.0
        and args.channel_veto_calibration_source_student_report is None
        and args.channel_veto_dagger_base_student_report is None
    ):
        parser.error(
            "--channel-veto-failure-recall-logit-margin requires channel-veto calibration or DAgger"
        )
    if (
        args.channel_veto_aware_temporal_trigger
        and args.channel_veto_calibration_source_student_report is None
    ):
        parser.error(
            "--channel-veto-aware-temporal-trigger requires "
            "--channel-veto-calibration-source-student-report"
        )
    if args.channel_veto_scale_temporal_amplitude and not args.channel_veto_aware_temporal_trigger:
        parser.error(
            "--channel-veto-scale-temporal-amplitude requires --channel-veto-aware-temporal-trigger"
        )
    common = {
        "opentrack_root": args.opentrack_root,
        "teacher_checkpoint_path": args.teacher_checkpoint,
        "teacher_config_path": args.teacher_config,
        "parent_actor_checkpoint_path": args.parent_actor_checkpoint,
        "snapshot_manifest_path": args.snapshot_manifest,
        "failure_state_manifest_path": args.failure_state_manifest,
        "corrective_teacher_report_path": args.corrective_teacher_report,
        "output_dir": args.output_dir,
        "source_checkout_path": args.source_checkout,
    }
    if args.frozen_candidate_student_report is not None:
        if args.frozen_benchmark_student_report is None:
            parser.error("--frozen-benchmark-student-report is required with frozen evaluation")
        result = evaluate_opentrack_recovery_corrective_student_frozen_bank(
            **common,
            candidate_student_report_path=args.frozen_candidate_student_report,
            frozen_benchmark_student_report_path=args.frozen_benchmark_student_report,
        )
    elif args.channel_veto_dagger_base_student_report is not None:
        required_channel_veto_dagger_inputs = {
            "--channel-veto-dagger-candidate-student-report": (
                args.channel_veto_dagger_candidate_student_report
            ),
            "--channel-veto-dagger-frozen-normal-student-report": (
                args.channel_veto_dagger_frozen_normal_student_report
            ),
            "--channel-veto-dagger-frozen-failure-state-manifest": (
                args.channel_veto_dagger_frozen_failure_state_manifest
            ),
            "--channel-veto-dagger-frozen-corrective-teacher-report": (
                args.channel_veto_dagger_frozen_corrective_teacher_report
            ),
        }
        missing = [
            name for name, value in required_channel_veto_dagger_inputs.items() if value is None
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required with channel-veto DAgger")
        result = run_opentrack_recovery_corrective_channel_veto_dagger(
            **common,
            base_student_report_path=args.channel_veto_dagger_base_student_report,
            candidate_student_report_path=args.channel_veto_dagger_candidate_student_report,
            frozen_normal_student_report_path=(
                args.channel_veto_dagger_frozen_normal_student_report
            ),
            frozen_failure_state_manifest_path=(
                args.channel_veto_dagger_frozen_failure_state_manifest
            ),
            frozen_corrective_teacher_report_path=(
                args.channel_veto_dagger_frozen_corrective_teacher_report
            ),
            training_steps=args.channel_veto_training_steps,
            batch_size=args.channel_veto_batch_size,
            logit_temperature=args.channel_veto_logit_temperature,
            failure_recall_logit_margin=(args.channel_veto_failure_recall_logit_margin),
        )
    elif args.channel_veto_calibration_source_student_report is not None:
        result = recheck_opentrack_recovery_corrective_channel_veto_calibration(
            **common,
            source_student_report_path=args.channel_veto_calibration_source_student_report,
            logit_temperature=args.channel_veto_logit_temperature,
            failure_recall_logit_margin=(args.channel_veto_failure_recall_logit_margin),
            veto_aware_temporal_trigger=args.channel_veto_aware_temporal_trigger,
            scale_temporal_amplitude_by_consensus=(args.channel_veto_scale_temporal_amplitude),
        )
    elif args.channel_veto_source_student_report is not None:
        required_channel_veto_inputs = {
            "--channel-veto-current-normal-student-report": (
                args.channel_veto_current_normal_student_report
            ),
            "--channel-veto-frozen-normal-student-report": (
                args.channel_veto_frozen_normal_student_report
            ),
            "--channel-veto-frozen-failure-state-manifest": (
                args.channel_veto_frozen_failure_state_manifest
            ),
            "--channel-veto-frozen-corrective-teacher-report": (
                args.channel_veto_frozen_corrective_teacher_report
            ),
        }
        missing = [name for name, value in required_channel_veto_inputs.items() if value is None]
        if missing:
            parser.error(f"{', '.join(missing)} required with channel-veto training")
        result = run_opentrack_recovery_corrective_channel_veto(
            **common,
            source_student_report_path=args.channel_veto_source_student_report,
            current_normal_student_report_path=args.channel_veto_current_normal_student_report,
            frozen_normal_student_report_path=args.channel_veto_frozen_normal_student_report,
            frozen_failure_state_manifest_path=args.channel_veto_frozen_failure_state_manifest,
            frozen_corrective_teacher_report_path=(
                args.channel_veto_frozen_corrective_teacher_report
            ),
            training_steps=args.channel_veto_training_steps,
            batch_size=args.channel_veto_batch_size,
        )
    elif args.effect_budget_source_student_report is not None:
        if args.effect_budget_frozen_normal_student_report is None:
            parser.error(
                "--effect-budget-frozen-normal-student-report is required with effect budgeting"
            )
        result = recheck_opentrack_recovery_corrective_student_effect_budget(
            **common,
            source_student_report_path=args.effect_budget_source_student_report,
            frozen_normal_student_report_path=args.effect_budget_frozen_normal_student_report,
            minimum_gain_fraction=args.effect_budget_minimum_gain_fraction,
            maximum_gain_fraction=args.effect_budget_maximum_gain_fraction,
        )
    elif args.hard_negative_base_student_report is not None:
        result = run_opentrack_recovery_corrective_temporal_hard_negative_gate(
            **common,
            base_student_report_path=args.hard_negative_base_student_report,
            confidence_threshold=args.confidence_threshold,
            confidence_logit_temperature=args.confidence_logit_temperature,
            hard_negative_refinement_rounds=args.hard_negative_refinement_rounds,
            failure_prefix_steps=args.hard_negative_failure_prefix_steps,
            failure_prefix_weight=args.hard_negative_failure_prefix_weight,
            authority_scale=args.temporal_authority_scale,
            frozen_normal_replay_student_report_path=args.frozen_normal_replay_student_report,
            temporal_config=CorrectiveTemporalLeaseConfig(
                open_threshold=args.temporal_open_threshold,
                exit_threshold=args.temporal_exit_threshold,
                required_open_steps=args.temporal_required_open_steps,
                maximum_lease_steps=args.temporal_maximum_lease_steps,
                cooldown_steps=args.temporal_cooldown_steps,
                maximum_slew=args.temporal_maximum_slew,
            ),
        )
    elif args.temporal_source_student_report is not None:
        result = recheck_opentrack_recovery_corrective_student_temporal_lease(
            **common,
            source_student_report_path=args.temporal_source_student_report,
            authority_scale=args.temporal_authority_scale,
            frozen_normal_veto_student_report_path=args.frozen_normal_veto_student_report,
            historical_veto_minimum_authority=args.historical_veto_minimum_authority,
            temporal_config=CorrectiveTemporalLeaseConfig(
                open_threshold=args.temporal_open_threshold,
                exit_threshold=args.temporal_exit_threshold,
                required_open_steps=args.temporal_required_open_steps,
                maximum_lease_steps=args.temporal_maximum_lease_steps,
                cooldown_steps=args.temporal_cooldown_steps,
                maximum_slew=args.temporal_maximum_slew,
            ),
        )
    elif args.confidence_base_student_report is not None:
        if args.confidence_negative_student_report is None:
            parser.error("--confidence-negative-student-report is required with confidence gating")
        result = run_opentrack_recovery_corrective_student_confidence_gate(
            **common,
            base_student_report_path=args.confidence_base_student_report,
            negative_student_report_path=args.confidence_negative_student_report,
            confidence_threshold=args.confidence_threshold,
            confidence_logit_temperature=args.confidence_logit_temperature,
        )
    elif args.dagger_source_student_report is not None:
        result = run_opentrack_recovery_corrective_student_dagger(
            **common,
            source_student_report_path=args.dagger_source_student_report,
            channel_count=args.channel_count,
            active_gain=args.active_gain,
        )
    else:
        result = run_opentrack_recovery_corrective_student(
            **common,
            config=RecoveryCorrectiveStudentConfig(
                training_steps=args.training_steps,
                random_seed=args.seed,
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "evaluate_opentrack_recovery_corrective_student_frozen_bank",
    "recheck_opentrack_recovery_corrective_student_gain",
    "recheck_opentrack_recovery_corrective_student_effect_budget",
    "recheck_opentrack_recovery_corrective_channel_veto_calibration",
    "run_opentrack_recovery_corrective_student_dagger",
    "run_opentrack_recovery_corrective_student_confidence_gate",
    "run_opentrack_recovery_corrective_channel_veto",
    "run_opentrack_recovery_corrective_channel_veto_dagger",
    "run_opentrack_recovery_corrective_temporal_hard_negative_gate",
    "recheck_opentrack_recovery_corrective_student_temporal_lease",
    "run_opentrack_recovery_corrective_student",
]
