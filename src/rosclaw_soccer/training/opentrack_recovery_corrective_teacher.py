"""Four-GPU short-horizon corrective teacher for exact OpenTrack failures."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_wrappers

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_teacher_ppo import (
    OpenTrackRecoveryMJXTeacherResidualEnv,
    _tree_hash,
)
from rosclaw_soccer.training.recovery_corrective_teacher import (
    RecoveryCorrectiveTeacherConfig,
    write_recovery_corrective_teacher_evidence,
)
from rosclaw_soccer.training.recovery_mjx import (
    RecoveryMJXTeacherResidualPPOConfig,
    compiled_mujoco_model_contract,
    validate_recovery_mjx_failure_state_manifest,
    validate_recovery_mjx_teacher_residual_report,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus

_JOINT_COUNT = 29
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


def _stratified_subset_failure_reset_keys(
    *,
    rng: jax.Array,
    environment_count: int,
    control_steps: np.ndarray[Any, Any],
) -> jax.Array:
    """Select unique exact-bank resets while balancing temporal failure windows."""

    failure_state_count = int(control_steps.shape[0])
    if not 1 <= environment_count <= failure_state_count:
        raise ValueError("corrective teacher failure-state subset size is invalid")
    pool_rng, permutation_rng = jax.random.split(rng)
    pool_size = max(4_096, environment_count * 64)
    key_pool = jax.random.split(pool_rng, pool_size)

    def selected_failure_index(reset_key: jax.Array) -> jax.Array:
        _, injection_rng = jax.random.split(reset_key)
        failure_index_rng = jax.random.split(injection_rng, 7)[5]
        return jax.random.randint(failure_index_rng, (), 0, failure_state_count)

    pool_indices = np.asarray(jax.jit(jax.vmap(selected_failure_index))(key_pool), dtype=np.int32)
    window_order = sorted({int(value) for value in control_steps.tolist()})
    selected_positions: list[int] = []
    selected_states: set[int] = set()
    for window_index in range(environment_count):
        target_step = window_order[window_index % len(window_order)]
        for position, failure_index in enumerate(pool_indices.tolist()):
            if (
                position not in selected_positions
                and failure_index not in selected_states
                and int(control_steps[failure_index]) == target_step
            ):
                selected_positions.append(position)
                selected_states.add(failure_index)
                break
    if len(selected_positions) < environment_count:
        for position, failure_index in enumerate(pool_indices.tolist()):
            if failure_index not in selected_states:
                selected_positions.append(position)
                selected_states.add(failure_index)
            if len(selected_positions) == environment_count:
                break
    if len(selected_positions) != environment_count:
        raise RuntimeError("corrective teacher stratified reset key pool is incomplete")
    selected_keys = key_pool[jnp.asarray(selected_positions, dtype=jnp.int32)]
    return jax.random.permutation(permutation_rng, selected_keys, axis=0, independent=False)


def _pseudo_huber(value: jax.Array, *, delta: float = 0.25) -> jax.Array:
    scaled = value / delta
    return delta**2 * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)


def _repeat_state_candidates(state: Any, candidate_count: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, candidate_count, axis=0),
        state,
    )


def _make_sharded_plan_rollout(
    *,
    wrapped_environment: Any,
    parent_policy: Any,
    config: RecoveryCorrectiveTeacherConfig,
) -> Any:
    """Build a pmap rollout; leading axis is one real accelerator per shard."""

    def rollout_device(
        initial_state: Any,
        plans: jax.Array,
        rollout_rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        state_count, candidate_count = plans.shape[:2]
        flat_plans = plans.reshape(
            (state_count * candidate_count, config.action_chunk_count, _JOINT_COUNT)
        )
        state = _repeat_state_candidates(initial_state, candidate_count)
        total_effect = jnp.zeros((state_count * candidate_count, 4), dtype=jnp.float32)
        total_action_cost = jnp.zeros((state_count * candidate_count,), dtype=jnp.float32)
        total_slew_cost = jnp.zeros_like(total_action_cost)
        finite_rollout = jnp.ones((state_count * candidate_count,), dtype=jnp.bool_)
        previous_delta = jnp.zeros((state_count * candidate_count, _JOINT_COUNT), jnp.float32)

        def step(
            carry: tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            step_index: jax.Array,
        ) -> tuple[
            tuple[
                Any,
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
                current_state,
                rng,
                effect,
                action_cost,
                slew_cost,
                finite_so_far,
                prior_delta,
            ) = carry
            rng, action_rng = jax.random.split(rng)
            baseline_action, _ = parent_policy(current_state.obs, action_rng)
            chunk_index = step_index // config.action_chunk_steps
            action_delta = flat_plans[:, chunk_index]
            action = jnp.clip(baseline_action + action_delta, -1.0, 1.0)
            next_state = wrapped_environment.step(current_state, action)
            metrics = next_state.metrics
            finite = jnp.all(jnp.isfinite(next_state.pipeline_state.data.qpos), axis=-1) & jnp.all(
                jnp.isfinite(next_state.pipeline_state.data.qvel), axis=-1
            )
            step_effect = jnp.stack(
                (
                    _pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
                    _pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
                    _pseudo_huber(metrics["pelvis_yaw_speed"] / 1.5),
                    1.0 - metrics["stable"],
                ),
                axis=-1,
            )
            return (
                next_state,
                rng,
                effect + jnp.nan_to_num(step_effect, nan=100.0, posinf=100.0),
                action_cost + jnp.mean(jnp.square(action_delta), axis=-1),
                slew_cost + jnp.mean(jnp.square(action_delta - prior_delta), axis=-1),
                finite_so_far & finite,
                action_delta,
            ), None

        final, _ = jax.lax.scan(
            step,
            (
                state,
                rollout_rng,
                total_effect,
                total_action_cost,
                total_slew_cost,
                finite_rollout,
                previous_delta,
            ),
            jnp.arange(config.horizon_steps, dtype=jnp.int32),
        )
        effect = final[2] / config.horizon_steps
        action_cost = final[3] / config.horizon_steps
        slew_cost = final[4] / config.horizon_steps
        finite = final[5]
        cost = (
            config.backward_cost_weight * effect[:, 0]
            + config.lateral_cost_weight * effect[:, 1]
            + config.yaw_cost_weight * effect[:, 2]
            + config.stability_deficit_weight * effect[:, 3]
            + config.action_magnitude_cost_weight * action_cost
            + config.action_slew_cost_weight * slew_cost
            + (~finite).astype(jnp.float32) * 100.0
        )
        return (
            cost.reshape((state_count, candidate_count)),
            effect.reshape((state_count, candidate_count, 4)),
            finite.reshape((state_count, candidate_count)),
        )

    return jax.pmap(rollout_device, axis_name="recovery_corrective_teacher_devices")


def _search_corrective_plans(
    *,
    rollout: Any,
    initial_state: Any,
    rollout_keys: jax.Array,
    config: RecoveryCorrectiveTeacherConfig,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    device_count = config.required_gpu_count
    states_per_device = config.state_count // device_count
    plan_shape = (
        device_count,
        states_per_device,
        config.candidate_count,
        config.action_chunk_count,
        _JOINT_COUNT,
    )
    mean = np.zeros(plan_shape[:2] + plan_shape[3:], dtype=np.float32)
    std = np.full_like(mean, config.initial_action_std)
    random = np.random.default_rng(config.random_seed)
    best_cost = np.full(plan_shape[:2], np.inf, dtype=np.float64)
    best_plan = np.zeros_like(mean)
    best_effect = np.zeros(plan_shape[:2] + (4,), dtype=np.float64)
    best_finite = np.zeros(plan_shape[:2], dtype=np.bool_)
    baseline_cost = np.zeros(plan_shape[:2], dtype=np.float64)
    baseline_effect = np.zeros(plan_shape[:2] + (4,), dtype=np.float64)

    for iteration in range(config.cem_iterations):
        noise = random.standard_normal(plan_shape).astype(np.float32)
        plans = np.clip(
            mean[:, :, None, :, :] + std[:, :, None, :, :] * noise,
            -config.maximum_action_increment,
            config.maximum_action_increment,
        )
        plans[:, :, 0] = 0.0
        plans[:, :, 1] = mean
        costs_device, effects_device, finite_device = rollout(
            initial_state,
            jnp.asarray(plans),
            rollout_keys,
        )
        costs = np.asarray(costs_device, dtype=np.float64)
        effects = np.asarray(effects_device, dtype=np.float64)
        finite = np.asarray(finite_device, dtype=np.bool_)
        if iteration == 0:
            baseline_cost = np.array(costs[:, :, 0], copy=True)
            baseline_effect = np.array(effects[:, :, 0], copy=True)
        directional_tolerance = np.maximum(
            np.abs(baseline_effect[..., :3]) * config.maximum_directional_cost_regression_fraction,
            config.maximum_directional_cost_regression_absolute,
        )
        feasible = np.all(
            effects[..., :3] <= baseline_effect[:, :, None, :3] + directional_tolerance[:, :, None],
            axis=-1,
        ) & (
            effects[..., 3]
            <= baseline_effect[:, :, None, 3] + config.maximum_stability_deficit_regression_absolute
        )
        constrained_costs = np.where(feasible, costs, costs + 100.0)
        improved_index = np.argmin(constrained_costs, axis=2)
        selected_cost = np.take_along_axis(costs, improved_index[:, :, None], axis=2)[..., 0]
        selected_plan = np.take_along_axis(
            plans,
            improved_index[:, :, None, None, None],
            axis=2,
        )[:, :, 0]
        selected_effect = np.take_along_axis(
            effects,
            improved_index[:, :, None, None],
            axis=2,
        )[:, :, 0]
        selected_finite = np.take_along_axis(
            finite,
            improved_index[:, :, None],
            axis=2,
        )[:, :, 0]
        improved = selected_cost < best_cost
        best_cost = np.where(improved, selected_cost, best_cost)
        best_plan = np.where(improved[..., None, None], selected_plan, best_plan)
        best_effect = np.where(improved[..., None], selected_effect, best_effect)
        best_finite = np.where(improved, selected_finite, best_finite)
        elite_index = np.argsort(constrained_costs, axis=2)[..., : config.elite_count]
        elites = np.take_along_axis(plans, elite_index[..., None, None], axis=2)
        mean = np.asarray(np.mean(elites, axis=2, dtype=np.float64), dtype=np.float32)
        std = np.maximum(
            np.std(elites, axis=2, dtype=np.float64), config.minimum_action_std
        ).astype(np.float32)

    return (
        best_plan.reshape((config.state_count, config.action_chunk_count, _JOINT_COUNT)),
        best_cost.reshape((config.state_count,)),
        best_effect.reshape((config.state_count, 4)),
        best_finite.reshape((config.state_count,)),
        baseline_cost.reshape((config.state_count,)),
        baseline_effect.reshape((config.state_count, 4)),
    )


def _measure_action_effect_jacobian(
    *,
    rollout: Any,
    initial_state: Any,
    rollout_keys: jax.Array,
    config: RecoveryCorrectiveTeacherConfig,
) -> np.ndarray[Any, Any]:
    device_count = config.required_gpu_count
    states_per_device = config.state_count // device_count
    plans = np.zeros(
        (
            device_count,
            states_per_device,
            config.candidate_count,
            config.action_chunk_count,
            _JOINT_COUNT,
        ),
        dtype=np.float32,
    )
    for joint_index in range(_JOINT_COUNT):
        plans[:, :, 1 + 2 * joint_index, :, joint_index] = config.finite_difference_increment
        plans[:, :, 2 + 2 * joint_index, :, joint_index] = -config.finite_difference_increment
    _, effect_device, _ = rollout(initial_state, jnp.asarray(plans), rollout_keys)
    effects = np.asarray(effect_device, dtype=np.float64)
    plus = effects[:, :, 1:59:2, :]
    minus = effects[:, :, 2:59:2, :]
    jacobian = (plus - minus) / (2.0 * config.finite_difference_increment)
    return np.transpose(jacobian, (0, 1, 3, 2)).reshape((config.state_count, 4, _JOINT_COUNT))


def run_opentrack_recovery_corrective_teacher(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    parent_actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_state_manifest_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: RecoveryCorrectiveTeacherConfig | None = None,
) -> dict[str, Any]:
    """Search corrective labels from exact failure states on four GPUs."""

    active = config or RecoveryCorrectiveTeacherConfig()
    root = opentrack_root.expanduser().resolve()
    teacher_checkpoint = teacher_checkpoint_path.expanduser().resolve()
    teacher_config = teacher_config_path.expanduser().resolve()
    parent_checkpoint = parent_actor_checkpoint_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    failure_manifest_path = failure_state_manifest_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if (
        not root.is_dir()
        or not teacher_checkpoint.is_dir()
        or not teacher_config.is_file()
        or not parent_checkpoint.is_dir()
        or not snapshot_path.is_file()
        or not failure_manifest_path.is_file()
    ):
        raise FileNotFoundError("OpenTrack corrective teacher inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("OpenTrack corrective teacher output must be new and external")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count:
        raise RuntimeError("OpenTrack corrective teacher requires exactly four visible GPUs")

    manifest = validate_recovery_mjx_failure_state_manifest(failure_manifest_path)
    if manifest.get("schema_version") != "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2":
        raise ValueError("OpenTrack corrective teacher requires exact policy-context bank v2")
    parent_report_path = parent_checkpoint.parent.parent / "training-report.json"
    parent_report = validate_recovery_mjx_teacher_residual_report(parent_report_path)
    parent_hash, _ = _tree_hash(parent_checkpoint)
    parent_training_tree_hash, _ = _tree_hash(parent_checkpoint.parent)
    teacher_hash, _ = _tree_hash(teacher_checkpoint)
    snapshot_hash = hash_bytes(snapshot_path.read_bytes())
    if (
        manifest.get("source_actor_checkpoint_hash") != parent_hash
        or parent_report.get("candidate_checkpoint_hash") != parent_training_tree_hash
        or manifest.get("teacher_checkpoint_hash") != teacher_hash
        or manifest.get("snapshot_manifest_hash") != snapshot_hash
        or parent_report.get("snapshot_manifest_hash") != snapshot_hash
        or manifest.get("source_route_manifest_hash") != parent_report.get("route_manifest_hash")
        or manifest.get("source_route_group_hash") != parent_report.get("route_group_hash")
    ):
        raise ValueError("OpenTrack corrective teacher lineage differs")
    actor_config_payload = parent_report.get("config")
    if not isinstance(actor_config_payload, dict):
        raise ValueError("OpenTrack corrective teacher parent config is absent")
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
        control_steps = np.array(archive["control_step"], copy=True, dtype=np.int32)
    motion_dataset_id = str(parent_report["motion_dataset_id"])
    motion_id = str(parent_report["motion_id"])
    entry_frame = int(parent_report["entry_frame"])
    time_dilation = int(parent_report["time_dilation"])
    motion_path = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1" / f"{motion_id}.npz"
    )
    if not motion_path.is_file():
        raise FileNotFoundError("OpenTrack corrective teacher motion archive is absent")

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
    environment = OpenTrackRecoveryMJXTeacherResidualEnv(
        teacher_environment=teacher_environment,
        trajectory_data=trajectory_data,
        teacher_policy=teacher_policy,
        snapshots=snapshots,
        time_dilation=time_dilation,
        terminal_balance_reference_frame=None,
        directional_curriculum=None,
        failure_state_bank=failure_bank,
        config=actor_config,
        parent_residual_policy=None,
        diagnostic_failure_state_reset_fraction=1.0,
    )
    if compiled_mujoco_model_contract(environment.mj_model) != manifest["compiled_model_contract"]:
        raise ValueError("OpenTrack corrective teacher compiled model differs")
    wrapped_environment = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(environment),
        episode_length=active.horizon_steps + 1,
        action_repeat=1,
    )
    reset_rng, rollout_rng = jax.random.split(jax.random.PRNGKey(active.random_seed))
    reset_keys = _stratified_subset_failure_reset_keys(
        rng=reset_rng,
        environment_count=active.state_count,
        control_steps=control_steps,
    ).reshape((active.required_gpu_count, active.state_count // active.required_gpu_count, 2))
    initial_state = jax.pmap(wrapped_environment.reset)(reset_keys)
    rollout_keys = jax.random.split(rollout_rng, active.required_gpu_count)
    rollout = _make_sharded_plan_rollout(
        wrapped_environment=wrapped_environment,
        parent_policy=parent_policy,
        config=active,
    )
    (
        teacher_plan,
        teacher_cost,
        teacher_effect,
        finite_rollout,
        baseline_cost,
        baseline_effect,
    ) = _search_corrective_plans(
        rollout=rollout,
        initial_state=initial_state,
        rollout_keys=rollout_keys,
        config=active,
    )
    action_effect_jacobian = _measure_action_effect_jacobian(
        rollout=rollout,
        initial_state=initial_state,
        rollout_keys=rollout_keys,
        config=active,
    )

    @jax.pmap
    def initial_parent_action(state: Any, action_rng: jax.Array) -> jax.Array:
        return parent_policy(state.obs, action_rng)[0]

    baseline_action = np.asarray(
        initial_parent_action(initial_state, rollout_keys), dtype=np.float32
    ).reshape((active.state_count, _JOINT_COUNT))
    actor_observation = initial_state.obs["state"]
    actor_observation_array = np.asarray(actor_observation, dtype=np.float32).reshape(
        (active.state_count, -1)
    )
    failure_state_indices = np.asarray(
        initial_state.info["selected_failure_state_index"], dtype=np.int32
    ).reshape((active.state_count,))
    corrective_increment = (
        np.clip(baseline_action + teacher_plan[:, 0], -1.0, 1.0) - baseline_action
    )
    teacher_action = np.clip(baseline_action + corrective_increment, -1.0, 1.0)
    improvement = (baseline_cost - teacher_cost) / np.maximum(np.abs(baseline_cost), 1.0e-12)
    directional_tolerance = np.maximum(
        np.abs(baseline_effect[:, :3]) * active.maximum_directional_cost_regression_fraction,
        active.maximum_directional_cost_regression_absolute,
    )
    retention_passed = np.all(
        teacher_effect[:, :3] <= baseline_effect[:, :3] + directional_tolerance,
        axis=1,
    ) & (
        teacher_effect[:, 3]
        <= baseline_effect[:, 3] + active.maximum_stability_deficit_regression_absolute
    )
    accepted = (
        finite_rollout
        & retention_passed
        & (improvement >= active.minimum_cost_improvement_fraction)
    )
    arrays = {
        "actor_observation": actor_observation_array,
        "baseline_action": baseline_action,
        "corrective_action_increment": corrective_increment,
        "teacher_action": teacher_action,
        "teacher_plan": teacher_plan,
        "baseline_cost": baseline_cost,
        "teacher_cost": teacher_cost,
        "cost_improvement_fraction": improvement,
        "teacher_accepted": accepted,
        "finite_rollout": finite_rollout,
        "failure_state_index": failure_state_indices,
        "control_step": control_steps[failure_state_indices],
        "baseline_effect_metrics": baseline_effect,
        "teacher_effect_metrics": teacher_effect,
        "action_effect_jacobian": action_effect_jacobian,
    }
    lineage = {
        "failure_state_manifest_hash": manifest["report_hash"],
        "failure_state_manifest_file_hash": hash_bytes(failure_manifest_path.read_bytes()),
        "failure_state_archive_hash": manifest["state_archive_hash"],
        "parent_training_report_hash": parent_report["report_hash"],
        "parent_checkpoint_hash": parent_hash,
        "teacher_checkpoint_hash": teacher_hash,
        "snapshot_manifest_hash": snapshot_hash,
        "route_manifest_hash": parent_report["route_manifest_hash"],
        "route_group_hash": parent_report["route_group_hash"],
        "motion_archive_hash": hash_bytes(motion_path.read_bytes()),
    }
    return write_recovery_corrective_teacher_evidence(
        output_dir=destination,
        config=active,
        arrays=arrays,
        lineage=lineage,
        devices=tuple(str(device) for device in devices),
        compiled_model_contract=compiled_mujoco_model_contract(environment.mj_model),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run four-GPU MJX corrective teacher search")
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--parent-actor-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--failure-state-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--state-count", default=16, type=int)
    parser.add_argument("--horizon-steps", default=20, type=int)
    parser.add_argument("--action-chunk-steps", default=5, type=int)
    parser.add_argument("--candidate-count", default=64, type=int)
    parser.add_argument("--cem-iterations", default=3, type=int)
    parser.add_argument("--seed", default=5_500, type=int)
    args = parser.parse_args()
    result = run_opentrack_recovery_corrective_teacher(
        opentrack_root=args.opentrack_root,
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_config_path=args.teacher_config,
        parent_actor_checkpoint_path=args.parent_actor_checkpoint,
        snapshot_manifest_path=args.snapshot_manifest,
        failure_state_manifest_path=args.failure_state_manifest,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        config=RecoveryCorrectiveTeacherConfig(
            state_count=args.state_count,
            horizon_steps=args.horizon_steps,
            action_chunk_steps=args.action_chunk_steps,
            candidate_count=args.candidate_count,
            cem_iterations=args.cem_iterations,
            random_seed=args.seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = ["run_opentrack_recovery_corrective_teacher"]
