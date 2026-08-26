"""Collect exact MJX recovery states at evidence-selected failure windows.

This is a training-host-only, SIM_ONLY integration.  It deterministically
replays a content-bound parent actor through the full recovery route and saves
qpos/qvel at steps selected by a temporal failure-window plan.  The resulting
states are curriculum inputs, never promotion evidence or hardware commands.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
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
    validate_recovery_mjx_failure_window_plan,
    validate_recovery_mjx_teacher_residual_report,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus


@dataclass(frozen=True)
class RecoveryMJXFailureWindowCollectionConfig:
    num_environments: int = 16
    random_seed: int = 5471
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_failure_collection_config.v2"

    def __post_init__(self) -> None:
        if (
            not 4 <= self.num_environments <= 128
            or self.num_environments % 4
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX failure collection config is invalid")


def _write_state_archive(path: Path, arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def collect_opentrack_recovery_mjx_failure_windows(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    actor_checkpoint_path: Path,
    snapshot_manifest_path: Path,
    failure_window_plan_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: RecoveryMJXFailureWindowCollectionConfig | None = None,
) -> dict[str, Any]:
    active = config or RecoveryMJXFailureWindowCollectionConfig()
    root = opentrack_root.expanduser().resolve()
    teacher_checkpoint = teacher_checkpoint_path.expanduser().resolve()
    teacher_config = teacher_config_path.expanduser().resolve()
    actor_checkpoint = actor_checkpoint_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    plan_path = failure_window_plan_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if (
        not root.is_dir()
        or not teacher_checkpoint.is_dir()
        or not teacher_config.is_file()
        or not actor_checkpoint.is_dir()
        or not snapshot_path.is_file()
        or not plan_path.is_file()
    ):
        raise FileNotFoundError("recovery MJX failure collection inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("recovery MJX failure collection output must be new and external")
    if len(jax.devices()) < 4:
        raise RuntimeError("recovery MJX failure collection requires four visible GPUs")

    plan = validate_recovery_mjx_failure_window_plan(plan_path)
    actor_report_path = actor_checkpoint.parent.parent / "training-report.json"
    actor_report = validate_recovery_mjx_teacher_residual_report(actor_report_path)
    actor_hash, actor_files = _tree_hash(actor_checkpoint)
    if (
        actor_hash != plan["source_checkpoint_hash"]
        or actor_report.get("route_manifest_hash") != plan["source_route_manifest_hash"]
        or actor_report.get("route_group_hash") != plan["source_route_group_hash"]
    ):
        raise ValueError("recovery MJX failure collection lineage differs")
    actor_config_payload = actor_report.get("config")
    if not isinstance(actor_config_payload, dict):
        raise ValueError("recovery MJX failure collection actor config is absent")
    actor_config = RecoveryMJXTeacherResidualPPOConfig(**actor_config_payload)
    if actor_config.terminal_balance_reset_fraction != 0.0:
        raise ValueError("recovery MJX failure collection requires a full-route parent actor")

    motion_dataset_id = str(actor_report["motion_dataset_id"])
    motion_id = str(actor_report["motion_id"])
    entry_frame = int(actor_report["entry_frame"])
    time_dilation = int(actor_report["time_dilation"])
    motion_path = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1" / f"{motion_id}.npz"
    )
    if not motion_path.is_file():
        raise FileNotFoundError("recovery MJX failure collection motion archive is absent")

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
    environment_config.episode_length = max(3_000, actor_config.episode_length + 100)
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
    actor_policy = checkpoint.load_policy(
        actor_checkpoint,
        network_factory=_make_recovery_ppo_networks,
        deterministic=True,
    )
    all_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    selected_indices = tuple(int(value) for value in actor_report["snapshot_indices"])
    snapshots = tuple(all_snapshots[index] for index in selected_indices)
    environment = OpenTrackRecoveryMJXTeacherResidualEnv(
        teacher_environment=teacher_environment,
        trajectory_data=trajectory_data,
        teacher_policy=teacher_policy,
        snapshots=snapshots,
        time_dilation=time_dilation,
        terminal_balance_reference_frame=None,
        directional_curriculum=None,
        failure_state_bank=None,
        config=actor_config,
    )
    wrapped_environment = brax_training.wrap(
        environment,
        episode_length=actor_config.episode_length,
        action_repeat=1,
        randomization_fn=None,
    )
    requested_steps = tuple(int(value) for value in plan["requested_collection_steps"])
    maximum_step = max(requested_steps)
    reset_rng, rollout_rng = jax.random.split(jax.random.PRNGKey(active.random_seed))
    reset_keys = jax.random.split(reset_rng, active.num_environments)
    initial_state = jax.jit(wrapped_environment.reset)(reset_keys)

    def rollout_step(carry: tuple[Any, jax.Array], unused: Any) -> tuple[Any, Any]:
        del unused
        state, rng = carry
        rng, action_rng = jax.random.split(rng)
        action, _ = actor_policy(state.obs, action_rng)
        next_state = wrapped_environment.step(state, action)
        data = next_state.pipeline_state.data
        trace = (
            data.qpos,
            data.qvel,
            next_state.info["episode_done"],
            next_state.info["handoff_frozen"],
            next_state.pipeline_state.info["traj_info"].traj_state.subtraj_step_no,
            next_state.pipeline_state.info["traj_info"].traj_state.subtraj_step_no_init,
            next_state.metrics["root_body_backward_speed"],
            next_state.metrics["root_body_lateral_speed"],
            next_state.metrics["pelvis_yaw_speed"],
            next_state.pipeline_state.info["last_motor_targets"],
            next_state.pipeline_state.info["last_action"],
            next_state.info["last_residual"],
            next_state.info["proprioception_history"],
            next_state.info["phase_repeat"],
        )
        return (next_state, rng), trace

    @jax.jit
    def rollout(state: Any, rng: jax.Array) -> Any:
        return jax.lax.scan(rollout_step, (state, rng), None, length=maximum_step)[1]

    (
        qpos_trace,
        qvel_trace,
        done_trace,
        handoff_trace,
        trajectory_step_trace,
        trajectory_initial_step_trace,
        backward_trace,
        lateral_trace,
        yaw_trace,
        motor_target_trace,
        teacher_action_trace,
        residual_trace,
        proprioception_history_trace,
        phase_repeat_trace,
    ) = rollout(initial_state, rollout_rng)
    qpos_trace.block_until_ready()
    traces = tuple(
        np.asarray(value)
        for value in (
            qpos_trace,
            qvel_trace,
            done_trace,
            handoff_trace,
            trajectory_step_trace,
            trajectory_initial_step_trace,
            backward_trace,
            lateral_trace,
            yaw_trace,
            motor_target_trace,
            teacher_action_trace,
            residual_trace,
            proprioception_history_trace,
            phase_repeat_trace,
        )
    )
    qpos_rows: list[np.ndarray[Any, Any]] = []
    qvel_rows: list[np.ndarray[Any, Any]] = []
    row_steps: list[int] = []
    row_environments: list[int] = []
    row_backward: list[float] = []
    row_lateral: list[float] = []
    row_yaw: list[float] = []
    row_handoff: list[bool] = []
    row_trajectory_step: list[int] = []
    row_trajectory_initial_step: list[int] = []
    row_motor_targets: list[np.ndarray[Any, Any]] = []
    row_teacher_actions: list[np.ndarray[Any, Any]] = []
    row_residuals: list[np.ndarray[Any, Any]] = []
    row_proprioception_histories: list[np.ndarray[Any, Any]] = []
    row_phase_repeats: list[int] = []
    for step in requested_steps:
        trace_index = step - 1
        if trace_index < 0:
            raise ValueError("recovery MJX failure collection step zero is unsupported")
        for environment_index in range(active.num_environments):
            qpos = traces[0][trace_index, environment_index]
            qvel = traces[1][trace_index, environment_index]
            if (
                bool(traces[2][trace_index, environment_index])
                or not np.all(np.isfinite(qpos))
                or not np.all(np.isfinite(qvel))
            ):
                continue
            qpos_rows.append(np.asarray(qpos, dtype=np.float32))
            qvel_rows.append(np.asarray(qvel, dtype=np.float32))
            row_steps.append(step)
            row_environments.append(environment_index)
            row_handoff.append(bool(traces[3][trace_index, environment_index]))
            row_trajectory_step.append(int(traces[4][trace_index, environment_index]))
            row_trajectory_initial_step.append(int(traces[5][trace_index, environment_index]))
            row_backward.append(float(traces[6][trace_index, environment_index]))
            row_lateral.append(float(traces[7][trace_index, environment_index]))
            row_yaw.append(float(traces[8][trace_index, environment_index]))
            row_motor_targets.append(
                np.asarray(traces[9][trace_index, environment_index], dtype=np.float32)
            )
            row_teacher_actions.append(
                np.asarray(traces[10][trace_index, environment_index], dtype=np.float32)
            )
            row_residuals.append(
                np.asarray(traces[11][trace_index, environment_index], dtype=np.float32)
            )
            row_proprioception_histories.append(
                np.asarray(traces[12][trace_index, environment_index], dtype=np.float32)
            )
            row_phase_repeats.append(int(traces[13][trace_index, environment_index]))
    if not qpos_rows:
        raise RuntimeError("recovery MJX failure collection produced no finite states")

    destination.mkdir(parents=True)
    archive_path = destination / "failure-window-states.npz"
    archive_arrays = {
        "qpos": np.stack(qpos_rows),
        "qvel": np.stack(qvel_rows),
        "control_step": np.asarray(row_steps, dtype=np.int32),
        "environment_index": np.asarray(row_environments, dtype=np.int32),
        "handoff_frozen": np.asarray(row_handoff, dtype=np.bool_),
        "trajectory_step": np.asarray(row_trajectory_step, dtype=np.int32),
        "trajectory_initial_step": np.asarray(row_trajectory_initial_step, dtype=np.int32),
        "root_body_backward_speed_mps": np.asarray(row_backward, dtype=np.float32),
        "root_body_lateral_speed_mps": np.asarray(row_lateral, dtype=np.float32),
        "pelvis_yaw_speed_rad_s": np.asarray(row_yaw, dtype=np.float32),
        "last_motor_targets": np.stack(row_motor_targets),
        "last_teacher_action": np.stack(row_teacher_actions),
        "last_residual": np.stack(row_residuals),
        "proprioception_history": np.stack(row_proprioception_histories),
        "phase_repeat": np.asarray(row_phase_repeats, dtype=np.int32),
    }
    _write_state_archive(archive_path, archive_arrays)
    teacher_hash, teacher_files = _tree_hash(teacher_checkpoint)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2",
        "config": {
            "activation_ceiling": active.activation_ceiling,
            "hardware_authorized": active.hardware_authorized,
            "num_environments": active.num_environments,
            "random_seed": active.random_seed,
            "schema_version": active.schema_version,
        },
        "source_failure_window_plan_hash": plan["report_hash"],
        "source_failure_window_plan_file_hash": hash_bytes(plan_path.read_bytes()),
        "source_training_report_hash": plan["source_training_report_hash"],
        "source_actor_checkpoint_hash": actor_hash,
        "source_actor_checkpoint_files": actor_files,
        "source_actor_config_hash": actor_report["config_hash"],
        "source_route_manifest_hash": plan["source_route_manifest_hash"],
        "source_route_group_hash": plan["source_route_group_hash"],
        "teacher_checkpoint_hash": teacher_hash,
        "teacher_checkpoint_files": teacher_files,
        "motion_archive_hash": hash_bytes(motion_path.read_bytes()),
        "snapshot_manifest_hash": hash_bytes(snapshot_path.read_bytes()),
        "compiled_model_contract": compiled_mujoco_model_contract(environment.mj_model),
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "deterministic_actor": True,
        "full_route_reset": True,
        "requested_collection_steps": list(requested_steps),
        "collected_state_count": len(qpos_rows),
        "state_archive": archive_path.name,
        "state_archive_hash": hash_bytes(archive_path.read_bytes()),
        "qpos_shape": list(archive_arrays["qpos"].shape),
        "qvel_shape": list(archive_arrays["qvel"].shape),
        "proprioception_history_shape": list(archive_arrays["proprioception_history"].shape),
        "context_features_collected": [
            "qpos",
            "qvel",
            "trajectory_step",
            "trajectory_initial_step",
            "handoff_frozen",
            "last_motor_targets",
            "last_teacher_action",
            "last_residual",
            "proprioception_history",
            "phase_repeat",
        ],
        "curriculum_use_only": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "failure-state-manifest.json", report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Collect bound MJX recovery failure states")
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--actor-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--failure-window-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--num-environments", default=16, type=int)
    parser.add_argument("--seed", default=5471, type=int)
    args = parser.parse_args()
    result = collect_opentrack_recovery_mjx_failure_windows(
        opentrack_root=args.opentrack_root,
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_config_path=args.teacher_config,
        actor_checkpoint_path=args.actor_checkpoint,
        snapshot_manifest_path=args.snapshot_manifest,
        failure_window_plan_path=args.failure_window_plan,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        config=RecoveryMJXFailureWindowCollectionConfig(
            num_environments=args.num_environments,
            random_seed=args.seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "RecoveryMJXFailureWindowCollectionConfig",
    "collect_opentrack_recovery_mjx_failure_windows",
]
