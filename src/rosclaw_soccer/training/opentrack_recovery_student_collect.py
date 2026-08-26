"""Collect privileged recovery teachers as proprio-to-absolute-target data."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _atomic_json,
    _file_hash,
    _run_bridge_trial,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _trial_from_dict,
    _verified_development_report,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryProprioceptionSpec,
    RecoveryTeacherEpisode,
    build_recovery_proprioception,
    write_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatch,
    RecoveryEntryMatcher,
    RecoveryEntrySearchConfig,
    RecoveryPerturbation,
    RecoveryPerturbationConfig,
    build_recovery_perturbation_holdout,
)

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _verified_holdout_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery bridge holdout report must be an object")
    declared_hash = payload.pop("report_hash", None)
    if (
        payload.get("schema_version")
        != "rosclaw_soccer.opentrack_recovery_bridge_holdout.v1"
        or declared_hash != hash_json(payload)
        or payload.get("route_reselection_count") != 0
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery bridge holdout report integrity failed")
    payload["report_hash"] = declared_hash
    return payload


def _teacher_body_hash(env: Any, mujoco: Any) -> str:
    actuator_names = [
        str(mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index))
        for index in range(env.mj_model.nu)
    ]
    return str(
        hash_json(
            {
                "schema_version": "rosclaw_soccer.g1_teacher_body_contract.v1",
                "actuator_names": actuator_names,
                "joint_ranges_rad": env.mj_model.jnt_range[1:30].tolist(),
                "body_mass_kg": env.mj_model.body_mass.tolist(),
                "body_inertia": env.mj_model.body_inertia.tolist(),
            }
        )
    )


def collect_opentrack_recovery_student_corpus(
    *,
    opentrack_root: Path,
    teacher_policy_path: Path,
    teacher_config_path: Path,
    motion_paths: tuple[Path, ...],
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    output_dir: Path,
    training_perturbation_config: RecoveryPerturbationConfig | None = None,
    proprioception_spec: RecoveryProprioceptionSpec | None = None,
) -> dict[str, Any]:
    """Collect successful routes while proving no sealed holdout state leaked."""

    root = opentrack_root.expanduser().resolve()
    policy_path = teacher_policy_path.expanduser().resolve()
    teacher_configuration_path = teacher_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    holdout_path = sealed_holdout_report_path.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    required_files = (
        policy_path,
        teacher_configuration_path,
        snapshot_path,
        development_path,
        holdout_path,
    )
    if not root.is_dir() or any(not path.is_file() for path in required_files):
        raise FileNotFoundError("recovery student collection inputs are incomplete")
    if (
        not _DATASET_ID.fullmatch(motion_dataset_id)
        or target.exists()
        or target == root
        or root in target.parents
    ):
        raise ValueError("recovery student collection destination or dataset is invalid")
    expected_motion_root = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1"
    )
    resolved_motions = tuple(path.expanduser().resolve() for path in motion_paths)
    if not resolved_motions or any(
        path.parent != expected_motion_root for path in resolved_motions
    ):
        raise ValueError("collection motions must belong to the declared dataset")

    development = _verified_development_report(development_path)
    sealed_holdout = _verified_holdout_report(holdout_path)
    if (
        sealed_holdout.get("development_report_hash") != development["report_hash"]
        or development["teacher_policy_hash"] != _file_hash(policy_path)
        or development["teacher_config_hash"] != _file_hash(teacher_configuration_path)
        or development["snapshot_manifest_hash"] != _file_hash(snapshot_path)
    ):
        raise ValueError("recovery student collection evidence bindings differ")
    sealed_hashes = {
        str(item["perturbed_snapshot_hash"])
        for item in sealed_holdout.get("perturbations", [])
    }
    if len(sealed_hashes) != sealed_holdout.get("trial_count"):
        raise ValueError("sealed recovery holdout identities are invalid")

    search_config = RecoveryEntrySearchConfig(**development["search_config"])
    matcher = RecoveryEntryMatcher.from_paths(resolved_motions, config=search_config)
    if matcher.library_hash != development["reference_library_hash"]:
        raise ValueError("recovery student reference library differs from development")
    exam_payload = dict(development["exam_config"])
    exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
    exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
    route_payload = development["post_skill_transfer"]["development_schedule"][
        "selected_trials"
    ]
    routes = tuple(_trial_from_dict(item) for item in route_payload)
    route_by_base = {item.snapshot_hash: item for item in routes}
    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    if set(route_by_base) != {item.snapshot_hash for item in base_snapshots}:
        raise ValueError("recovery student routes do not cover base snapshots")

    perturbation_config = training_perturbation_config or RecoveryPerturbationConfig(
        samples_per_snapshot=2,
        joint_position_half_width_rad=0.030,
        joint_velocity_half_width_rad_s=0.080,
        root_tilt_half_width_rad=0.025,
        root_linear_velocity_half_width_mps=0.050,
        root_angular_velocity_half_width_rad_s=0.080,
        seed_namespace="rosclaw-s52-recovery-student-train-v1",
    )
    if perturbation_config.seed_namespace == "rosclaw-s51-recovery-holdout-v1":
        raise ValueError("training and sealed holdout namespaces must differ")
    perturbed = build_recovery_perturbation_holdout(
        base_snapshots, config=perturbation_config
    )
    if sealed_hashes & {item.snapshot_hash for item, _ in perturbed}:
        raise ValueError("sealed recovery holdout leaked into student training")
    active_proprioception = proprioception_spec or RecoveryProprioceptionSpec()

    training_states: list[
        tuple[RecoverySnapshot, RecoverySnapshot, RecoveryPerturbation | None]
    ] = [(item, item, None) for item in base_snapshots]
    base_by_hash = {item.snapshot_hash: item for item in base_snapshots}
    training_states.extend(
        (base_by_hash[record.base_snapshot_hash], snapshot, record)
        for snapshot, record in perturbed
    )

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module(
        "track_mj.envs.g1_tracking.play.play_g1_env_tracking_general"
    )
    constants = importlib.import_module(
        "track_mj.envs.g1_tracking.g1_tracking_constants"
    )
    ort = importlib.import_module("onnxruntime")
    teacher_payload = json.loads(teacher_configuration_path.read_text(encoding="utf-8"))
    if not isinstance(teacher_payload, dict) or not isinstance(
        teacher_payload.get("env_config"), dict
    ):
        raise ValueError("OpenTrack teacher config has no environment contract")
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    if (
        tuple(item.name for item in session.get_inputs()) != ("obs",)
        or tuple(item.name for item in session.get_outputs())
        != ("continuous_actions",)
    ):
        raise ValueError("OpenTrack recovery teacher IO is incompatible")

    def make_env(match: RecoveryEntryMatch) -> Any:
        environment_config = copy.deepcopy(
            tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
        )
        environment_config.update(teacher_payload["env_config"])
        environment_config.reference_traj_config.name = {
            motion_dataset_id: [match.motion_id]
        }
        environment_config.reference_traj_config.random_start = False
        environment_config.reference_traj_config.fixed_start_frame = match.entry_frame
        environment_class = tmj.registry.get(
            "G1TrackingGeneral", "tracking_play_env_class"
        )
        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            return environment_class(
                config=environment_config,
                play_ref_motion=False,
                use_viewer=False,
                use_renderer=False,
                exp_name="rosclaw-s52-recovery-student-collect",
            )
        finally:
            os.chdir(previous_directory)

    teacher_hash = _file_hash(policy_path)
    collected: list[RecoveryTeacherEpisode] = []
    attempts: list[dict[str, Any]] = []
    default_joint_position: NDArray[np.float64] | None = None
    joint_lower: NDArray[np.float64] | None = None
    joint_upper: NDArray[np.float64] | None = None
    teacher_body_hash: str | None = None
    for episode_index, (base, initial, perturbation) in enumerate(training_states):
        route: RecoveryBridgeTrial = route_by_base[base.snapshot_hash]
        environment = make_env(route.match)
        proprio_frames: list[NDArray[np.float32]] = []
        absolute_targets: list[NDArray[np.float32]] = []
        ready_flags: list[bool] = []
        try:
            identity = np.arange(29)
            if not np.array_equal(environment.action_joint_ids, identity) or not np.array_equal(
                environment.obs_joint_ids, identity
            ):
                raise ValueError("student collection requires canonical 29-joint order")
            current_default = np.asarray(environment._default_qpos, dtype=np.float64)
            current_lower = np.asarray(environment._lowers, dtype=np.float64)
            current_upper = np.asarray(environment._uppers, dtype=np.float64)
            current_body_hash = _teacher_body_hash(environment, mujoco)
            if default_joint_position is None:
                default_joint_position = current_default
                joint_lower = current_lower
                joint_upper = current_upper
                teacher_body_hash = current_body_hash
            else:
                if joint_lower is None or joint_upper is None:
                    raise ValueError("teacher joint bounds were not initialized")
                if (
                    not np.array_equal(default_joint_position, current_default)
                    or not np.array_equal(joint_lower, current_lower)
                    or not np.array_equal(joint_upper, current_upper)
                    or teacher_body_hash != current_body_hash
                ):
                    raise ValueError("teacher body changed during student collection")

            def record_transition(
                env: Any,
                state: Any,
                action: NDArray[np.float64],
                step: int,
                proprio_buffer: list[NDArray[np.float32]] = proprio_frames,
                target_buffer: list[NDArray[np.float32]] = absolute_targets,
                readiness: list[bool] = ready_flags,
            ) -> None:
                del step
                gravity = (
                    env.mj_data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3).T
                    @ np.asarray((0.0, 0.0, -1.0))
                )
                proprio_buffer.append(
                    build_recovery_proprioception(
                        projected_gravity_body=gravity,
                        pelvis_gyro_rad_s=env.get_gyro("pelvis"),
                        joint_position_rad=env.mj_data.qpos[7:],
                        joint_velocity_rad_s=env.mj_data.qvel[6:],
                        last_motor_target_rad=state.info["last_motor_targets"],
                        default_joint_position_rad=env._default_qpos,
                        spec=active_proprioception,
                    )
                )
                reference_qpos, _ = env.th.get_current_traj_data_fast(
                    state.info["traj_info"]
                )
                motor_target = np.asarray(env._default_qpos, dtype=np.float64).copy()
                motor_target[env.action_joint_ids] = (
                    np.asarray(reference_qpos)[7:][env.action_joint_ids]
                    + action * env._config.action_scale
                )
                target_buffer.append(motor_target.astype(np.float32))
                readiness.append(False)

            def mark_ready(
                env: Any,
                step: int,
                handoff: bool,
                readiness: list[bool] = ready_flags,
            ) -> None:
                del env, step
                if not readiness:
                    raise ValueError("teacher callback ordering is invalid")
                readiness[-1] = handoff

            trial, trace = _run_bridge_trial(
                env=environment,
                session=session,
                snapshot=initial,
                snapshot_hash=initial.snapshot_hash,
                match=route.match,
                teacher_policy_hash=teacher_hash,
                time_dilation=route.time_dilation,
                config=exam_config,
                mujoco=mujoco,
                transition_callback=record_transition,
                frame_callback=mark_ready,
            )
        finally:
            environment.close()
        attempts.append(
            {
                "episode_index": episode_index,
                "base_snapshot_hash": base.snapshot_hash,
                "initial_snapshot_hash": initial.snapshot_hash,
                "perturbation_hash": (
                    None if perturbation is None else perturbation.perturbation_hash
                ),
                "fixed_route_trial_hash": route.trial_hash,
                "teacher_trial": trial.to_dict() | {"trial_hash": trial.trial_hash},
                "trace_summary": trace,
            }
        )
        if not trial.succeeded:
            continue
        collected.append(
            RecoveryTeacherEpisode(
                episode_id=f"teacher-{episode_index:03d}",
                base_snapshot_hash=base.snapshot_hash,
                initial_snapshot_hash=initial.snapshot_hash,
                fixed_route_trial_hash=route.trial_hash,
                perturbation_hash=(
                    None if perturbation is None else perturbation.perturbation_hash
                ),
                proprio=np.stack(proprio_frames),
                absolute_motor_targets_rad=np.stack(absolute_targets),
                ready_handoff=np.asarray(ready_flags, dtype=np.bool_),
                time_dilation=route.time_dilation,
                teacher_succeeded=True,
            )
        )

    if (
        default_joint_position is None
        or joint_lower is None
        or joint_upper is None
        or teacher_body_hash is None
        or not collected
    ):
        raise ValueError("recovery student collection produced no valid corpus")
    corpus_manifest = write_recovery_distillation_corpus(
        episodes=collected,
        output_dir=target,
        corpus_name="recovery-student-train-v1",
        proprioception_spec=active_proprioception,
        teacher_policy_hash=teacher_hash,
        body_hash=teacher_body_hash,
        physics_scene_hash=_file_hash(
            Path(constants.task_to_xml("flat_terrain")).expanduser().resolve()
        ),
        development_report_hash=str(development["report_hash"]),
        default_joint_position_rad=default_joint_position,
        joint_lower_rad=joint_lower,
        joint_upper_rad=joint_upper,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_student_collection.v1",
        "development_report_hash": development["report_hash"],
        "sealed_holdout_report_hash": sealed_holdout["report_hash"],
        "sealed_holdout_state_count": len(sealed_hashes),
        "training_holdout_overlap_count": 0,
        "training_perturbation_config": asdict(perturbation_config),
        "training_perturbation_config_hash": perturbation_config.config_hash,
        "proprioception_spec": asdict(active_proprioception),
        "proprioception_spec_hash": active_proprioception.spec_hash,
        "attempted_episode_count": len(training_states),
        "accepted_episode_count": len(collected),
        "rejected_episode_count": len(training_states) - len(collected),
        "attempts": attempts,
        "corpus_manifest": "recovery-student-train-v1.json",
        "corpus_manifest_hash": corpus_manifest["manifest_hash"],
        "sample_count": corpus_manifest["sample_count"],
        "teacher_policy_hash": teacher_hash,
        "teacher_body_hash": teacher_body_hash,
        "teacher_scene_hash": corpus_manifest["physics_scene_hash"],
        "contains_reference_features": False,
        "target_semantics": active_proprioception.output_semantics,
        "teacher_role": "PRIVILEGED_DATA_GENERATOR_ONLY",
        "promotion_eligible": False,
        "claim_boundary": "TRAINING_CORPUS_NOT_STUDENT_PHYSICS_EVIDENCE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target / "collection-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-policy", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--motion-path", required=True, action="append", type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--sealed-holdout-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-snapshot", default=2, type=int)
    args = parser.parse_args()
    perturbation_config = RecoveryPerturbationConfig(
        samples_per_snapshot=args.samples_per_snapshot,
        joint_position_half_width_rad=0.030,
        joint_velocity_half_width_rad_s=0.080,
        root_tilt_half_width_rad=0.025,
        root_linear_velocity_half_width_mps=0.050,
        root_angular_velocity_half_width_rad_s=0.080,
        seed_namespace="rosclaw-s52-recovery-student-train-v1",
    )
    report = collect_opentrack_recovery_student_corpus(
        opentrack_root=args.opentrack_root,
        teacher_policy_path=args.teacher_policy,
        teacher_config_path=args.teacher_config,
        motion_paths=tuple(args.motion_path),
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        output_dir=args.output_dir,
        training_perturbation_config=perturbation_config,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "accepted_episode_count": report["accepted_episode_count"],
                "sample_count": report["sample_count"],
                "training_holdout_overlap_count": report[
                    "training_holdout_overlap_count"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_opentrack_recovery_student_corpus"]
