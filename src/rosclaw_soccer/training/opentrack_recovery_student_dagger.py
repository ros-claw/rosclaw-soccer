"""Collect teacher labels on student-visited recovery states with DAgger."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _atomic_json,
    _file_hash,
    _restore_reference,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _trial_from_dict,
    _verified_development_report,
)
from rosclaw_soccer.training.opentrack_recovery_student_collect import (
    _verified_holdout_report,
)
from rosclaw_soccer.training.opentrack_recovery_student_exam import _verified_json
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryTeacherEpisode,
    build_recovery_proprioception,
    denormalize_absolute_motor_targets,
    load_recovery_distillation_corpus,
    recovery_teacher_episodes_from_corpus,
    write_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryPerturbation,
    RecoveryPerturbationConfig,
    body_gravity_vector,
    build_recovery_perturbation_holdout,
)


def _verified_collection_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery student collection report must be an object")
    declared = payload.pop("report_hash", None)
    if (
        payload.get("schema_version")
        != "rosclaw_soccer.recovery_student_collection.v1"
        or declared != hash_json(payload)
        or payload.get("training_holdout_overlap_count") != 0
        or payload.get("contains_reference_features") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery student collection report integrity failed")
    payload["report_hash"] = declared
    return payload


def collect_recovery_student_dagger_corpus(
    *,
    opentrack_root: Path,
    teacher_policy_path: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    base_corpus_manifest_path: Path,
    base_collection_report_path: Path,
    student_artifact_manifest_path: Path,
    output_dir: Path,
    teacher_action_fraction: float = 0.75,
    dagger_round: int = 1,
) -> dict[str, Any]:
    """Aggregate one fail-closed DAgger round outside the sealed holdout."""

    if not math.isfinite(teacher_action_fraction) or not 0.05 <= teacher_action_fraction <= 0.95:
        raise ValueError("DAgger teacher action fraction must be in [0.05, 0.95]")
    if not 1 <= dagger_round <= 32:
        raise ValueError("DAgger round must be in [1, 32]")
    root = opentrack_root.expanduser().resolve()
    teacher_path = teacher_policy_path.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    holdout_path = sealed_holdout_report_path.expanduser().resolve()
    corpus_path = base_corpus_manifest_path.expanduser().resolve()
    collection_path = base_collection_report_path.expanduser().resolve()
    artifact_path = student_artifact_manifest_path.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    files = (
        teacher_path,
        environment_path,
        snapshot_path,
        development_path,
        holdout_path,
        corpus_path,
        collection_path,
        artifact_path,
    )
    if not root.is_dir() or any(not path.is_file() for path in files):
        raise FileNotFoundError("recovery student DAgger inputs are incomplete")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("recovery student DAgger output must be new and external")

    development = _verified_development_report(development_path)
    sealed = _verified_holdout_report(holdout_path)
    collection = _verified_collection_report(collection_path)
    artifact = _verified_json(
        artifact_path,
        schema_version="rosclaw_soccer.recovery_student_artifact.v1",
        hash_key="manifest_hash",
    )
    corpus = load_recovery_distillation_corpus(corpus_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    student_path = artifact_path.parent / str(artifact.get("onnx"))
    if (
        sealed.get("development_report_hash") != development["report_hash"]
        or collection.get("development_report_hash") != development["report_hash"]
        or collection.get("sealed_holdout_report_hash") != sealed["report_hash"]
        or artifact.get("corpus_manifest_hash") != corpus.manifest_hash
        or artifact.get("contains_reference_features") is not False
        or artifact.get("onnx_hash") != _file_hash(student_path)
        or development.get("teacher_policy_hash") != _file_hash(teacher_path)
        or development.get("teacher_config_hash") != _file_hash(environment_path)
        or development.get("snapshot_manifest_hash") != _file_hash(snapshot_path)
    ):
        raise ValueError("recovery student DAgger evidence bindings differ")

    routes = tuple(
        _trial_from_dict(item)
        for item in development["post_skill_transfer"]["development_schedule"][
            "selected_trials"
        ]
    )
    route_by_base = {item.snapshot_hash: item for item in routes}
    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    perturbation_payload = dict(collection["training_perturbation_config"])
    training_perturbation = RecoveryPerturbationConfig(**perturbation_payload)
    training_perturbed = build_recovery_perturbation_holdout(
        base_snapshots, config=training_perturbation
    )
    sealed_hashes = {
        str(item["perturbed_snapshot_hash"]) for item in sealed["perturbations"]
    }
    if sealed_hashes & {item.snapshot_hash for item, _ in training_perturbed}:
        raise ValueError("sealed holdout leaked into recovery student DAgger")
    base_by_hash = {item.snapshot_hash: item for item in base_snapshots}
    training_states: list[
        tuple[RecoverySnapshot, RecoverySnapshot, RecoveryPerturbation | None]
    ] = [(item, item, None) for item in base_snapshots]
    training_states.extend(
        (base_by_hash[identity.base_snapshot_hash], snapshot, identity)
        for snapshot, identity in training_perturbed
    )
    expected_initial_hashes = {
        str(item["initial_snapshot_hash"]) for item in collection["attempts"]
    }
    if expected_initial_hashes != {item.snapshot_hash for _, item, _ in training_states}:
        raise ValueError("DAgger states differ from the unsealed training distribution")
    corpus_initial_hashes = {str(item["initial_snapshot_hash"]) for item in corpus.rows}
    if not expected_initial_hashes <= corpus_initial_hashes:
        raise ValueError("DAgger base corpus lost the original training distribution")

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module(
        "track_mj.envs.g1_tracking.play.play_g1_env_tracking_general"
    )
    ort = importlib.import_module("onnxruntime")
    teacher_session = ort.InferenceSession(
        str(teacher_path), providers=["CPUExecutionProvider"]
    )
    student_session = ort.InferenceSession(
        str(student_path), providers=["CPUExecutionProvider"]
    )
    if (
        tuple(item.name for item in teacher_session.get_inputs()) != ("obs",)
        or tuple(item.name for item in teacher_session.get_outputs())
        != ("continuous_actions",)
        or tuple(item.name for item in student_session.get_inputs())
        != ("proprio_history",)
        or tuple(item.name for item in student_session.get_outputs())
        != ("normalized_absolute_motor_target",)
    ):
        raise ValueError("recovery student DAgger policy IO differs")
    environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    if not isinstance(environment_payload.get("env_config"), dict):
        raise ValueError("recovery student DAgger environment config is invalid")
    exam_payload = dict(development["exam_config"])
    exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
    exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)

    def make_env(route: RecoveryBridgeTrial) -> Any:
        environment_config = copy.deepcopy(
            tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
        )
        environment_config.update(environment_payload["env_config"])
        environment_config.reference_traj_config.name = {
            motion_dataset_id: [route.match.motion_id]
        }
        environment_config.reference_traj_config.random_start = False
        environment_config.reference_traj_config.fixed_start_frame = (
            route.match.entry_frame
        )
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
                exp_name="rosclaw-s52-recovery-student-dagger",
            )
        finally:
            os.chdir(previous_directory)

    dagger_episodes: list[RecoveryTeacherEpisode] = []
    attempts: list[dict[str, Any]] = []
    for episode_index, (base, initial, identity) in enumerate(training_states):
        route = route_by_base[base.snapshot_hash]
        env = make_env(route)
        frames: list[NDArray[np.float32]] = []
        labels: list[NDArray[np.float32]] = []
        ready_flags: list[bool] = []
        try:
            state = env.reset()
            qpos = np.asarray(initial.qpos, dtype=np.float64).copy()
            qvel = np.asarray(initial.qvel, dtype=np.float64).copy()
            qpos[:2] = np.asarray(env.mj_data.qpos[:2], dtype=np.float64)
            env.mj_data.qpos[:] = qpos
            env.mj_data.qvel[:] = qvel
            mujoco.mj_forward(env.mj_model, env.mj_data)
            trajectory = env.th.get_current_traj_data(state.info["traj_info"])
            env.ref_mj_data.qpos[:] = trajectory.qpos
            env.ref_mj_data.qvel[:] = trajectory.qvel
            mujoco.mj_forward(env.mj_model, env.ref_mj_data)
            state.info["last_motor_targets"] = qpos[7:].copy()
            observation, _ = env.get_obs(trajectory, state.info)
            state = type(state)(state.info, observation)
            history: deque[NDArray[np.float32]] = deque(
                maxlen=corpus.proprioception_spec.history_steps
            )
            ready_pose_streak = 0
            stable_streak = 0
            maximum_stable_streak = 0
            ready_carry: Any | None = None
            ready_trigger_step: int | None = None
            finite_state = True
            peak_angular_speed = 0.0
            maximum_steps = int(round(exam_config.maximum_duration_sec / env.dt))
            executed_steps = 0
            for step in range(maximum_steps):
                carry = state.info["traj_info"]
                gravity = (
                    env.mj_data.site_xmat[env._pelvis_imu_site_id].reshape(3, 3).T
                    @ np.asarray((0.0, 0.0, -1.0))
                )
                frame = build_recovery_proprioception(
                    projected_gravity_body=gravity,
                    pelvis_gyro_rad_s=env.get_gyro("pelvis"),
                    joint_position_rad=env.mj_data.qpos[7:],
                    joint_velocity_rad_s=env.mj_data.qvel[6:],
                    last_motor_target_rad=state.info["last_motor_targets"],
                    default_joint_position_rad=corpus.default_joint_position_rad,
                    spec=corpus.proprioception_spec,
                )
                if not history:
                    history.extend(
                        frame.copy()
                        for _ in range(corpus.proprioception_spec.history_steps)
                    )
                else:
                    history.append(frame)
                student_normalized = student_session.run(
                    ["normalized_absolute_motor_target"],
                    {"proprio_history": np.stack(history)[None, :, :]},
                )[0][0]
                student_target = denormalize_absolute_motor_targets(
                    np.asarray(student_normalized, dtype=np.float32),
                    joint_lower_rad=corpus.joint_lower_rad,
                    joint_upper_rad=corpus.joint_upper_rad,
                )
                teacher_action = teacher_session.run(
                    ["continuous_actions"],
                    {"obs": np.asarray(state.obs["state"], dtype=np.float32)[None, :]},
                )[0][0]
                reference_qpos, _ = env.th.get_current_traj_data_fast(carry)
                teacher_target = np.asarray(env._default_qpos, dtype=np.float64).copy()
                teacher_target[env.action_joint_ids] = (
                    np.asarray(reference_qpos)[7:][env.action_joint_ids]
                    + teacher_action * env._config.action_scale
                )
                teacher_target = np.clip(
                    teacher_target, corpus.joint_lower_rad, corpus.joint_upper_rad
                )
                mixed_target = (
                    teacher_action_fraction * teacher_target
                    + (1.0 - teacher_action_fraction) * student_target
                )
                mixed_action = (
                    mixed_target[env.action_joint_ids]
                    - np.asarray(reference_qpos)[7:][env.action_joint_ids]
                ) / env._config.action_scale
                frames.append(frame)
                labels.append(np.asarray(teacher_target, dtype=np.float32))
                ready_flags.append(False)
                state = env.step(state, mixed_action)
                should_hold = (
                    ready_carry is not None
                    or step % route.time_dilation != route.time_dilation - 1
                )
                if should_hold:
                    state = _restore_reference(
                        env=env,
                        state=state,
                        carry=ready_carry if ready_carry is not None else carry,
                        mujoco=mujoco,
                    )
                current_qpos = np.asarray(env.mj_data.qpos, dtype=np.float64)
                current_qvel = np.asarray(env.mj_data.qvel, dtype=np.float64)
                finite_state = bool(
                    np.all(np.isfinite(current_qpos))
                    and np.all(np.isfinite(current_qvel))
                )
                executed_steps = step + 1
                if not finite_state:
                    break
                upright = float(-body_gravity_vector(current_qpos[3:7])[2])
                linear_speed = float(np.linalg.norm(current_qvel[:3]))
                angular_speed = float(np.linalg.norm(current_qvel[3:6]))
                peak_angular_speed = max(peak_angular_speed, angular_speed)
                ready_pose = bool(
                    current_qpos[2] >= exam_config.ready_pelvis_height_m
                    and upright >= exam_config.ready_upright_projection
                )
                ready_pose_streak = ready_pose_streak + 1 if ready_pose else 0
                if (
                    ready_carry is None
                    and ready_pose_streak >= exam_config.ready_pose_hold_frames
                ):
                    ready_carry = state.info["traj_info"]
                    ready_trigger_step = step
                stable = bool(
                    ready_pose
                    and linear_speed <= exam_config.maximum_stable_linear_speed_mps
                    and angular_speed <= exam_config.maximum_stable_angular_speed_rad_s
                )
                stable_streak = stable_streak + 1 if stable else 0
                maximum_stable_streak = max(maximum_stable_streak, stable_streak)
                ready_flags[-1] = ready_carry is not None
                if stable_streak >= exam_config.final_stable_frames:
                    break
            rollout_succeeded = bool(
                finite_state and stable_streak >= exam_config.final_stable_frames
            )
        finally:
            env.close()
        dagger_episodes.append(
            RecoveryTeacherEpisode(
                episode_id=f"dagger{dagger_round}-{episode_index:03d}",
                base_snapshot_hash=base.snapshot_hash,
                initial_snapshot_hash=initial.snapshot_hash,
                fixed_route_trial_hash=route.trial_hash,
                perturbation_hash=(
                    None if identity is None else identity.perturbation_hash
                ),
                proprio=np.stack(frames),
                absolute_motor_targets_rad=np.stack(labels),
                ready_handoff=np.asarray(ready_flags, dtype=np.bool_),
                time_dilation=route.time_dilation,
                teacher_succeeded=True,
                rollout_controller="MIXED_STUDENT_TEACHER",
                rollout_succeeded=rollout_succeeded,
            )
        )
        attempts.append(
            {
                "episode_index": episode_index,
                "base_snapshot_hash": base.snapshot_hash,
                "initial_snapshot_hash": initial.snapshot_hash,
                "fixed_route_trial_hash": route.trial_hash,
                "rollout_succeeded": rollout_succeeded,
                "finite_state": finite_state,
                "ready_handoff_triggered": ready_trigger_step is not None,
                "executed_sec": executed_steps * env.dt,
                "maximum_final_stable_sec": maximum_stable_streak * env.dt,
                "peak_root_angular_speed_rad_s": peak_angular_speed,
                "labeled_sample_count": len(frames),
            }
        )

    original_episodes = recovery_teacher_episodes_from_corpus(corpus)
    combined_episodes = original_episodes + tuple(dagger_episodes)
    corpus_name = f"recovery-student-dagger{dagger_round}-v1"
    manifest = write_recovery_distillation_corpus(
        episodes=combined_episodes,
        output_dir=target,
        corpus_name=corpus_name,
        proprioception_spec=corpus.proprioception_spec,
        teacher_policy_hash=str(corpus_payload["teacher_policy_hash"]),
        body_hash=str(corpus_payload["body_hash"]),
        physics_scene_hash=str(corpus_payload["physics_scene_hash"]),
        development_report_hash=str(corpus_payload["development_report_hash"]),
        default_joint_position_rad=corpus.default_joint_position_rad,
        joint_lower_rad=corpus.joint_lower_rad,
        joint_upper_rad=corpus.joint_upper_rad,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_student_dagger_collection.v1",
        "base_corpus_manifest_hash": corpus.manifest_hash,
        "base_collection_report_hash": collection["report_hash"],
        "development_report_hash": development["report_hash"],
        "sealed_holdout_report_hash": sealed["report_hash"],
        "student_artifact_manifest_hash": artifact["manifest_hash"],
        "dagger_round": dagger_round,
        "teacher_action_fraction": teacher_action_fraction,
        "student_action_fraction": 1.0 - teacher_action_fraction,
        "dagger_episode_count": len(dagger_episodes),
        "dagger_rollout_passed_count": sum(
            episode.rollout_succeeded for episode in dagger_episodes
        ),
        "dagger_rollout_pass_rate": sum(
            episode.rollout_succeeded for episode in dagger_episodes
        )
        / len(dagger_episodes),
        "dagger_labeled_sample_count": sum(
            episode.proprio.shape[0] for episode in dagger_episodes
        ),
        "combined_episode_count": len(combined_episodes),
        "combined_sample_count": manifest["sample_count"],
        "combined_corpus_manifest": f"{corpus_name}.json",
        "combined_corpus_manifest_hash": manifest["manifest_hash"],
        "attempts": attempts,
        "sealed_holdout_state_count": len(sealed_hashes),
        "training_holdout_overlap_count": 0,
        "teacher_role": "PRIVILEGED_DAGGER_LABELER_ONLY",
        "student_contains_reference_features": False,
        "promotion_eligible": False,
        "claim_boundary": "DAGGER_TRAINING_CORPUS_NOT_STUDENT_PHYSICS_EVIDENCE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target / "dagger-collection-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-policy", required=True, type=Path)
    parser.add_argument("--environment-config", required=True, type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--sealed-holdout-report", required=True, type=Path)
    parser.add_argument("--base-corpus-manifest", required=True, type=Path)
    parser.add_argument("--base-collection-report", required=True, type=Path)
    parser.add_argument("--student-artifact-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--teacher-action-fraction", default=0.75, type=float)
    parser.add_argument("--dagger-round", default=1, type=int)
    args = parser.parse_args()
    report = collect_recovery_student_dagger_corpus(
        opentrack_root=args.opentrack_root,
        teacher_policy_path=args.teacher_policy,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        base_corpus_manifest_path=args.base_corpus_manifest,
        base_collection_report_path=args.base_collection_report,
        student_artifact_manifest_path=args.student_artifact_manifest,
        output_dir=args.output_dir,
        teacher_action_fraction=args.teacher_action_fraction,
        dagger_round=args.dagger_round,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "dagger_rollout_pass_rate": report["dagger_rollout_pass_rate"],
                "dagger_labeled_sample_count": report["dagger_labeled_sample_count"],
                "training_holdout_overlap_count": report[
                    "training_holdout_overlap_count"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_recovery_student_dagger_corpus"]
