"""Evaluate a reference-free episodic recovery muscle-memory baseline."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _atomic_json,
    _file_hash,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _verified_development_report,
    _wilson_lower_bound,
)
from rosclaw_soccer.training.opentrack_recovery_student_collect import (
    _teacher_body_hash,
    _verified_holdout_report,
)
from rosclaw_soccer.training.opentrack_recovery_student_exam import (
    RecoveryStudentPhysicsTrial,
    _run_student_trial,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryDistillationCorpus,
    load_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryPerturbationConfig,
    build_recovery_perturbation_holdout,
)


def _memory_initial_vector(
    snapshot: RecoverySnapshot,
    corpus: RecoveryDistillationCorpus,
) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32],
        np.concatenate(
            (
                (
                    np.asarray(snapshot.qpos[7:], dtype=np.float32)
                    - corpus.default_joint_position_rad
                )
                / 0.030,
                (np.asarray(snapshot.qvel[6:], dtype=np.float32) * 0.05) / 0.004,
            )
        ).astype(np.float32),
    )


def _memory_candidates(
    corpus: RecoveryDistillationCorpus,
) -> tuple[tuple[int, str, str, NDArray[np.float32], NDArray[np.float32]], ...]:
    candidates: list[tuple[int, str, str, NDArray[np.float32], NDArray[np.float32]]] = []
    for row in corpus.rows:
        if str(row.get("rollout_controller", "PRIVILEGED_TEACHER")) != ("PRIVILEGED_TEACHER"):
            continue
        index = int(row["episode_index"])
        start = int(row["start_row"])
        count = int(row["row_count"])
        initial = np.concatenate(
            (
                corpus.proprio[start, 6:35] / 0.030,
                corpus.proprio[start, 35:64] / 0.004,
            )
        ).astype(np.float32)
        candidates.append(
            (
                index,
                str(row["base_snapshot_hash"]),
                str(row["initial_snapshot_hash"]),
                initial,
                np.asarray(
                    corpus.absolute_motor_targets_rad[start : start + count],
                    dtype=np.float32,
                ),
            )
        )
    if not candidates:
        raise ValueError("recovery skill memory has no privileged-teacher episodes")
    return tuple(candidates)


def _select_memory(
    snapshot: RecoverySnapshot,
    corpus: RecoveryDistillationCorpus,
    candidates: tuple[tuple[int, str, str, NDArray[np.float32], NDArray[np.float32]], ...],
) -> tuple[int, str, str, NDArray[np.float32], float]:
    exact = [item for item in candidates if item[2] == snapshot.snapshot_hash]
    if exact:
        selected = exact[0]
        distance = 0.0
    else:
        query = _memory_initial_vector(snapshot, corpus)
        selected = min(
            candidates,
            key=lambda item: float(np.mean(np.square(query - item[3]))),
        )
        distance = float(np.sqrt(np.mean(np.square(query - selected[3]))))
    return selected[0], selected[1], selected[2], selected[4], distance


def run_opentrack_recovery_skill_memory_exam(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    corpus_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Replay selected successful muscle memories using direct PD targets."""

    root = opentrack_root.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    holdout_path = sealed_holdout_report_path.expanduser().resolve()
    corpus_path = corpus_manifest_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not root.is_dir() or any(
        not path.is_file()
        for path in (
            environment_path,
            snapshot_path,
            development_path,
            holdout_path,
            corpus_path,
        )
    ):
        raise FileNotFoundError("recovery skill-memory exam inputs are incomplete")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("recovery skill-memory output must be new and external")
    development = _verified_development_report(development_path)
    sealed = _verified_holdout_report(holdout_path)
    corpus = load_recovery_distillation_corpus(corpus_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if (
        sealed.get("development_report_hash") != development["report_hash"]
        or development.get("snapshot_manifest_hash") != _file_hash(snapshot_path)
        or development.get("teacher_config_hash") != _file_hash(environment_path)
        or corpus_payload.get("development_report_hash") != development["report_hash"]
        or corpus_payload.get("contains_reference_features") is not False
    ):
        raise ValueError("recovery skill-memory evidence bindings differ")
    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    holdout_config = RecoveryPerturbationConfig(**dict(sealed["perturbation_config"]))
    sealed_snapshots = build_recovery_perturbation_holdout(base_snapshots, config=holdout_config)
    for (snapshot, identity), recorded in zip(
        sealed_snapshots, sealed["perturbations"], strict=True
    ):
        if (
            snapshot.snapshot_hash != recorded["perturbed_snapshot_hash"]
            or identity.perturbation_hash != recorded["perturbation_hash"]
        ):
            raise ValueError("recovery skill-memory sealed state identity differs")

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.play.play_g1_env_tracking_general")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    if not isinstance(environment_payload.get("env_config"), dict):
        raise ValueError("recovery skill-memory environment config is invalid")
    first_route = development["post_skill_transfer"]["development_schedule"]["selected_trials"][0]
    match = first_route["match"]
    environment_config = copy.deepcopy(
        tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
    )
    environment_config.update(environment_payload["env_config"])
    environment_config.reference_traj_config.name = {motion_dataset_id: [match["motion_id"]]}
    environment_config.reference_traj_config.random_start = False
    environment_config.reference_traj_config.fixed_start_frame = match["entry_frame"]
    environment_class = tmj.registry.get("G1TrackingGeneral", "tracking_play_env_class")
    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        environment = environment_class(
            config=environment_config,
            play_ref_motion=False,
            use_viewer=False,
            use_renderer=False,
            exp_name="rosclaw-s52-recovery-skill-memory-exam",
        )
    finally:
        os.chdir(previous_directory)
    try:
        if (
            _teacher_body_hash(environment, mujoco) != corpus_payload["body_hash"]
            or _file_hash(Path(constants.task_to_xml("flat_terrain")).expanduser().resolve())
            != corpus_payload["physics_scene_hash"]
        ):
            raise ValueError("recovery skill-memory body or scene differs")
        exam_payload = dict(development["exam_config"])
        exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
        exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
        candidates = _memory_candidates(corpus)
        memory_hash = str(
            hash_json(
                {
                    "schema_version": "rosclaw_soccer.recovery_skill_memory.v1",
                    "corpus_manifest_hash": corpus.manifest_hash,
                    "candidate_episode_indexes": [item[0] for item in candidates],
                    "selection_features": ("JOINT_POSITION_FROM_DEFAULT_AND_SCALED_JOINT_VELOCITY"),
                    "selection_metric": "NORMALIZED_RMS_NEAREST_NEIGHBOR",
                    "phase_progression": "INTERNAL_MONOTONIC_CONTROL_STEP",
                    "contains_reference_features": False,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        trials: list[RecoveryStudentPhysicsTrial] = []
        selections: list[dict[str, Any]] = []
        suite_states: tuple[
            tuple[
                Literal["DEVELOPMENT_BASE", "SEALED_LOCAL_HOLDOUT"],
                list[tuple[RecoverySnapshot, str]],
            ],
            ...,
        ] = (
            ("DEVELOPMENT_BASE", [(item, item.snapshot_hash) for item in base_snapshots]),
            (
                "SEALED_LOCAL_HOLDOUT",
                [
                    (snapshot, identity.base_snapshot_hash)
                    for snapshot, identity in sealed_snapshots
                ],
            ),
        )
        for suite, states in suite_states:
            for snapshot, base_hash in states:
                index, selected_base, selected_initial, sequence, distance = _select_memory(
                    snapshot, corpus, candidates
                )
                trial, _ = _run_student_trial(
                    env=environment,
                    session=None,
                    snapshot=snapshot,
                    base_snapshot_hash=base_hash,
                    suite=suite,
                    student_onnx_hash=memory_hash,
                    corpus=corpus,
                    exam_config=exam_config,
                    constants=constants,
                    mujoco=mujoco,
                    absolute_target_sequence=sequence,
                    preserve_sequence_target_authority=True,
                )
                trials.append(trial)
                selections.append(
                    {
                        "trial_hash": trial.trial_hash,
                        "selected_episode_index": index,
                        "selected_base_snapshot_hash": selected_base,
                        "selected_initial_snapshot_hash": selected_initial,
                        "selection_distance": distance,
                        "memory_length": sequence.shape[0],
                    }
                )
    finally:
        environment.close()
    base_trials = [item for item in trials if item.suite == "DEVELOPMENT_BASE"]
    holdout_trials = [item for item in trials if item.suite == "SEALED_LOCAL_HOLDOUT"]
    passed = sum(item.succeeded for item in holdout_trials)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_skill_memory_exam.v1",
        "development_report_hash": development["report_hash"],
        "sealed_holdout_report_hash": sealed["report_hash"],
        "corpus_manifest_hash": corpus.manifest_hash,
        "skill_memory_hash": memory_hash,
        "control_path": "PROPRIO_SELECTED_EPISODIC_MEMORY_TO_ABSOLUTE_TARGET_TO_PD",
        "selection_features": "JOINT_POSITION_AND_VELOCITY_PROPRIOCEPTION_ONLY",
        "phase_progression": "INTERNAL_MONOTONIC_CONTROL_STEP",
        "motor_target_authority": "RECORDED_TEACHER_PD_TARGET_ENVELOPE",
        "physical_joint_limits_enforced_by_mujoco": True,
        "torque_limits_enforced_each_substep": True,
        "reference_phase_reads_during_control": 0,
        "teacher_identity_reads_during_control": 0,
        "environment_step_calls_during_control": 0,
        "trajectory_handler_reads_during_control": 0,
        "base_trial_count": len(base_trials),
        "base_passed_count": sum(item.succeeded for item in base_trials),
        "base_pass_rate": sum(item.succeeded for item in base_trials) / len(base_trials),
        "sealed_holdout_trial_count": len(holdout_trials),
        "sealed_holdout_passed_count": passed,
        "sealed_holdout_pass_rate": passed / len(holdout_trials),
        "sealed_holdout_wilson_95_lower_bound": _wilson_lower_bound(
            passed=passed, count=len(holdout_trials)
        ),
        "trials": [item.to_dict() | {"trial_hash": item.trial_hash} for item in trials],
        "selections": selections,
        "maximum_peak_root_angular_speed_rad_s": max(
            item.peak_root_angular_speed_rad_s for item in trials
        ),
        "mean_torque_saturation_fraction": float(
            np.mean([item.torque_saturation_fraction for item in trials])
        ),
        "student_contains_reference_features": False,
        "physical_truth": True,
        "physics_backend": "opentrack_mujoco_cpu_direct_pd",
        "promotion_eligible": False,
        "promotion_blockers": [
            *([] if passed == len(holdout_trials) else ["SEALED_LOCAL_HOLDOUT_FAILED"]),
            "EPISODIC_MEMORY_IS_NOT_A_COMPACT_NEURAL_STUDENT",
            "LOCAL_PERTURBATION_HOLDOUT_IS_NOT_NEW_POST_SKILL_EPISODES",
            "NO_SOURCE_SCENE_FULL_CHAIN_ROLLOUT",
        ],
        "claim_boundary": "DATA_DRIVEN_MUSCLE_MEMORY_BASELINE_NOT_DEPLOYMENT_PROMOTION",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--environment-config", required=True, type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--sealed-holdout-report", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    report = run_opentrack_recovery_skill_memory_exam(
        opentrack_root=args.opentrack_root,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        corpus_manifest_path=args.corpus_manifest,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "base_pass_rate": report["base_pass_rate"],
                "sealed_holdout_pass_rate": report["sealed_holdout_pass_rate"],
                "promotion_eligible": report["promotion_eligible"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_opentrack_recovery_skill_memory_exam"]
