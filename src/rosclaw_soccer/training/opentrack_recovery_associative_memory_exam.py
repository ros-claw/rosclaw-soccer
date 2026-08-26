"""CPU MuJoCo exam for proprioceptive associative recovery muscle memory."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

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
from rosclaw_soccer.training.recovery_associative_memory import (
    RecoveryAssociativeMemory,
    RecoveryAssociativeMemoryConfig,
    build_recovery_memory_episodes,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import load_recovery_distillation_corpus
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryPerturbationConfig,
    build_recovery_perturbation_holdout,
)


def run_opentrack_recovery_associative_memory_exam(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    corpus_manifest_path: Path,
    output_path: Path,
    config: RecoveryAssociativeMemoryConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed memory bank; this function never trains on holdout states."""

    root = opentrack_root.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    holdout_path = sealed_holdout_report_path.expanduser().resolve()
    corpus_path = corpus_manifest_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    required = (
        environment_path,
        snapshot_path,
        development_path,
        holdout_path,
        corpus_path,
    )
    if not root.is_dir() or any(not path.is_file() for path in required):
        raise FileNotFoundError("associative recovery exam inputs are incomplete")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("associative recovery evidence must be new and external")
    active = config or RecoveryAssociativeMemoryConfig()
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
        raise ValueError("associative recovery evidence bindings differ")

    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    holdout_config = RecoveryPerturbationConfig(**dict(sealed["perturbation_config"]))
    sealed_snapshots = build_recovery_perturbation_holdout(
        base_snapshots, config=holdout_config
    )
    for (snapshot, identity), recorded in zip(
        sealed_snapshots, sealed["perturbations"], strict=True
    ):
        if (
            snapshot.snapshot_hash != recorded["perturbed_snapshot_hash"]
            or identity.perturbation_hash != recorded["perturbation_hash"]
        ):
            raise ValueError("associative recovery sealed state identity differs")
    sealed_hashes = {snapshot.snapshot_hash for snapshot, _ in sealed_snapshots}
    episodes = build_recovery_memory_episodes(corpus)
    if sealed_hashes & {episode.initial_snapshot_hash for episode in episodes}:
        raise ValueError("associative recovery memory overlaps sealed holdout")
    memory_hash = str(
        hash_json(
            {
                "schema_version": "rosclaw_soccer.recovery_associative_memory.v1",
                "corpus_manifest_hash": corpus.manifest_hash,
                "episode_initial_hashes": [
                    episode.initial_snapshot_hash for episode in episodes
                ],
                "config": asdict(active),
            }
        )
    )

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.play.play_g1_env_tracking_general")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    if not isinstance(environment_payload.get("env_config"), dict):
        raise ValueError("associative recovery environment config is invalid")
    first_route = development["post_skill_transfer"]["development_schedule"][
        "selected_trials"
    ][0]["match"]
    environment_config = copy.deepcopy(
        tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
    )
    environment_config.update(environment_payload["env_config"])
    environment_config.reference_traj_config.name = {
        motion_dataset_id: [first_route["motion_id"]]
    }
    environment_config.reference_traj_config.random_start = False
    environment_config.reference_traj_config.fixed_start_frame = first_route["entry_frame"]
    environment_class = tmj.registry.get("G1TrackingGeneral", "tracking_play_env_class")
    previous = Path.cwd()
    try:
        os.chdir(root)
        environment = environment_class(
            config=environment_config,
            play_ref_motion=False,
            use_viewer=False,
            use_renderer=False,
            exp_name="rosclaw-s53-associative-recovery-exam",
        )
    finally:
        os.chdir(previous)

    trials: list[RecoveryStudentPhysicsTrial] = []
    retrieval: list[dict[str, Any]] = []
    try:
        if (
            _teacher_body_hash(environment, mujoco) != corpus_payload["body_hash"]
            or _file_hash(Path(constants.task_to_xml("flat_terrain")).resolve())
            != corpus_payload["physics_scene_hash"]
        ):
            raise ValueError("associative recovery body or physics scene differs")
        exam_payload = dict(development["exam_config"])
        exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
        exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
        suites: tuple[
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
        for suite, states in suites:
            for snapshot, base_hash in states:
                memory = RecoveryAssociativeMemory(episodes, config=active)
                memory.reset(snapshot, base_snapshot_hash=base_hash, corpus=corpus)
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
                    absolute_target_provider=memory.target,
                    preserve_sequence_target_authority=True,
                )
                trials.append(trial)
                retrieval.append(
                    {
                        "trial_hash": trial.trial_hash,
                        "selection_switch_count": memory.selection_switch_count,
                        "phase_adjustment_count": memory.phase_adjustment_count,
                        "phase_hold_count": memory.phase_hold_count,
                        "maximum_query_distance": memory.maximum_query_distance,
                    }
                )
    finally:
        environment.close()

    base_trials = [trial for trial in trials if trial.suite == "DEVELOPMENT_BASE"]
    holdout_trials = [trial for trial in trials if trial.suite == "SEALED_LOCAL_HOLDOUT"]
    base_passed = sum(trial.succeeded for trial in base_trials)
    holdout_passed = sum(trial.succeeded for trial in holdout_trials)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_associative_memory_exam.v1",
        "development_report_hash": development["report_hash"],
        "sealed_holdout_report_hash": sealed["report_hash"],
        "corpus_manifest_hash": corpus.manifest_hash,
        "memory_hash": memory_hash,
        "memory_episode_count": len(episodes),
        "config": asdict(active),
        "control_path": (
            "CURRENT_PROPRIO_ASSOCIATIVE_MEMORY_TO_ABSOLUTE_TARGET_TO_DIRECT_PD"
        ),
        "phase_progression": "INTERNAL_MONOTONIC_CONTROL_STEP",
        "base_trial_count": len(base_trials),
        "base_passed_count": base_passed,
        "base_pass_rate": base_passed / len(base_trials),
        "sealed_holdout_trial_count": len(holdout_trials),
        "sealed_holdout_passed_count": holdout_passed,
        "sealed_holdout_pass_rate": holdout_passed / len(holdout_trials),
        "sealed_holdout_wilson_95_lower_bound": _wilson_lower_bound(
            passed=holdout_passed, count=len(holdout_trials)
        ),
        "trials": [trial.to_dict() | {"trial_hash": trial.trial_hash} for trial in trials],
        "retrieval": retrieval,
        "reference_phase_reads_during_control": 0,
        "teacher_identity_reads_during_control": 0,
        "environment_step_calls_during_control": 0,
        "trajectory_handler_reads_during_control": 0,
        "sealed_holdout_state_reads_during_memory_build": 0,
        "sealed_holdout_identity_reads_for_overlap_guard": len(sealed_hashes),
        "physical_truth": True,
        "physics_backend": "opentrack_mujoco_cpu_direct_pd",
        "promotion_eligible": False,
        "promotion_blockers": [
            *(
                []
                if holdout_passed == len(holdout_trials)
                else ["SEALED_LOCAL_HOLDOUT_FAILED"]
            ),
            "ASSOCIATIVE_MEMORY_IS_NOT_YET_COMPACT_NEURAL_POLICY",
            "NO_SOURCE_SCENE_FULL_CHAIN_ROLLOUT",
        ],
        "claim_boundary": "DATA_DRIVEN_ASSOCIATIVE_MEMORY_BASELINE_SIM_ONLY",
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
    parser.add_argument("--nearest-neighbors", default=3, type=int)
    args = parser.parse_args()
    report = run_opentrack_recovery_associative_memory_exam(
        opentrack_root=args.opentrack_root,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        corpus_manifest_path=args.corpus_manifest,
        output_path=args.output_path,
        config=RecoveryAssociativeMemoryConfig(nearest_neighbors=args.nearest_neighbors),
    )
    print(
        json.dumps(
            {
                "base_pass_rate": report["base_pass_rate"],
                "sealed_holdout_pass_rate": report["sealed_holdout_pass_rate"],
                "promotion_eligible": report["promotion_eligible"],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_opentrack_recovery_associative_memory_exam"]
