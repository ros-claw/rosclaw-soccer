"""Training-only physics cross-validation for associative recovery memory."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _atomic_json,
    _file_hash,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _verified_development_report,
)
from rosclaw_soccer.training.opentrack_recovery_student_collect import (
    _teacher_body_hash,
)
from rosclaw_soccer.training.opentrack_recovery_student_exam import (
    _run_student_trial,
    _verified_json,
)
from rosclaw_soccer.training.recovery_associative_memory import (
    RecoveryAssociativeMemory,
    RecoveryAssociativeMemoryConfig,
    build_recovery_memory_episodes,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus
from rosclaw_soccer.training.recovery_student import load_recovery_distillation_corpus
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryPerturbationConfig,
    build_recovery_perturbation_holdout,
)


def _candidate_configs() -> dict[str, RecoveryAssociativeMemoryConfig]:
    return {
        "FIXED_NEAREST_1": RecoveryAssociativeMemoryConfig(
            nearest_neighbors=1,
            dynamic_retrieval=False,
        ),
        "PHASE_ADAPTIVE_5_15_P002": RecoveryAssociativeMemoryConfig(
            nearest_neighbors=1,
            dynamic_retrieval=False,
            phase_search_back_steps=5,
            phase_search_forward_steps=15,
            phase_deviation_penalty=0.02,
        ),
        "PHASE_ADAPTIVE_5_15_P010": RecoveryAssociativeMemoryConfig(
            nearest_neighbors=1,
            dynamic_retrieval=False,
            phase_search_back_steps=5,
            phase_search_forward_steps=15,
            phase_deviation_penalty=0.10,
        ),
        "PHASE_ADAPTIVE_10_30_P005": RecoveryAssociativeMemoryConfig(
            nearest_neighbors=1,
            dynamic_retrieval=False,
            phase_search_back_steps=10,
            phase_search_forward_steps=30,
            phase_deviation_penalty=0.05,
        ),
    }


def run_opentrack_recovery_associative_memory_cv(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    collection_report_path: Path,
    corpus_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select retrieval topology without loading any sealed evaluation state."""

    root = opentrack_root.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    collection_path = collection_report_path.expanduser().resolve()
    corpus_path = corpus_manifest_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    required = (
        environment_path,
        snapshot_path,
        development_path,
        collection_path,
        corpus_path,
    )
    if not root.is_dir() or any(not path.is_file() for path in required):
        raise FileNotFoundError("associative recovery CV inputs are incomplete")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("associative recovery CV evidence must be new and external")
    development = _verified_development_report(development_path)
    collection = _verified_json(
        collection_path,
        schema_version="rosclaw_soccer.recovery_student_collection.v1",
        hash_key="report_hash",
    )
    corpus = load_recovery_distillation_corpus(corpus_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if (
        development.get("snapshot_manifest_hash") != _file_hash(snapshot_path)
        or development.get("teacher_config_hash") != _file_hash(environment_path)
        or collection.get("development_report_hash") != development["report_hash"]
        or collection.get("corpus_manifest_hash") != corpus.manifest_hash
        or collection.get("training_holdout_overlap_count") != 0
        or corpus_payload.get("contains_reference_features") is not False
    ):
        raise ValueError("associative recovery CV evidence bindings differ")

    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    perturbation_config = RecoveryPerturbationConfig(
        **dict(collection["training_perturbation_config"])
    )
    generated = build_recovery_perturbation_holdout(
        base_snapshots, config=perturbation_config
    )
    all_episodes = build_recovery_memory_episodes(corpus)
    accepted = {episode.initial_snapshot_hash for episode in all_episodes}
    by_base: dict[str, list[Any]] = {}
    for snapshot, identity in generated:
        if snapshot.snapshot_hash in accepted:
            by_base.setdefault(identity.base_snapshot_hash, []).append(snapshot)
    validation = [
        snapshot
        for snapshots in by_base.values()
        for index, snapshot in enumerate(snapshots)
        if index % 4 == 3
    ]
    validation_hashes = frozenset(snapshot.snapshot_hash for snapshot in validation)
    memory_episodes = build_recovery_memory_episodes(
        corpus, excluded_initial_hashes=validation_hashes
    )
    if (
        len(validation) < 18
        or validation_hashes & {episode.initial_snapshot_hash for episode in memory_episodes}
    ):
        raise ValueError("associative recovery CV split is invalid")
    validation_rows = [
        (
            snapshot,
            next(
                identity.base_snapshot_hash
                for generated_snapshot, identity in generated
                if generated_snapshot.snapshot_hash == snapshot.snapshot_hash
            ),
        )
        for snapshot in validation
    ]
    base_hashes = {item.snapshot_hash for item in base_snapshots}
    if any(base_hash not in base_hashes for _, base_hash in validation_rows):
        raise ValueError("associative recovery CV base binding is invalid")

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.play.play_g1_env_tracking_general")
    constants = importlib.import_module("track_mj.envs.g1_tracking.g1_tracking_constants")
    environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    if not isinstance(environment_payload.get("env_config"), dict):
        raise ValueError("associative recovery CV environment config is invalid")
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
            exp_name="rosclaw-s53-associative-recovery-cv",
        )
    finally:
        os.chdir(previous)

    results: dict[str, Any] = {}
    try:
        if (
            _teacher_body_hash(environment, mujoco) != corpus_payload["body_hash"]
            or _file_hash(Path(constants.task_to_xml("flat_terrain")).resolve())
            != corpus_payload["physics_scene_hash"]
        ):
            raise ValueError("associative recovery CV body or physics scene differs")
        exam_payload = dict(development["exam_config"])
        exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
        exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
        for name, config in _candidate_configs().items():
            successes = 0
            switches = []
            phase_adjustments = []
            phase_holds = []
            torque_saturation = []
            trials = []
            candidate_hash = str(
                hash_json(
                    {
                        "schema_version": "rosclaw_soccer.recovery_associative_cv_candidate.v1",
                        "corpus_manifest_hash": corpus.manifest_hash,
                        "excluded_initial_hashes": sorted(validation_hashes),
                        "config": asdict(config),
                    }
                )
            )
            for snapshot, base_hash in validation_rows:
                memory = RecoveryAssociativeMemory(memory_episodes, config=config)
                memory.reset(snapshot, base_snapshot_hash=base_hash, corpus=corpus)
                trial, _ = _run_student_trial(
                    env=environment,
                    session=None,
                    snapshot=snapshot,
                    base_snapshot_hash=base_hash,
                    suite="DEVELOPMENT_BASE",
                    student_onnx_hash=candidate_hash,
                    corpus=corpus,
                    exam_config=exam_config,
                    constants=constants,
                    mujoco=mujoco,
                    absolute_target_provider=memory.target,
                    preserve_sequence_target_authority=True,
                )
                successes += int(trial.succeeded)
                switches.append(memory.selection_switch_count)
                phase_adjustments.append(memory.phase_adjustment_count)
                phase_holds.append(memory.phase_hold_count)
                torque_saturation.append(trial.torque_saturation_fraction)
                trials.append(trial.to_dict() | {"trial_hash": trial.trial_hash})
            results[name] = {
                "candidate_hash": candidate_hash,
                "config": asdict(config),
                "trial_count": len(validation_rows),
                "success_count": successes,
                "success_rate": successes / len(validation_rows),
                "mean_selection_switch_count": float(np.mean(switches)),
                "maximum_selection_switch_count": max(switches),
                "mean_phase_adjustment_count": float(np.mean(phase_adjustments)),
                "mean_phase_hold_count": float(np.mean(phase_holds)),
                "mean_torque_saturation_fraction": float(np.mean(torque_saturation)),
                "trials": trials,
            }
    finally:
        environment.close()

    winner_name = max(
        results,
        key=lambda name: (
            results[name]["success_rate"],
            -results[name]["mean_selection_switch_count"],
            -results[name]["mean_torque_saturation_fraction"],
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_associative_memory_cv.v1",
        "development_report_hash": development["report_hash"],
        "collection_report_hash": collection["report_hash"],
        "corpus_manifest_hash": corpus.manifest_hash,
        "training_perturbation_config_hash": perturbation_config.config_hash,
        "validation_initial_hashes": sorted(validation_hashes),
        "validation_trial_count": len(validation_rows),
        "memory_episode_count": len(memory_episodes),
        "sealed_holdout_reports_loaded": 0,
        "sealed_holdout_states_read": 0,
        "results": results,
        "selected_candidate": winner_name,
        "selected_config": results[winner_name]["config"],
        "physical_truth": True,
        "physics_backend": "opentrack_mujoco_cpu_direct_pd",
        "promotion_eligible": False,
        "claim_boundary": "TRAINING_ONLY_RETRIEVAL_SELECTION_NOT_PROMOTION_EVIDENCE",
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
    parser.add_argument("--collection-report", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    report = run_opentrack_recovery_associative_memory_cv(
        opentrack_root=args.opentrack_root,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        collection_report_path=args.collection_report,
        corpus_manifest_path=args.corpus_manifest,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "selected_candidate": report["selected_candidate"],
                "selected_success_rate": report["results"][report["selected_candidate"]][
                    "success_rate"
                ],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_opentrack_recovery_associative_memory_cv"]
