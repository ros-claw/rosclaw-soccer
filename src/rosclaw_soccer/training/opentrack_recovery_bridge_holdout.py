"""Fixed-route perturbation holdout for the OpenTrack recovery bridge.

The development exam may search teacher motion, entry phase and time dilation.
This exam freezes those choices before creating deterministic unseen local
perturbations.  It measures robustness of a privileged cross-scene teacher;
it does not turn that teacher into a deployable recovery gate.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _append_trial_journal,
    _atomic_json,
    _file_hash,
    _git_head,
    _load_trial_journal,
    _run_bridge_trial,
    _scene_compatibility,
    _trial_key,
)
from rosclaw_soccer.training.recovery_snapshot import (
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatch,
    RecoveryEntryMatcher,
    RecoveryEntrySearchConfig,
    RecoveryPerturbationConfig,
    build_recovery_perturbation_holdout,
)

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _verified_development_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery bridge development report must be an object")
    declared_hash = payload.pop("report_hash", None)
    if (
        payload.get("schema_version")
        != "rosclaw_soccer.opentrack_recovery_bridge_exam.v1"
        or declared_hash != hash_json(payload)
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery bridge development report integrity failed")
    schedule = payload.get("post_skill_transfer", {}).get("development_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("recovery bridge development schedule is absent")
    schedule_hash = schedule.pop("schedule_hash", None)
    if schedule_hash != hash_json(schedule):
        raise ValueError("recovery bridge development schedule integrity failed")
    schedule["schedule_hash"] = schedule_hash
    payload["report_hash"] = declared_hash
    return payload


def _trial_from_dict(payload: dict[str, Any]) -> RecoveryBridgeTrial:
    raw = dict(payload)
    recorded_hash = raw.pop("trial_hash", None)
    match_payload = raw.pop("match", None)
    if not isinstance(match_payload, dict):
        raise ValueError("fixed recovery route has no valid match")
    trial = RecoveryBridgeTrial(match=RecoveryEntryMatch(**match_payload), **raw)
    if trial.trial_hash != recorded_hash:
        raise ValueError("fixed recovery route integrity failed")
    return trial


def _wilson_lower_bound(*, passed: int, count: int, z: float = 1.96) -> float:
    if count <= 0 or not 0 <= passed <= count:
        raise ValueError("Wilson interval counts are invalid")
    proportion = passed / count
    denominator = 1.0 + z * z / count
    centre = proportion + z * z / (2.0 * count)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    )
    return max(0.0, (centre - margin) / denominator)


def run_opentrack_recovery_bridge_holdout(
    *,
    opentrack_root: Path,
    teacher_policy_path: Path,
    teacher_config_path: Path,
    motion_paths: tuple[Path, ...],
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    source_scene_path: Path,
    development_report_path: Path,
    output_path: Path,
    perturbation_config: RecoveryPerturbationConfig | None = None,
) -> dict[str, Any]:
    """Run unseen local perturbations with all development choices frozen."""

    root = opentrack_root.expanduser().resolve()
    policy_path = teacher_policy_path.expanduser().resolve()
    teacher_configuration_path = teacher_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    source_path = source_scene_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    required_files = (
        policy_path,
        teacher_configuration_path,
        snapshot_path,
        source_path,
        development_path,
    )
    if not root.is_dir() or any(not path.is_file() for path in required_files):
        raise FileNotFoundError("OpenTrack recovery holdout inputs are incomplete")
    if not _DATASET_ID.fullmatch(motion_dataset_id):
        raise ValueError("OpenTrack motion dataset id is invalid")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("holdout evidence must be new and outside the OpenTrack checkout")
    expected_motion_root = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1"
    )
    resolved_motions = tuple(path.expanduser().resolve() for path in motion_paths)
    if not resolved_motions or any(
        path.parent != expected_motion_root for path in resolved_motions
    ):
        raise ValueError("motion paths must belong to the declared OpenTrack dataset")

    development = _verified_development_report(development_path)
    if (
        development["teacher_policy_hash"] != _file_hash(policy_path)
        or development["teacher_config_hash"] != _file_hash(teacher_configuration_path)
        or development["snapshot_manifest_hash"] != _file_hash(snapshot_path)
    ):
        raise ValueError("holdout inputs differ from the frozen development exam")
    search_config = RecoveryEntrySearchConfig(**development["search_config"])
    exam_payload = dict(development["exam_config"])
    exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
    exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
    matcher = RecoveryEntryMatcher.from_paths(resolved_motions, config=search_config)
    if matcher.library_hash != development["reference_library_hash"]:
        raise ValueError("holdout reference library differs from development")

    schedule = development["post_skill_transfer"]["development_schedule"]
    raw_selected = schedule.get("selected_trials")
    if not isinstance(raw_selected, list) or not raw_selected:
        raise ValueError("holdout requires a non-empty frozen route schedule")
    selected = tuple(_trial_from_dict(item) for item in raw_selected)
    if any(not trial.succeeded for trial in selected):
        raise ValueError("holdout cannot evaluate an unsuccessful development route")
    routes = {trial.snapshot_hash: trial for trial in selected}
    corpus = load_recovery_snapshot_corpus(snapshot_path)
    if set(routes) != {item.snapshot_hash for item in corpus}:
        raise ValueError("frozen routes do not cover the snapshot corpus exactly")
    active_perturbation = perturbation_config or RecoveryPerturbationConfig()
    holdout = build_recovery_perturbation_holdout(
        corpus, config=active_perturbation
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
                exp_name="rosclaw-s51-recovery-bridge-holdout",
            )
        finally:
            os.chdir(previous_directory)

    teacher_hash = _file_hash(policy_path)
    journal_path = target.with_suffix(target.suffix + ".trials.jsonl")
    journal_bindings = {
        "development_report_hash": str(development["report_hash"]),
        "perturbation_config_hash": active_perturbation.config_hash,
        "reference_library_hash": matcher.library_hash,
        "snapshot_manifest_hash": str(development["snapshot_manifest_hash"]),
        "teacher_policy_hash": teacher_hash,
    }
    recovered_trials = _load_trial_journal(
        journal_path, expected_bindings=journal_bindings
    )
    trials: list[RecoveryBridgeTrial] = []
    traces: list[dict[str, Any]] = []
    perturbations: list[dict[str, Any]] = []
    for perturbed, perturbation in holdout:
        fixed = routes[perturbation.base_snapshot_hash]
        key = _trial_key(
            kind="holdout",
            snapshot_hash=perturbed.snapshot_hash,
            match_hash=fixed.match.match_hash,
            time_dilation=fixed.time_dilation,
        )
        if key in recovered_trials:
            trial, trace = recovered_trials[key]
        else:
            environment = make_env(fixed.match)
            try:
                trial, trace = _run_bridge_trial(
                    env=environment,
                    session=session,
                    snapshot=perturbed,
                    snapshot_hash=perturbed.snapshot_hash,
                    match=fixed.match,
                    teacher_policy_hash=teacher_hash,
                    time_dilation=fixed.time_dilation,
                    config=exam_config,
                    mujoco=mujoco,
                )
            finally:
                environment.close()
            trace.update(
                {
                    "base_snapshot_hash": perturbation.base_snapshot_hash,
                    "perturbation_hash": perturbation.perturbation_hash,
                    "fixed_development_trial_hash": fixed.trial_hash,
                }
            )
            _append_trial_journal(
                journal_path,
                {
                    "bindings": journal_bindings,
                    "key": key,
                    "trace": trace,
                    "trial": trial.to_dict() | {"trial_hash": trial.trial_hash},
                },
            )
        if (
            trace.get("base_snapshot_hash") != perturbation.base_snapshot_hash
            or trace.get("perturbation_hash") != perturbation.perturbation_hash
            or trace.get("fixed_development_trial_hash") != fixed.trial_hash
        ):
            raise ValueError("recovered holdout trial lost its fixed-route binding")
        trials.append(trial)
        traces.append(trace)
        perturbations.append(
            asdict(perturbation)
            | {"perturbation_hash": perturbation.perturbation_hash}
        )

    per_base: list[dict[str, Any]] = []
    for base_hash in sorted(routes):
        indices = [
            index
            for index, perturbation in enumerate(perturbations)
            if perturbation["base_snapshot_hash"] == base_hash
        ]
        passed = sum(trials[index].succeeded for index in indices)
        per_base.append(
            {
                "base_snapshot_hash": base_hash,
                "fixed_development_trial_hash": routes[base_hash].trial_hash,
                "passed_count": passed,
                "trial_count": len(indices),
                "pass_rate": passed / len(indices),
            }
        )
    passed_count = sum(item.succeeded for item in trials)
    pass_rate = passed_count / len(trials)
    local_holdout_passed = bool(
        pass_rate >= 0.80
        and all(item["pass_rate"] >= 2.0 / 3.0 for item in per_base)
        and all(item.finite_state for item in trials)
    )
    teacher_scene_path = Path(constants.task_to_xml("flat_terrain")).resolve()
    compatibility = _scene_compatibility(
        source_scene_path=source_path,
        teacher_scene_path=teacher_scene_path,
        mujoco=mujoco,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_recovery_bridge_holdout.v1",
        "development_report_hash": development["report_hash"],
        "development_schedule_hash": schedule["schedule_hash"],
        "route_selection_frozen_before_holdout": True,
        "route_reselection_count": 0,
        "perturbation_config": asdict(active_perturbation),
        "perturbation_config_hash": active_perturbation.config_hash,
        "perturbations": perturbations,
        "snapshot_count": len(corpus),
        "trial_count": len(trials),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "wilson_95_lower_bound": _wilson_lower_bound(
            passed=passed_count, count=len(trials)
        ),
        "acceptance_contract": {
            "minimum_overall_pass_rate": 0.80,
            "minimum_per_base_pass_rate": 2.0 / 3.0,
            "all_states_finite": True,
        },
        "local_holdout_passed": local_holdout_passed,
        "per_base_results": per_base,
        "trials": [
            item.to_dict() | {"trial_hash": item.trial_hash} for item in trials
        ],
        "trace_summaries": traces,
        "teacher_policy_hash": teacher_hash,
        "teacher_config_hash": _file_hash(teacher_configuration_path),
        "reference_library_hash": matcher.library_hash,
        "snapshot_manifest_hash": _file_hash(snapshot_path),
        "opentrack_commit": _git_head(root),
        "physics_backend": "opentrack_mujoco_cpu",
        "physical_truth": True,
        "scene_compatibility": compatibility,
        "trial_journal": {
            "path": journal_path.name,
            "hash": _file_hash(journal_path),
            "recovered_trial_count": len(recovered_trials),
            "completed_trial_count": len(trials),
        },
        "teacher_role": "PRIVILEGED_REFERENCE_CONDITIONED_TRAINING_TEACHER",
        "promotion_eligible": False,
        "promotion_blockers": [
            "REFERENCE_PHASE_AND_TEACHER_ID_ARE_PRIVILEGED",
            "LOCAL_PERTURBATION_HOLDOUT_IS_NOT_NEW_POST_SKILL_EPISODES",
            *([] if compatibility["scene_equivalent"] else ["PHYSICS_SCENE_NOT_EQUIVALENT"]),
            "NO_SOURCE_SCENE_FULL_CHAIN_ROLLOUT",
            "NO_PROPRIOCEPTIVE_STUDENT_DISTILLATION",
        ],
        "claim_boundary": (
            "FIXED_ROUTE_LOCAL_ROBUSTNESS_NOT_DEPLOYABLE_OR_SOURCE_SCENE_PROMOTION"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-policy", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--motion-path", required=True, action="append", type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--samples-per-snapshot", default=3, type=int)
    args = parser.parse_args()
    report = run_opentrack_recovery_bridge_holdout(
        opentrack_root=args.opentrack_root,
        teacher_policy_path=args.teacher_policy,
        teacher_config_path=args.teacher_config,
        motion_paths=tuple(args.motion_path),
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        source_scene_path=args.source_scene,
        development_report_path=args.development_report,
        output_path=args.output_path,
        perturbation_config=RecoveryPerturbationConfig(
            samples_per_snapshot=args.samples_per_snapshot
        ),
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "pass_rate": report["pass_rate"],
                "local_holdout_passed": report["local_holdout_passed"],
                "promotion_eligible": report["promotion_eligible"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_opentrack_recovery_bridge_holdout"]
