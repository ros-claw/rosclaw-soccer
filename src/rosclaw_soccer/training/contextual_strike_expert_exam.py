"""Seal the S210 contextual strike-memory DAgger and holdout result."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.contextual_strike_expert_learning import (
    load_contextual_strike_expert_memory,
)
from rosclaw_soccer.training.contextual_strike_expert_probe import (
    validate_contextual_strike_expert_probe,
)
from rosclaw_soccer.training.dynamic_strike_coordination_probe import (
    validate_dynamic_strike_coordination_probe,
)

_TRAINING_CASES = (
    "train0600",
    "train0625",
    "train0650",
    "train0675",
    "train0700",
    "train0725",
    "train0750",
    "train0775",
    "train0800",
)
_HOLDOUT_CASES = (
    "hold06125",
    "hold06375",
    "hold06625",
    "hold06875",
    "hold07125",
    "hold07375",
    "hold07625",
    "hold07875",
)
_REJECTION_CASE = "reject0900"
_BASELINE_CASES = ("gy060", "gy065", "gy070", "gy075", "gy080")
_CORRECTED_FAILURE_TARGETS = (0.65, 0.70)
_RETAINED_TARGETS = (0.60, 0.75, 0.80)
_STATUS_PASS = "PASS_DAGGER_MEMORY_REJECT_HELDOUT_GENERALIZATION"
_STATUS_REJECTED = "REJECTED_DAGGER_MEMORY_EVIDENCE"


def run_contextual_strike_expert_exam(
    *,
    evidence_dir: Path,
    memory_artifact_dir: Path,
    baseline_probe_dir: Path,
    primary_probe_dir: Path,
    replay_probe_dir: Path,
) -> dict[str, Any]:
    """Evaluate frozen external evidence and persist one fail-closed report."""

    root = evidence_dir.expanduser().resolve()
    checkout = Path(__file__).parents[3]
    if root == checkout or checkout in root.parents:
        raise ValueError("contextual strike exam evidence must remain outside the checkout")
    if root.exists() and any(root.iterdir()):
        raise ValueError("contextual strike exam evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    report = _evaluate(
        memory_artifact_dir=memory_artifact_dir.expanduser().resolve(),
        baseline_probe_dir=baseline_probe_dir.expanduser().resolve(),
        primary_probe_dir=primary_probe_dir.expanduser().resolve(),
        replay_probe_dir=replay_probe_dir.expanduser().resolve(),
    )
    report["report_hash"] = hash_json(report)
    destination = root / "contextual-strike-expert-exam.json"
    _atomic_json(destination, report)
    return validate_contextual_strike_expert_exam(destination)


def validate_contextual_strike_expert_exam(path: Path) -> dict[str, Any]:
    """Re-evaluate every source artifact, route trace, and result gate."""

    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    declared = value.pop("report_hash", None)
    try:
        source_paths = value.get("source_paths")
        if not isinstance(source_paths, dict):
            raise ValueError("contextual strike exam source paths are absent")
        expected = _evaluate(
            memory_artifact_dir=Path(str(source_paths.get("memory_artifact")))
            .expanduser()
            .resolve(),
            baseline_probe_dir=Path(str(source_paths.get("baseline_probe_dir")))
            .expanduser()
            .resolve(),
            primary_probe_dir=Path(str(source_paths.get("primary_probe_dir")))
            .expanduser()
            .resolve(),
            replay_probe_dir=Path(str(source_paths.get("replay_probe_dir"))).expanduser().resolve(),
        )
        if declared != hash_json(value) or value != expected:
            raise ValueError("contextual strike exam evidence or implementation changed")
    finally:
        value["report_hash"] = declared
    return value


def _evaluate(
    *,
    memory_artifact_dir: Path,
    baseline_probe_dir: Path,
    primary_probe_dir: Path,
    replay_probe_dir: Path,
) -> dict[str, Any]:
    memory, training = load_contextual_strike_expert_memory(memory_artifact_dir)
    baseline_reports = {
        name: validate_dynamic_strike_coordination_probe(baseline_probe_dir / name / "probe.json")
        for name in _BASELINE_CASES
    }
    primary_reports = {
        name: validate_contextual_strike_expert_probe(primary_probe_dir / name / "probe.json")
        for name in (*_TRAINING_CASES, *_HOLDOUT_CASES, _REJECTION_CASE)
    }
    replay_reports = {
        name: validate_contextual_strike_expert_probe(replay_probe_dir / name / "probe.json")
        for name in (*_TRAINING_CASES, *_HOLDOUT_CASES, _REJECTION_CASE)
    }
    baseline_rows = [_baseline_row(name, baseline_reports[name]) for name in _BASELINE_CASES]
    training_rows = [_contextual_row(name, primary_reports[name]) for name in _TRAINING_CASES]
    holdout_rows = [_contextual_row(name, primary_reports[name]) for name in _HOLDOUT_CASES]
    rejection = _contextual_row(_REJECTION_CASE, primary_reports[_REJECTION_CASE])
    exact_replays = {
        name: _exact_replay(primary_reports[name], replay_reports[name])
        for name in (*_TRAINING_CASES, *_HOLDOUT_CASES, _REJECTION_CASE)
    }
    rejection_codes = _selection_codes(primary_probe_dir / _REJECTION_CASE / "trajectory.npz")
    baseline_successes = sum(bool(row["success"]) for row in baseline_rows)
    training_successes = sum(bool(row["success"]) for row in training_rows)
    holdout_successes = sum(bool(row["success"]) for row in holdout_rows)
    training_by_target = {float(row["target_y_m"]): row for row in training_rows}
    all_contextual = tuple(primary_reports.values()) + tuple(replay_reports.values())
    evidence_gates = {
        "memory_is_source_and_implementation_bound": bool(
            memory.memory_hash == training.get("memory_hash")
            and memory.dataset_snapshot_hash == training.get("dataset_snapshot_hash")
            and isinstance(training.get("source_files"), dict)
            and len(training["source_files"]) > 0
            and training.get("implementation_hash")
            == hash_json(training.get("implementation_files"))
        ),
        "baseline_is_three_of_five": baseline_successes == 3,
        "all_nine_training_anchors_pass": training_successes == len(_TRAINING_CASES),
        "dagger_failures_corrected": all(
            bool(training_by_target[target]["success"]) for target in _CORRECTED_FAILURE_TARGETS
        ),
        "previous_successes_retained": all(
            bool(training_by_target[target]["success"]) for target in _RETAINED_TARGETS
        ),
        "all_training_anchors_safe": all(bool(row["world_safe"]) for row in training_rows),
        "all_eighteen_exact_replays": all(exact_replays.values()),
        "unresolved_context_safely_abstains": bool(
            rejection["success"] is False
            and rejection["world_safe"] is True
            and rejection["selected_frames"] == 0
            and int(rejection["abstained_frames"]) > 0
            and rejection_codes
            and set(rejection_codes).issubset({2, 3})
        ),
        "sim_only_no_hardware": bool(
            all(
                report.get("activation_ceiling") == "SIM_ONLY"
                and report.get("hardware_command_sent") is False
                for report in (*baseline_reports.values(), *all_contextual)
            )
            and memory.activation_ceiling == "SIM_ONLY"
            and memory.hardware_authorized is False
        ),
    }
    holdout_rate = holdout_successes / len(_HOLDOUT_CASES)
    promotion_gates = {
        **evidence_gates,
        "unseen_midpoint_generalization_at_least_80pct": holdout_rate >= 0.80,
    }
    evidence_passed = bool(evidence_gates and all(evidence_gates.values()))
    promotion_eligible = bool(promotion_gates and all(promotion_gates.values()))
    source_paths = {
        "memory_artifact": str(memory_artifact_dir),
        "baseline_probe_dir": str(baseline_probe_dir),
        "primary_probe_dir": str(primary_probe_dir),
        "replay_probe_dir": str(replay_probe_dir),
    }
    source_files = _source_files(
        memory_artifact_dir=memory_artifact_dir,
        baseline_probe_dir=baseline_probe_dir,
        primary_probe_dir=primary_probe_dir,
        replay_probe_dir=replay_probe_dir,
    )
    implementation_files = _implementation_files()
    return {
        "schema_version": "rosclaw_soccer.contextual_strike_expert_exam.v1",
        "status": _STATUS_PASS if evidence_passed else _STATUS_REJECTED,
        "evidence_passed": evidence_passed,
        "promotion_eligible": promotion_eligible,
        "memory_hash": memory.memory_hash,
        "dataset_snapshot_hash": memory.dataset_snapshot_hash,
        "memory_training_report_hash": training["report_hash"],
        "summary": {
            "baseline_successes": baseline_successes,
            "baseline_cases": len(_BASELINE_CASES),
            "baseline_success_rate": baseline_successes / len(_BASELINE_CASES),
            "training_successes": training_successes,
            "training_cases": len(_TRAINING_CASES),
            "training_success_rate": training_successes / len(_TRAINING_CASES),
            "holdout_successes": holdout_successes,
            "holdout_cases": len(_HOLDOUT_CASES),
            "holdout_success_rate": holdout_rate,
            "exact_replays": sum(exact_replays.values()),
            "exact_replay_cases": len(exact_replays),
        },
        "baseline_rows": baseline_rows,
        "training_rows": training_rows,
        "holdout_rows": holdout_rows,
        "rejection_row": rejection,
        "rejection_selection_codes": rejection_codes,
        "exact_replays": exact_replays,
        "evidence_gates": evidence_gates,
        "promotion_gates": promotion_gates,
        "claim": {
            "qualified": "DATA_BOUND_DAGGER_GROWTH_WITH_HELDOUT_GENERALIZATION_REJECTED",
            "not_claimed": [
                "CONTINUOUS_TARGET_GENERALIZATION",
                "ONLINE_RL",
                "END_TO_END_CONTROL",
                "REAL_ROBOT_TRANSFER",
            ],
        },
        "why_not_promoted": (
            f"unseen target midpoint success is {holdout_successes}/{len(_HOLDOUT_CASES)} "
            f"({holdout_rate:.1%}), below the frozen 80% gate"
        ),
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "source_paths": source_paths,
        "source_files": source_files,
        "implementation_files": implementation_files,
        "implementation_hash": hash_json(implementation_files),
    }


def _baseline_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    assessment = cast(dict[str, Any], report["assessment"])
    gates = cast(dict[str, bool], assessment["gates"])
    metrics = cast(dict[str, Any], assessment["metrics"])
    projection = cast(dict[str, Any], report["shot_projection"])
    success = bool(
        report["world_safe"] is True
        and assessment["phase_sequence"] == [1, 2, 3, 4, 5, 6]
        and projection["whole_ball_inside_goal"] is True
        and gates.get("physical_teammate_pass_received") is True
        and gates.get("physical_foot_strike_in_strike_phase") is True
        and gates.get("stable_recovery_completed") is True
    )
    return {
        "case": name,
        "target_y_m": float(cast(dict[str, Any], report["goal_spec"])["target_y_m"]),
        "success": success,
        "world_safe": bool(report["world_safe"]),
        "phase_sequence": assessment["phase_sequence"],
        "whole_ball_inside_goal": bool(projection["whole_ball_inside_goal"]),
        "target_error_m": projection["target_error_m"],
        "receive_to_strike_sec": metrics["receive_to_strike_sec"],
        "peak_shot_speed_mps": metrics["peak_shot_speed_mps"],
        "trajectory_digest": report["trajectory_digest"],
    }


def _contextual_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    assessment = cast(dict[str, Any], report["assessment"])
    metrics = cast(dict[str, Any], assessment["metrics"])
    projection = cast(dict[str, Any], report["shot_projection"])
    route = cast(dict[str, Any], report["route_summary"])
    return {
        "case": name,
        "target_y_m": float(cast(dict[str, Any], report["goal_spec"])["target_y_m"]),
        "success": bool(report["candidate_success"]),
        "world_safe": bool(report["world_safe"]),
        "phase_sequence": assessment["phase_sequence"],
        "whole_ball_inside_goal": bool(projection["whole_ball_inside_goal"]),
        "target_error_m": projection["target_error_m"],
        "receive_to_strike_sec": metrics["receive_to_strike_sec"],
        "peak_shot_speed_mps": metrics["peak_shot_speed_mps"],
        "selected_expert_indices": route["selected_expert_indices"],
        "maximum_selected_distance": route["maximum_selected_distance"],
        "selected_frames": route["selected_frames"],
        "abstained_frames": route["abstained_frames"],
        "trajectory_digest": report["trajectory_digest"],
    }


def _exact_replay(primary: dict[str, Any], replay: dict[str, Any]) -> bool:
    return bool(
        primary["trajectory_digest"] == replay["trajectory_digest"]
        and primary["assessment"] == replay["assessment"]
        and primary["route_summary"] == replay["route_summary"]
        and primary["shot_projection"] == replay["shot_projection"]
        and primary["world_result"] == replay["world_result"]
        and primary["candidate_success"] is replay["candidate_success"]
    )


def _selection_codes(path: Path) -> list[int]:
    with np.load(path, allow_pickle=False) as archive:
        consulted = np.asarray(archive["strike_context_memory_consulted"], dtype=np.bool_)
        codes = np.asarray(archive["strike_context_selection_code"], dtype=np.int64)
    return sorted(set(int(value) for value in codes[consulted]))


def _source_files(
    *,
    memory_artifact_dir: Path,
    baseline_probe_dir: Path,
    primary_probe_dir: Path,
    replay_probe_dir: Path,
) -> dict[str, str]:
    paths = [memory_artifact_dir / "training.json"]
    for root, names in (
        (baseline_probe_dir, _BASELINE_CASES),
        (primary_probe_dir, (*_TRAINING_CASES, *_HOLDOUT_CASES, _REJECTION_CASE)),
        (replay_probe_dir, (*_TRAINING_CASES, *_HOLDOUT_CASES, _REJECTION_CASE)),
    ):
        for name in names:
            paths.extend((root / name / "probe.json", root / name / "trajectory.npz"))
    return {str(path): hash_bytes(path.read_bytes()) for path in paths}


def _implementation_files() -> dict[str, str]:
    root = Path(__file__).parents[1]
    files = {
        "memory": root / "growth" / "contextual_strike_experts.py",
        "learning": root / "training" / "contextual_strike_expert_learning.py",
        "probe": root / "training" / "contextual_strike_expert_probe.py",
        "world": root / "skills" / "team" / "independent_team_world.py",
        "exam": Path(__file__),
    }
    return {name: hash_bytes(path.read_bytes()) for name, path in files.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("contextual strike exam report must be an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--memory-artifact", required=True, type=Path)
    parser.add_argument("--baseline-probe-dir", required=True, type=Path)
    parser.add_argument("--primary-probe-dir", required=True, type=Path)
    parser.add_argument("--replay-probe-dir", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_contextual_strike_expert_exam(
        evidence_dir=arguments.evidence_dir,
        memory_artifact_dir=arguments.memory_artifact,
        baseline_probe_dir=arguments.baseline_probe_dir,
        primary_probe_dir=arguments.primary_probe_dir,
        replay_probe_dir=arguments.replay_probe_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "run_contextual_strike_expert_exam",
    "validate_contextual_strike_expert_exam",
]
