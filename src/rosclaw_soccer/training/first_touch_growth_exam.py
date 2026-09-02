"""Compare a frozen First Touch baseline with one matched learned adapter.

The exam consumes CPU MuJoCo reports produced by ``first_touch_physics``.  It
does not rerun physics, score pixels, promote a policy, or open hardware.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _load_bound_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    declared = str(report.pop("report_hash", ""))
    if declared != hash_json(report):
        raise ValueError(f"First Touch report hash mismatch: {resolved}")
    if report.get("schema_version") != "rosclaw_soccer.first_touch_physics_evidence.v1":
        raise ValueError("unsupported First Touch physics report")
    physics = report.get("physics", {})
    trajectory = resolved.parent / str(physics.get("trajectory_artifact", ""))
    if not trajectory.is_file() or physics.get("trajectory_artifact_hash") != hash_bytes(
        trajectory.read_bytes()
    ):
        raise ValueError("First Touch trajectory artifact is missing or changed")
    report["report_hash"] = declared
    return cast(dict[str, Any], report)


def _task_loss(report: dict[str, Any]) -> float:
    measurement = report["measurement"]
    gate = report["gate"]
    return float(
        measurement["target_error_m"] / gate["maximum_target_error_m"]
        + measurement["direction_error_deg"] / gate["maximum_direction_error_deg"]
        + measurement["next_action_latency_sec"] / gate["maximum_next_action_latency_sec"]
        + max(
            0.0,
            measurement["outgoing_speed_mps"] - gate["maximum_outgoing_speed_mps"],
        )
        / gate["maximum_outgoing_speed_mps"]
    )


def compare_first_touch_growth(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    candidate_replay_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Persist a fail-closed, matched-scenario acquisition comparison."""

    output = output_path.expanduser().resolve()
    if output.exists():
        raise ValueError("First Touch growth exam output already exists")
    baseline = _load_bound_report(baseline_report_path)
    candidate = _load_bound_report(candidate_report_path)
    candidate_replay = _load_bound_report(candidate_replay_report_path)
    matched_fields = {
        "scenario_hash": baseline["scenario_hash"] == candidate["scenario_hash"],
        "gate_config_hash": baseline["evaluation"]["gate_config_hash"]
        == candidate["evaluation"]["gate_config_hash"],
        "body_hash": baseline["provenance"]["body_hash"] == candidate["provenance"]["body_hash"],
        "kick_prior_hash": baseline["provenance"]["kick_prior_hash"]
        == candidate["provenance"]["kick_prior_hash"],
        "implementation_hash": baseline["provenance"]["implementation_hash"]
        == candidate["provenance"]["implementation_hash"],
        "source_commit": baseline["provenance"]["source_commit"]
        == candidate["provenance"]["source_commit"],
    }
    trajectories_independent = (
        baseline["physics"]["trajectory_digest"] != candidate["physics"]["trajectory_digest"]
    )
    candidates_independent = baseline["candidate_hash"] != candidate["candidate_hash"]
    deterministic_candidate_replay = bool(
        candidate_replay["scenario_hash"] == candidate["scenario_hash"]
        and candidate_replay["candidate_hash"] == candidate["candidate_hash"]
        and candidate_replay["evaluation"]["gate_config_hash"]
        == candidate["evaluation"]["gate_config_hash"]
        and candidate_replay["provenance"] == candidate["provenance"]
        and candidate_replay["measurement_hash"] == candidate["measurement_hash"]
        and candidate_replay["evaluation_hash"] == candidate["evaluation_hash"]
        and candidate_replay["physics"]["trajectory_digest"]
        == candidate["physics"]["trajectory_digest"]
        and candidate_replay["physics"]["trajectory_artifact_hash"]
        == candidate["physics"]["trajectory_artifact_hash"]
    )
    baseline_passed = bool(baseline["evaluation"]["passed"])
    candidate_passed = bool(candidate["evaluation"]["passed"])
    candidate_safe = bool(
        candidate["physics"]["finite_state"]
        and not candidate["physics"]["joint_limit_violation"]
        and not candidate["physics"]["torque_limit_violation"]
        and not candidate["physics"]["post_touch_fall"]
    )
    baseline_loss = _task_loss(baseline)
    candidate_loss = _task_loss(candidate)
    paired_passed = bool(
        all(matched_fields.values())
        and trajectories_independent
        and candidates_independent
        and deterministic_candidate_replay
        and not baseline_passed
        and candidate_passed
        and candidate_safe
        and candidate_loss < baseline_loss
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.first_touch_paired_growth_exam.v1",
        "status": "PASS_PAIRED_ACQUISITION" if paired_passed else "REJECTED_PAIRED_ACQUISITION",
        "matched_fields": matched_fields,
        "trajectories_independent": trajectories_independent,
        "candidates_independent": candidates_independent,
        "deterministic_candidate_replay": deterministic_candidate_replay,
        "baseline": {
            "report_hash": baseline["report_hash"],
            "file_hash": hash_bytes(baseline_report_path.expanduser().resolve().read_bytes()),
            "candidate_hash": baseline["candidate_hash"],
            "passed": baseline_passed,
            "primary_failure": baseline["evaluation"]["primary_failure"],
            "task_loss": baseline_loss,
        },
        "candidate": {
            "report_hash": candidate["report_hash"],
            "file_hash": hash_bytes(candidate_report_path.expanduser().resolve().read_bytes()),
            "candidate_hash": candidate["candidate_hash"],
            "passed": candidate_passed,
            "safe": candidate_safe,
            "task_loss": candidate_loss,
        },
        "candidate_replay": {
            "report_hash": candidate_replay["report_hash"],
            "file_hash": hash_bytes(
                candidate_replay_report_path.expanduser().resolve().read_bytes()
            ),
            "trajectory_digest": candidate_replay["physics"]["trajectory_digest"],
            "measurement_hash": candidate_replay["measurement_hash"],
            "evaluation_hash": candidate_replay["evaluation_hash"],
        },
        "improvement": {
            "task_loss_reduction": baseline_loss - candidate_loss,
            "target_error_reduction_m": baseline["measurement"]["target_error_m"]
            - candidate["measurement"]["target_error_m"],
            "direction_error_reduction_deg": baseline["measurement"]["direction_error_deg"]
            - candidate["measurement"]["direction_error_deg"],
            "next_action_latency_reduction_sec": baseline["measurement"]["next_action_latency_sec"]
            - candidate["measurement"]["next_action_latency_sec"],
        },
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "single_matched_scenario": True,
            "promotion_eligible": False,
            "hardware_command_sent": False,
            "statement": (
                "A paired acquisition pass demonstrates local learning only; balanced "
                "acquisition and retention suites remain mandatory."
            ),
        },
    }
    report["exam_hash"] = hash_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_first_touch_growth(
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        candidate_replay_report_path=args.candidate_replay_report,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = ["compare_first_touch_growth"]
