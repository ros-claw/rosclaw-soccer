from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.first_touch_growth_exam import compare_first_touch_growth

_HASH = "sha256:" + "a" * 64


def _report(root: Path, name: str, *, passed: bool, scenario_hash: str = _HASH) -> Path:
    directory = root / name
    directory.mkdir()
    trajectory = directory / "trajectory.npz"
    trajectory.write_bytes(name.encode())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.first_touch_physics_evidence.v1",
        "scenario_hash": scenario_hash,
        "candidate_hash": "sha256:" + ("b" if passed else "c") * 64,
        "gate": {
            "maximum_target_error_m": 0.35,
            "maximum_direction_error_deg": 20.0,
            "maximum_next_action_latency_sec": 0.7,
            "maximum_outgoing_speed_mps": 2.5,
        },
        "measurement": {
            "target_error_m": 0.1 if passed else 1.0,
            "direction_error_deg": 2.0 if passed else 180.0,
            "next_action_latency_sec": 0.04 if passed else 0.71,
            "outgoing_speed_mps": 1.2 if passed else 0.0,
        },
        "evaluation": {
            "gate_config_hash": _HASH,
            "passed": passed,
            "primary_failure": None if passed else "soccer.touch_too_soft",
        },
        "physics": {
            "trajectory_artifact": trajectory.name,
            "trajectory_artifact_hash": hash_bytes(trajectory.read_bytes()),
            "trajectory_digest": hash_bytes((name + ".trace").encode()),
            "finite_state": True,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "post_touch_fall": False,
        },
        "provenance": {
            "body_hash": _HASH,
            "kick_prior_hash": _HASH,
            "implementation_hash": _HASH,
            "source_commit": "1" * 40,
        },
    }
    report["report_hash"] = hash_json(report)
    path = directory / "first-touch-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_paired_exam_accepts_only_a_matched_safe_gain(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", passed=False)
    candidate = _report(tmp_path, "candidate", passed=True)

    report = compare_first_touch_growth(
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        output_path=tmp_path / "exam.json",
    )

    assert report["status"] == "PASS_PAIRED_ACQUISITION"
    assert report["improvement"]["task_loss_reduction"] > 0.0
    assert report["evidence_boundary"]["promotion_eligible"] is False


def test_paired_exam_rejects_a_scenario_mismatch(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", passed=False)
    candidate = _report(
        tmp_path,
        "candidate",
        passed=True,
        scenario_hash="sha256:" + "d" * 64,
    )

    report = compare_first_touch_growth(
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        output_path=tmp_path / "exam.json",
    )

    assert report["status"] == "REJECTED_PAIRED_ACQUISITION"
    assert report["matched_fields"]["scenario_hash"] is False
