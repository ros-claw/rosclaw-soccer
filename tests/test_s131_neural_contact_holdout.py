from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.neural_contact_holdout_exam import (
    _implementation_hash,
    fresh_neural_contact_holdouts,
    run_neural_contact_holdout_exam,
    validate_neural_contact_holdout_exam,
)


def test_neural_contact_holdouts_are_unique_and_predeclared() -> None:
    contexts = fresh_neural_contact_holdouts()
    assert len(contexts) == 6
    assert len({context.context_hash for context in contexts}) == len(contexts)
    assert all(context.case_id.startswith("s131.sealed.") for context in contexts)


def test_neural_contact_holdout_requires_explicit_lineage() -> None:
    parameters = inspect.signature(run_neural_contact_holdout_exam).parameters
    for name in (
        "source_canary_report_path",
        "teacher_discovery_report_path",
        "actor_training_report_path",
        "actor_path",
        "handoff_actor_path",
        "output_dir",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(validate_neural_contact_holdout_exam)


def test_rejected_holdout_remains_content_bound_evidence(tmp_path: Path) -> None:
    trajectory = tmp_path / "case.npz"
    trajectory.write_bytes(b"sealed-trajectory")
    artifact = {"file": trajectory.name, "file_hash": hash_bytes(trajectory.read_bytes())}
    primary = {
        "actor_active": True,
        "teacher_active": False,
        "scripted_contact_active": False,
        "trajectory": artifact,
    }
    request = {
        "schema_version": "rosclaw.growth.neural_contact_holdout_request.v1",
        "partition": "FRESH_S131_LOCAL_HOLDOUT",
        "minimum_goal_count": 4,
        "actor_hash": "sha256:actor",
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    report = {
        "schema_version": "rosclaw.growth.neural_contact_holdout_exam.v1",
        "status": "REJECTED_NEURAL_CONTACT_LOCAL_HOLDOUT",
        "promotion_eligible": False,
        "partition": "FRESH_S131_LOCAL_HOLDOUT",
        "request_hash": hash_bytes(request_path.read_bytes()),
        "actor_hash": "sha256:actor",
        "gates": {
            "all_exact_replay": True,
            "all_safe": True,
            "all_strict_right_foot_chain": False,
            "minimum_four_goals": False,
            "actor_executed_all": True,
            "actor_sole_contact_residual": True,
        },
        "rows": [
            {
                "goal": False,
                "safe": True,
                "strict_right_foot_chain": False,
                "exact_replay": True,
                "primary": primary,
                "replay": {"trajectory": artifact},
            }
        ],
        "implementation_hash": _implementation_hash(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = tmp_path / "holdout-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert validate_neural_contact_holdout_exam(report_path)["status"].startswith("REJECTED")

    report["status"] = "PASS_NEURAL_CONTACT_LOCAL_HOLDOUT"
    report["report_hash"] = hash_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="authority"):
        validate_neural_contact_holdout_exam(report_path)
