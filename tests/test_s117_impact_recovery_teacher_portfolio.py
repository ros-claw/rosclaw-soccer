from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_distillation import (
    ImpactRecoveryDistillationConfig,
)
from rosclaw_soccer.training.impact_recovery_teacher_portfolio import (
    ImpactRecoveryTeacherPortfolioConfig,
    build_impact_recovery_teacher_portfolio,
    validate_impact_recovery_teacher_portfolio,
)

_DIGEST = "sha256:" + "1" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_report(
    root: Path,
    *,
    name: str,
    states: tuple[int, ...],
    successful: bool,
) -> Path:
    destination = root / name
    destination.mkdir()
    config = ImpactRecoveryDistillationConfig(
        student_model_type="RIDGE_CURRENT_FRAME",
        label_outcome_mode="SUCCESSOR_FRONTIER",
        ridge_regularization_grid=(1.0, 10.0, 100.0),
        hidden_width=32,
        training_steps=100,
        holdout_state_count=2,
        required_validation_loss_improvement_fraction=0.0,
    )
    horizon = 2
    curriculum_index = np.repeat(np.asarray(states, dtype=np.int32), horizon)
    control_step = np.tile(np.arange(horizon, dtype=np.int32), len(states))
    observation = np.zeros((curriculum_index.size, 756), dtype=np.float32)
    signal = curriculum_index.astype(np.float32)[:, None] * 0.01
    observation[:, -189:-160] = signal + control_step[:, None] * 0.001
    action = np.tanh(observation[:, -189:-160]).astype(np.float32)
    accepted = np.ones(len(states), dtype=np.bool_)
    corpus = {
        "actor_observation": observation,
        "commanded_action": action,
        "curriculum_index": curriculum_index,
        "control_step": control_step,
        "accepted_state_row": np.repeat(accepted, horizon),
        "gated_state_accepted": accepted,
        "teacher_success": np.full(len(states), successful, dtype=np.bool_),
        "teacher_maximum_stable_streak": np.full(
            len(states), 25.0 if successful else 15.0, dtype=np.float32
        ),
        "cost_improvement_fraction": np.full(len(states), 0.5, dtype=np.float32),
    }
    corpus_path = destination / "gated-distillation-corpus.npz"
    np.savez_compressed(corpus_path, **corpus)  # type: ignore[arg-type]
    model = {
        "input_mean": np.zeros(189, dtype=np.float32),
        "input_std": np.ones(189, dtype=np.float32),
        "weight": np.zeros((189, 29), dtype=np.float32),
        "bias": np.zeros(29, dtype=np.float32),
    }
    model_path = destination / "student-model.npz"
    np.savez_compressed(model_path, **model)  # type: ignore[arg-type]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_distilled_student.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "teacher_report_hash": _DIGEST,
        "teacher_report_file_hash": _DIGEST,
        "teacher_corpus_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "curriculum_archive_hash": _DIGEST,
        "body_hash": _DIGEST,
        "action_semantics": "TEACHER_NOVELTY_GATED_BOUNDED_29_JOINT_PD_RESIDUAL",
        "residual_authority_steps": horizon,
        "corpus": corpus_path.name,
        "corpus_hash": hash_bytes(corpus_path.read_bytes()),
        "student_model": model_path.name,
        "student_model_hash": hash_bytes(model_path.read_bytes()),
        "training_metrics": {"validation_loss_improvement_fraction": 0.1},
        "student_exam_eligible": True,
        "device_count": 4,
        "all_devices_used": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "distillation-report.json"
    _write_json(report_path, report)
    return report_path


def test_portfolio_config_rejects_authority_and_trivial_coverage() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryTeacherPortfolioConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryTeacherPortfolioConfig(minimum_source_count=1)
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryTeacherPortfolioConfig(minimum_union_state_gain=0)


def test_portfolio_selects_one_robust_source_per_whole_state(tmp_path: Path) -> None:
    first = _source_report(
        tmp_path,
        name="first",
        states=(0, 1, 2, 3, 4, 5),
        successful=False,
    )
    second = _source_report(
        tmp_path,
        name="second",
        states=(2, 3, 4, 5, 6, 7),
        successful=True,
    )
    report = build_impact_recovery_teacher_portfolio(
        source_report_paths=(first, second),
        output_dir=tmp_path / "portfolio",
        source_checkout_path=Path(__file__).parents[1],
        config=ImpactRecoveryTeacherPortfolioConfig(minimum_union_state_gain=1),
    )

    assert report["union_state_count"] == 8
    assert report["union_state_gain"] == 2
    assert report["source_selected_state_counts"] == [2, 6]
    assert report["overlap_state_count"] == 4
    assert report["coverage_eligible"] is True
    assert report["portfolio_exam_eligible"] is True
    assert report["promotion_authority"] == "NONE"
    assert {
        row["source_index"] for row in report["state_winners"] if row["curriculum_index"] >= 2
    } == {1}


def test_portfolio_validator_recomputes_coverage(tmp_path: Path) -> None:
    first = _source_report(
        tmp_path,
        name="first",
        states=(0, 1, 2, 3, 4, 5),
        successful=False,
    )
    second = _source_report(
        tmp_path,
        name="second",
        states=(2, 3, 4, 5, 6, 7),
        successful=True,
    )
    destination = tmp_path / "portfolio"
    build_impact_recovery_teacher_portfolio(
        source_report_paths=(first, second),
        output_dir=destination,
        source_checkout_path=Path(__file__).parents[1],
        config=ImpactRecoveryTeacherPortfolioConfig(minimum_union_state_gain=1),
    )
    path = destination / "portfolio-report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["union_state_count"] += 1
    payload["report_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "report_hash"}
    )
    _write_json(path, payload)

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_teacher_portfolio(path)
