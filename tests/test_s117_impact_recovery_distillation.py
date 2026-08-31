from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_distillation import (
    ImpactRecoveryDistillationConfig,
    _student_policy,
    _train_student,
    validate_impact_recovery_distilled_evaluation,
    validate_impact_recovery_distilled_student,
)

_DIGEST = "sha256:" + "1" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_distilled_student_validator_binds_model_and_authority(tmp_path: Path) -> None:
    config = ImpactRecoveryDistillationConfig(hidden_width=32, training_steps=100)
    model = {
        "input_mean": np.zeros(12, np.float32),
        "input_std": np.ones(12, np.float32),
        "w1": np.zeros((12, 32), np.float32),
        "b1": np.zeros(32, np.float32),
        "w2": np.zeros((32, 32), np.float32),
        "b2": np.zeros(32, np.float32),
        "w3": np.zeros((32, 29), np.float32),
        "b3": np.zeros(29, np.float32),
    }
    model_path = tmp_path / "student-model.npz"
    np.savez_compressed(model_path, **model)  # type: ignore[arg-type]
    corpus_path = tmp_path / "gated-distillation-corpus.npz"
    np.savez_compressed(corpus_path, value=np.zeros(1, np.float32))
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
        "corpus": corpus_path.name,
        "corpus_hash": hash_bytes(corpus_path.read_bytes()),
        "student_model": model_path.name,
        "student_model_hash": hash_bytes(model_path.read_bytes()),
        "training_metrics": {"validation_loss_improvement_fraction": 0.5},
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
    report_path = tmp_path / "distillation-report.json"
    _write_json(report_path, report)

    validated = validate_impact_recovery_distilled_student(report_path)

    assert validated["student_model_hash"] == hash_bytes(model_path.read_bytes())
    policy = _student_policy(model)(None)
    action, _ = policy(np.zeros((2, 12), np.float32), np.zeros(2, np.uint32))
    assert np.asarray(action).shape == (2, 29)

    report["student_exam_eligible"] = False
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_distilled_student(report_path)


def test_distillation_config_rejects_hardware_authority() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryDistillationConfig(hardware_authorized=True)


def test_ridge_student_uses_current_frame_and_holds_out_whole_states() -> None:
    random = np.random.default_rng(117)
    state_index = np.repeat(np.arange(10, dtype=np.int32), 4)
    observation = random.normal(size=(40, 756)).astype(np.float32)
    action = np.tanh(observation[:, -189:-160]).astype(np.float32)
    config = ImpactRecoveryDistillationConfig(
        student_model_type="RIDGE_CURRENT_FRAME",
        ridge_regularization=1.0,
        hidden_width=32,
        training_steps=100,
        holdout_state_count=2,
    )

    model, metrics = _train_student(
        observation=observation,
        action=action,
        state_index=state_index,
        config=config,
    )

    assert model["weight"].shape == (189, 29)
    assert metrics["validation_state_count"] == 2


def test_distilled_evaluation_validator_recomputes_repeat_totals(tmp_path: Path) -> None:
    seeds = (57_151, 57_152)
    config = {
        "num_envs": 32,
        "seeds": list(seeds),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_evaluation_config.v1",
    }
    repeats = [
        {"seed": seeds[0], "success_count": 4, "success_rate": 4 / 32},
        {"seed": seeds[1], "success_count": 5, "success_rate": 5 / 32},
    ]
    population = {
        "episode_count": 64,
        "success_count": 9,
        "success_rate": 9 / 64,
        "repeats": repeats,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_distilled_evaluation.v1",
        "config": config,
        "config_hash": hash_json(config),
        "student_report_hash": _DIGEST,
        "student_model_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "populations": {"acquisition": population, "retention": dict(population)},
        "physics_backend": "MUJOCO_MJX",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "fixed exam",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "evaluation-report.json"
    _write_json(path, report)

    validated = validate_impact_recovery_distilled_evaluation(path)
    assert validated["populations"]["acquisition"]["success_count"] == 9

    report["populations"]["retention"]["success_count"] = 10
    report["populations"]["retention"]["success_rate"] = 10 / 64
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    _write_json(path, report)
    with pytest.raises(ValueError, match="repeat totals"):
        validate_impact_recovery_distilled_evaluation(path)
