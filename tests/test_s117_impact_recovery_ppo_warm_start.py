from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_mjx import _tree_hash
from rosclaw_soccer.training.impact_recovery_ppo_warm_start import (
    ImpactRecoveryPPOWarmStartConfig,
    _whole_state_split,
    validate_impact_recovery_ppo_warm_start,
)

_DIGEST = "sha256:" + "1" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoints" / "000000000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "params").write_bytes(b"warm")
    (checkpoint / "ppo_network_config.json").write_text("{}", encoding="utf-8")
    warm_hash, warm_files = _tree_hash(checkpoint)
    parent_files = [{"path": "params", "size_bytes": 6, "hash": hash_bytes(b"parent")}]
    config = ImpactRecoveryPPOWarmStartConfig(
        training_steps=100,
        batch_size=16,
        calibration_state_count=2,
        exam_state_count=2,
        selection_interval_steps=10,
        trainable_scope="LOCATION_HEAD",
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_ppo_warm_start.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "parent_training_report_hash": _DIGEST,
        "parent_training_report_file_hash": _DIGEST,
        "parent_checkpoint_hash": hash_json(parent_files),
        "parent_checkpoint_files": parent_files,
        "distillation_report_hash": _DIGEST,
        "distillation_report_file_hash": _DIGEST,
        "teacher_report_hash": _DIGEST,
        "corpus_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "body_hash": _DIGEST,
        "split_semantics": "WHOLE_CURRICULUM_STATE_TRAIN_CALIBRATION_SEALED_EXAM",
        "split_states": {"training": [1, 2], "calibration": [3, 4], "exam": [5, 6]},
        "sample_counts": {"training": 20, "calibration": 10, "exam": 10},
        "training_backend": "JAX_CPU_SUPERVISED_ACTOR_ONLY",
        "trainable_scope": "LOCATION_HEAD",
        "normalizer_frozen": True,
        "critic_frozen": True,
        "scale_head_frozen_by_zero_gradient": True,
        "selection_metric": "CALIBRATION_ACTION_MSE",
        "best_step": 10,
        "training_sec": 1.0,
        "progress": [],
        "metrics": {
            "calibration_improvement_fraction": 0.20,
            "exam_improvement_fraction": 0.10,
        },
        "warm_checkpoint": "checkpoints/000000000001",
        "warm_checkpoint_hash": warm_hash,
        "warm_checkpoint_files": warm_files,
        "warm_start_eligible": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "warm-start-report.json"
    _write_json(path, report)
    return path


def test_warm_start_config_rejects_authority_and_unknown_scope() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryPPOWarmStartConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryPPOWarmStartConfig(trainable_scope="UNKNOWN")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryPPOWarmStartConfig(
            required_calibration_improvement_fraction=-0.01
        )
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryPPOWarmStartConfig(required_exam_improvement_fraction=-0.01)


def test_whole_state_split_is_deterministic_and_disjoint() -> None:
    state_index = np.repeat(np.arange(12, dtype=np.int32), 3)
    config = ImpactRecoveryPPOWarmStartConfig(
        training_steps=100,
        calibration_state_count=2,
        exam_state_count=2,
        selection_interval_steps=10,
    )

    first = _whole_state_split(state_index, config)
    second = _whole_state_split(state_index, config)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert set(first["training"]).isdisjoint(first["calibration"])
    assert set(first["training"]).isdisjoint(first["exam"])
    assert set(first["calibration"]).isdisjoint(first["exam"])


def test_warm_start_validator_recomputes_eligibility_and_checkpoint(tmp_path: Path) -> None:
    report = validate_impact_recovery_ppo_warm_start(_report(tmp_path))

    assert report["warm_start_eligible"] is True
    assert report["trainable_scope"] == "LOCATION_HEAD"
    assert report["promotion_authority"] == "NONE"


def test_warm_start_validator_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    path = _report(tmp_path)
    (tmp_path / "checkpoints" / "000000000001" / "params").write_bytes(b"changed")

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_ppo_warm_start(path)
