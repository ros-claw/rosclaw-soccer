from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_corrective_teacher import (
    RecoveryCorrectiveTeacherConfig,
    validate_recovery_corrective_teacher_evidence,
    write_recovery_corrective_teacher_evidence,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _lineage() -> dict[str, str]:
    names = (
        "failure_state_manifest_hash",
        "failure_state_manifest_file_hash",
        "failure_state_archive_hash",
        "parent_training_report_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
        "motion_archive_hash",
    )
    return {name: _digest(format(index, "x")) for index, name in enumerate(names)}


def _arrays(config: RecoveryCorrectiveTeacherConfig) -> dict[str, np.ndarray]:
    state_count = config.state_count
    observation = np.arange(state_count * 12, dtype=np.float32).reshape(state_count, 12) / 100
    baseline_action = np.zeros((state_count, 29), dtype=np.float32)
    plan = np.zeros((state_count, config.action_chunk_count, 29), dtype=np.float32)
    plan[:, 0, 0] = np.asarray((0.05, -0.04, 0.0, 0.06), dtype=np.float32)
    increment = np.array(plan[:, 0], copy=True)
    baseline_cost = np.ones((state_count,), dtype=np.float64)
    teacher_cost = np.asarray((0.80, 0.95, 1.00, 0.70), dtype=np.float64)
    improvement = (baseline_cost - teacher_cost) / baseline_cost
    finite = np.ones((state_count,), dtype=np.bool_)
    accepted = finite & (improvement >= config.minimum_cost_improvement_fraction)
    jacobian = np.zeros((state_count, 4, 29), dtype=np.float64)
    jacobian[:, 0, 0] = 2.0
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 2, 2] = 0.5
    jacobian[:, 3, 3] = 0.25
    return {
        "actor_observation": observation,
        "baseline_action": baseline_action,
        "corrective_action_increment": increment,
        "teacher_action": np.clip(baseline_action + increment, -1.0, 1.0),
        "teacher_plan": plan,
        "baseline_cost": baseline_cost,
        "teacher_cost": teacher_cost,
        "cost_improvement_fraction": improvement,
        "teacher_accepted": accepted,
        "finite_rollout": finite,
        "failure_state_index": np.arange(state_count, dtype=np.int32),
        "control_step": np.asarray((200, 300, 399, 400), dtype=np.int32),
        "baseline_effect_metrics": np.ones((state_count, 4), dtype=np.float64),
        "teacher_effect_metrics": np.full((state_count, 4), 0.8, dtype=np.float64),
        "action_effect_jacobian": jacobian,
    }


def test_corrective_teacher_config_requires_bounded_four_gpu_search() -> None:
    config = RecoveryCorrectiveTeacherConfig(state_count=4, candidate_count=64)
    assert config.action_chunk_count == 4
    assert config.elite_count == 8
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    with pytest.raises(ValueError, match="invalid"):
        RecoveryCorrectiveTeacherConfig(state_count=6)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryCorrectiveTeacherConfig(state_count=4, candidate_count=16)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryCorrectiveTeacherConfig(state_count=4, hardware_authorized=True)


def test_corrective_teacher_corpus_preserves_counterexamples_and_fails_closed(
    tmp_path: Path,
) -> None:
    config = RecoveryCorrectiveTeacherConfig(
        state_count=4,
        candidate_count=64,
        minimum_teacher_action_rms=0.001,
    )
    output = tmp_path / "evidence"
    report = write_recovery_corrective_teacher_evidence(
        output_dir=output,
        config=config,
        arrays=_arrays(config),
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        compiled_model_contract={"nq": 36, "nv": 35, "nu": 29},
    )
    assert report["teacher_search"]["accepted_count"] == 3
    assert report["teacher_search"]["rejected_counterexample_count"] == 1
    assert report["counterexamples_preserved"] is True
    assert report["action_effect_jacobian"]["locally_controllable"] is True
    assert report["supervised_warm_start_eligible"] is True
    assert report["promotion_authority"] == "NONE"

    report_path = output / "teacher-report.json"
    last_bit = json.loads(report_path.read_text(encoding="utf-8"))
    sensitivity = last_bit["action_effect_jacobian"]["median_metric_sensitivity_l2"]
    sensitivity["stability_deficit"] = float(np.nextafter(sensitivity["stability_deficit"], np.inf))
    last_bit.pop("report_hash")
    last_bit["report_hash"] = hash_json(last_bit)
    report_path.write_text(json.dumps(last_bit), encoding="utf-8")
    assert (
        validate_recovery_corrective_teacher_evidence(report_path)["report_hash"]
        == last_bit["report_hash"]
    )

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["promotion_authority"] = "HARDWARE"
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_teacher_evidence(report_path)


def test_corrective_teacher_rejects_inconsistent_labels(tmp_path: Path) -> None:
    config = RecoveryCorrectiveTeacherConfig(state_count=4, candidate_count=64)
    arrays = _arrays(config)
    arrays["teacher_accepted"] = np.ones((4,), dtype=np.bool_)
    with pytest.raises(ValueError, match="inconsistent"):
        write_recovery_corrective_teacher_evidence(
            output_dir=tmp_path / "bad-evidence",
            config=config,
            arrays=arrays,
            lineage=_lineage(),
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            compiled_model_contract={"nq": 36, "nv": 35, "nu": 29},
        )


def test_corrective_teacher_rejects_directional_tradeoff_label(tmp_path: Path) -> None:
    config = RecoveryCorrectiveTeacherConfig(state_count=4, candidate_count=64)
    arrays = _arrays(config)
    arrays["teacher_effect_metrics"][0, 0] = arrays["baseline_effect_metrics"][0, 0] + 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        write_recovery_corrective_teacher_evidence(
            output_dir=tmp_path / "directional-regression",
            config=config,
            arrays=arrays,
            lineage=_lineage(),
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            compiled_model_contract={"nq": 36, "nv": 35, "nu": 29},
        )
