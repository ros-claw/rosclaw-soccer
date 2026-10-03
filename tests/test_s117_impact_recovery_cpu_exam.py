from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from rosclaw.continual.champion_registry import (
    DominanceMetricRole,
    PairedDominanceEvidence,
    PairedDominanceMetric,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_cpu_exam import (
    ImpactRecoveryCPUExamConfig,
    _population_summary,
    _teacher_novelty_gate,
    validate_impact_recovery_cpu_exam,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    _teacher_novelty_gate as teacher_novelty_gate_jax,
)

_DIGEST = "sha256:" + "1" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _episodes(success_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(8):
        rows.append(
            {
                "seed": 1 if index < 4 else 2,
                "episode_index": index % 4,
                "curriculum_row": index,
                "success": index < success_count,
                "terminal_reason": "SUCCESS" if index < success_count else "TIME_LIMIT",
            }
        )
    return rows


def _report(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    suite = tmp_path / "scenario-suite.npz"
    np.savez_compressed(suite, value=np.arange(4, dtype=np.int32))
    config = ImpactRecoveryCPUExamConfig(
        episodes_per_seed=4,
        seeds=(1, 2),
        minimum_acquisition_gain_count=1,
        maximum_retention_drop_count=1,
    )
    populations = {
        "acquisition": _population_summary(_episodes(1), _episodes(3)),
        "retention": _population_summary(_episodes(7), _episodes(6)),
    }
    checkpoint_files = [{"path": "params", "size_bytes": 4, "hash": hash_bytes(b"safe")}]
    checkpoint_hash = hash_json(checkpoint_files)
    suite_hash = hash_bytes(suite.read_bytes())
    dominance = PairedDominanceEvidence(
        incumbent_artifact_hash=_DIGEST,
        challenger_artifact_hash=checkpoint_hash,
        scenario_suite_hash=suite_hash,
        metrics=(
            PairedDominanceMetric(
                metric_id="acquisition_success_count",
                incumbent_value=1.0,
                challenger_value=3.0,
                higher_is_better=True,
                role=DominanceMetricRole.OBJECTIVE,
                minimum_improvement=1.0,
            ),
            PairedDominanceMetric(
                metric_id="retention_success_count",
                incumbent_value=7.0,
                challenger_value=6.0,
                higher_is_better=True,
                role=DominanceMetricRole.GUARDRAIL,
                maximum_regression=1.0,
            ),
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_cpu_exam.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "training_report_hash": _DIGEST,
        "training_report_file_hash": _DIGEST,
        "gpu_evaluation_report_hash": _DIGEST,
        "gpu_evaluation_report_file_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "curriculum_archive_hash": _DIGEST,
        "body_hash": _DIGEST,
        "compiled_model_contract": {"model_hash": _DIGEST},
        "selected_checkpoint_hash": checkpoint_hash,
        "selected_checkpoint_files": checkpoint_files,
        "scenario_suite": suite.name,
        "scenario_suite_hash": suite_hash,
        "scenario_count": 16,
        "scenario_pairing": "IDENTICAL_INITIAL_QPOS_QVEL_AND_CURRICULUM_ROW",
        "incumbent": "CONTENT_BOUND_DYNAMIC_GAIN_MEMORY_WITH_ZERO_RESIDUAL",
        "challenger": "CONTENT_BOUND_DYNAMIC_GAIN_MEMORY_WITH_LEARNED_29_DOF_RESIDUAL",
        "populations": populations,
        "dominance_evidence": dominance.to_dict(),
        "dominance_evidence_hash": dominance.evidence_hash,
        "decision": "CANDIDATE_READY_FOR_TEAM_FULL_CHAIN_EXAM",
        "physics_backend": "CPU_MUJOCO",
        "jax_inference_backend": "CPU",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Isolated CPU-MuJoCo recovery exam; team full-chain exam still required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "cpu-exam.json"
    _write_json(path, report)
    return path


def test_cpu_exam_config_rejects_authority_and_invalid_suite() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCPUExamConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCPUExamConfig(episodes_per_seed=3)
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCPUExamConfig(seeds=(1, 1))


def test_numpy_teacher_gate_matches_training_contract() -> None:
    config = ImpactRecoveryMJXConfig(
        gain_memory_mode="DYNAMIC",
        residual_gate_mode="TEACHER_NOVELTY",
    )
    reference_qpos = np.zeros(36, dtype=np.float32)
    reference_qpos[2] = 0.75
    reference_qpos[3] = 1.0
    reference_qvel = np.zeros(35, dtype=np.float32)
    qpos = reference_qpos.copy()
    qpos[2] += 0.06
    qvel = reference_qvel.copy()
    qvel[3] = 0.4

    cpu = _teacher_novelty_gate(qpos, qvel, reference_qpos, reference_qvel, config)
    mjx = teacher_novelty_gate_jax(
        jnp.asarray(qpos),
        jnp.asarray(qvel),
        jnp.asarray(reference_qpos),
        jnp.asarray(reference_qvel),
        config,
    )

    assert cpu == pytest.approx(float(mjx), abs=1.0e-6)


def test_paired_summary_counts_rescues_and_regressions() -> None:
    summary = _population_summary(_episodes(4), _episodes(6))

    assert summary["incumbent_success_count"] == 4
    assert summary["challenger_success_count"] == 6
    assert summary["success_gain_count"] == 2
    assert summary["paired_outcomes"] == {
        "both_success": 4,
        "challenger_only_success": 2,
        "incumbent_only_success": 0,
        "both_failure": 2,
    }


def test_cpu_exam_validator_recomputes_pairing_and_decision(tmp_path: Path) -> None:
    path = _report(tmp_path)

    report = validate_impact_recovery_cpu_exam(path)

    assert report["decision"] == "CANDIDATE_READY_FOR_TEAM_FULL_CHAIN_EXAM"
    assert report["promotion_authority"] == "NONE"
    assert report["populations"]["acquisition"]["success_gain_count"] == 2


def test_cpu_exam_validator_rejects_aggregate_and_suite_tampering(tmp_path: Path) -> None:
    path = _report(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["populations"]["acquisition"]["challenger_success_count"] = 4
    payload["report_hash"] = hash_json({k: v for k, v in payload.items() if k != "report_hash"})
    _write_json(path, payload)

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_cpu_exam(path)

    second = tmp_path / "second"
    path = _report(second)
    (second / "scenario-suite.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_cpu_exam(path)
