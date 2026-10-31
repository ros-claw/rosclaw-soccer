from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.impact_recovery_corrective_teacher import (
    ImpactRecoveryCorrectiveTeacherConfig,
    _aggregate_robust_variants,
    _selected_reset_keys,
    _write_evidence,
    validate_impact_recovery_corrective_teacher,
)

_DIGEST = "sha256:" + "1" * 64


def _arrays(config: ImpactRecoveryCorrectiveTeacherConfig) -> dict[str, np.ndarray[Any, Any]]:
    count = config.state_count
    baseline = np.full(count, 1.0, dtype=np.float64)
    teacher = np.full(count, 0.8, dtype=np.float64)
    plan = np.zeros((count, config.action_chunk_count, 29), dtype=np.float32)
    plan[:, 0, 0] = 0.1
    effect = np.full((count, 6), 0.2, dtype=np.float64)
    return {
        "actor_observation": np.zeros((count, 32), dtype=np.float32),
        "teacher_plan": plan,
        "corrective_action": plan[:, 0],
        "baseline_cost": baseline,
        "teacher_cost": teacher,
        "cost_improvement_fraction": (baseline - teacher) / baseline,
        "teacher_accepted": np.ones(count, dtype=np.bool_),
        "finite_rollout": np.ones(count, dtype=np.bool_),
        "curriculum_index": np.arange(count, dtype=np.int32),
        "elapsed_since_contact_sec": np.arange(count, dtype=np.float32) * 0.1,
        "baseline_effect_metrics": effect,
        "teacher_effect_metrics": effect - 0.01,
    }


def test_corrective_teacher_evidence_is_content_bound_and_sim_only(tmp_path: Path) -> None:
    config = ImpactRecoveryCorrectiveTeacherConfig(state_count=4, horizon_steps=10)
    lineage = {
        "curriculum_manifest_hash": _DIGEST,
        "curriculum_manifest_file_hash": _DIGEST,
        "curriculum_archive_hash": _DIGEST,
        "body_hash": _DIGEST,
        "training_model_hash": _DIGEST,
    }

    report = _write_evidence(
        output_dir=tmp_path / "teacher",
        config=config,
        arrays=_arrays(config),
        lineage=lineage,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        compiled_model={"model_hash": _DIGEST},
    )

    assert report["accepted_count"] == 4
    assert report["supervised_warm_start_eligible"] is True
    assert report["promotion_authority"] == "NONE"


def test_corrective_teacher_rejects_archive_tampering(tmp_path: Path) -> None:
    config = ImpactRecoveryCorrectiveTeacherConfig(state_count=4, horizon_steps=10)
    lineage = {
        name: _DIGEST
        for name in (
            "curriculum_manifest_hash",
            "curriculum_manifest_file_hash",
            "curriculum_archive_hash",
            "body_hash",
            "training_model_hash",
        )
    }
    _write_evidence(
        output_dir=tmp_path / "teacher",
        config=config,
        arrays=_arrays(config),
        lineage=lineage,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        compiled_model={"model_hash": _DIGEST},
    )
    archive = tmp_path / "teacher" / "corrective-teacher-corpus.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(ValueError):
        validate_impact_recovery_corrective_teacher(tmp_path / "teacher" / "teacher-report.json")


def test_corrective_teacher_config_rejects_hardware_authority() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCorrectiveTeacherConfig(hardware_authorized=True)

    smooth = ImpactRecoveryCorrectiveTeacherConfig(
        horizon_steps=80,
        action_chunk_steps=5,
        plan_knot_count=4,
    )
    assert smooth.action_chunk_count == 16
    assert smooth.search_knot_count == 4

    robust = ImpactRecoveryCorrectiveTeacherConfig(
        state_count=16,
        robust_variants_per_state=4,
        robust_worst_case_weight=0.5,
    )
    assert robust.state_count * robust.robust_variants_per_state == 64

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCorrectiveTeacherConfig(
            state_count=96,
            robust_variants_per_state=8,
        )


def test_robust_reset_keys_group_multiple_perturbations_of_each_failure() -> None:
    acquisition = np.asarray((2, 4, 6, 8), dtype=np.int32)
    elapsed = np.arange(10, dtype=np.float32)
    keys = _selected_reset_keys(
        rng=jax.random.PRNGKey(117),
        state_count=4,
        acquisition_indexes=acquisition,
        elapsed=elapsed,
        variants_per_state=3,
    )

    def selected_index(reset_key: jax.Array) -> jax.Array:
        _, index_rng, _, _ = jax.random.split(reset_key, 4)
        _, _, acquisition_rng = jax.random.split(index_rng, 3)
        offset = jax.random.randint(acquisition_rng, (), 0, acquisition.shape[0])
        return jnp.asarray(acquisition)[offset]

    grouped = np.asarray(jax.vmap(selected_index)(keys), dtype=np.int32).reshape((4, 3))
    assert np.all(grouped == grouped[:, :1])
    assert np.unique(grouped[:, 0]).size == 4


def test_robust_variant_reduction_blends_mean_and_worst_and_fails_closed() -> None:
    config = ImpactRecoveryCorrectiveTeacherConfig(
        state_count=4,
        robust_variants_per_state=2,
        robust_worst_case_weight=0.5,
    )
    cost = jnp.arange(1.0, 9.0)
    effect = np.zeros((8, 6), dtype=np.float32)
    effect[[1, 5], 3:5] = 0.2
    effect[[3, 7], 3:5] = 0.4
    finite = jnp.asarray((True, True, True, True, True, True, True, False))
    success = jnp.asarray((True, True, True, True, True, True, True, False))
    streak = jnp.asarray((25.0, 24.0, 20.0, 19.0, 18.0, 17.0, 2.0, 1.0))

    (
        robust_cost,
        robust_effect,
        robust_finite,
        robust_success,
        robust_streak,
        stability_regression,
    ) = _aggregate_robust_variants(
        cost=cost,
        effect=jnp.asarray(effect),
        finite=finite,
        success=success,
        maximum_streak=streak,
        state_count=2,
        config=config,
    )

    np.testing.assert_allclose(np.asarray(robust_cost), ((2.5, 3.5), (6.5, 7.5)))
    assert np.asarray(robust_effect).shape == (2, 2, 6)
    assert np.asarray(robust_finite).tolist() == [[True, True], [True, False]]
    assert np.asarray(robust_success).tolist() == [[True, True], [True, False]]
    assert np.asarray(robust_streak).tolist() == [[20.0, 19.0], [2.0, 1.0]]
    np.testing.assert_allclose(
        np.asarray(stability_regression), ((0.0, 0.4), (0.0, 0.4)), atol=1.0e-6
    )


def test_robust_teacher_evidence_binds_successor_diagnostics(tmp_path: Path) -> None:
    config = ImpactRecoveryCorrectiveTeacherConfig(
        state_count=4,
        horizon_steps=10,
        robust_variants_per_state=2,
        robust_worst_case_weight=0.5,
        objective_mode="SUCCESSOR_STREAK",
        minimum_successor_success_fraction=0.75,
    )
    arrays = _arrays(config)
    arrays.update(
        maximum_stability_regression=np.zeros(4, dtype=np.float64),
        baseline_success=np.asarray((False, False, False, False)),
        teacher_success=np.asarray((True, False, True, False)),
        baseline_maximum_stable_streak=np.asarray((0.0, 1.0, 2.0, 3.0)),
        teacher_maximum_stable_streak=np.asarray((25.0, 8.0, 25.0, 9.0)),
    )
    diagnostics = {
        "robust_variants_per_state": 2,
        "robust_worst_case_weight": 0.5,
        "baseline_success_count": 0,
        "teacher_success_count": 2,
        "baseline_median_maximum_stable_streak": 1.5,
        "teacher_median_maximum_stable_streak": 17.0,
        "teacher_maximum_stable_streak": 25.0,
    }
    lineage = {
        name: _DIGEST
        for name in (
            "curriculum_manifest_hash",
            "curriculum_manifest_file_hash",
            "curriculum_archive_hash",
            "body_hash",
            "training_model_hash",
        )
    }
    report = _write_evidence(
        output_dir=tmp_path / "robust",
        config=config,
        arrays=arrays,
        lineage=lineage,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        compiled_model={"model_hash": _DIGEST},
        objective_diagnostics=diagnostics,
    )
    assert report["objective_diagnostics"] == diagnostics
    assert report["successor_success_fraction"] == 0.5
    assert report["supervised_warm_start_eligible"] is False

    report_path = tmp_path / "robust" / "teacher-report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["objective_diagnostics"]["teacher_success_count"] = 3
    payload.pop("report_hash")
    payload["report_hash"] = hash_json(payload)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_corrective_teacher(report_path)
