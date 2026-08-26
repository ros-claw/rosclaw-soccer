from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.skills.athlete_foundation.foundation_shootout import (
    FoundationEvaluation,
    FoundationMetrics,
    FoundationResultStatus,
    FoundationShootout,
    FoundationThresholds,
)

_HASH = "sha256:" + "a" * 64


def _metrics(*, success: float = 0.95, foot_slip: float = 0.05) -> FoundationMetrics:
    return FoundationMetrics(
        tracking_success_rate=success,
        joint_rmse_rad=0.15,
        keypoint_mpjpe_m=0.05,
        foot_slip_mps=foot_slip,
        minimum_pelvis_height_m=0.72,
        peak_torque_fraction=0.90,
        torque_saturation_rate=0.01,
        p95_root_angular_speed_rad_s=1.1,
        joint_jerk_rms_rad_s3=20.0,
        transition_error_rad=0.10,
        recovery_rate=0.90,
        finite_state=True,
    )


def _result(backend: str, metrics: FoundationMetrics) -> FoundationEvaluation:
    return FoundationEvaluation(
        backend_id=backend,
        backend_contract_hash=_HASH,
        motion_atlas_hash=_HASH,
        body_hash=_HASH,
        environment_hash=_HASH,
        seed_commitment_hash=_HASH,
        candidate_artifact_hash=_HASH,
        physics_backend="mujoco_cpu",
        episode_count=50,
        metrics=metrics,
        status=FoundationResultStatus.PHYSICS_QUALIFIED,
        reasons=(),
    )


def _shootout(*evaluations: FoundationEvaluation) -> FoundationShootout:
    return FoundationShootout(
        backend_contract_hashes=tuple((f"backend-{index}", _HASH) for index in range(4)),
        motion_atlas_hash=_HASH,
        body_hash=_HASH,
        environment_hash=_HASH,
        seed_commitment_hash=_HASH,
        thresholds=FoundationThresholds(),
        evaluations=evaluations,
    )


def test_shootout_has_no_winner_before_matched_physics() -> None:
    shootout = _shootout()

    assert shootout.winner_backend_id is None
    assert shootout.to_dict()["status"] == "PHYSICS_EVIDENCE_PENDING"


def test_shootout_selects_only_threshold_qualified_result() -> None:
    result = _result("backend-0", _metrics())
    shootout = _shootout(result)

    assert shootout.winner_backend_id == "backend-0"
    assert shootout.shootout_hash.startswith("sha256:")


def test_shootout_fails_closed_on_false_qualification_or_wrong_reasons() -> None:
    bad_metrics = _metrics(success=0.5, foot_slip=0.4)
    with pytest.raises(ValueError, match="despite threshold"):
        _shootout(_result("backend-0", bad_metrics))
    reasons = FoundationThresholds().reasons(bad_metrics)
    unqualified = replace(
        _result("backend-0", bad_metrics),
        status=FoundationResultStatus.PHYSICS_UNQUALIFIED,
        reasons=reasons,
    )
    assert _shootout(unqualified).winner_backend_id is None
    with pytest.raises(ValueError, match="do not match"):
        _shootout(replace(unqualified, reasons=("made_up",)))
