from __future__ import annotations

from rosclaw.continual.residual_adaptation import ResidualAdaptationContract

from rosclaw_soccer.evidence.opentrack_promotion import (
    evaluate_opentrack_residual_candidate,
)


def _digest(label: str) -> str:
    return "sha256:" + (label.encode().hex() + "0" * 64)[:64]


def _contract() -> ResidualAdaptationContract:
    return ResidualAdaptationContract(
        run_id="run-1",
        backend_contract_hash=_digest("backend"),
        parent_artifact_hash=_digest("parent"),
        body_hash=_digest("body"),
        rehearsal_dataset_hash=_digest("rehearsal"),
        acquisition_dataset_hash=_digest("acquisition"),
        frozen_parameter_selectors=("hidden",),
        trainable_parameter_selectors=("adapter",),
        device_ids=(0, 1, 2, 3),
        maximum_world_steps=100,
        policy_learning_rate=1e-5,
        rehearsal_fraction=0.6,
        acquisition_fraction=0.4,
        maximum_residual_output_rms=0.05,
    )


def _report(*, parent: bool) -> dict[str, object]:
    reports = []
    for index in range(8):
        reports.append(
            {
                "episode_id": f"episode-{index}",
                "suite_id": "retention" if index < 4 else "acquisition",
                "critical": index < 4,
                "fell": bool(index >= (7 if parent else 8)),
                "joint_squared_error_sum": 1.0 if parent else 0.9,
                "joint_error_count": 10,
                "keypoint_squared_error_sum": 1.0 if parent else 0.9,
                "keypoint_error_count": 10,
                "joint_jerk_squared_sum": 1.0 if parent else 0.9,
                "joint_jerk_count": 10,
                "minimum_pelvis_height_m": 0.70 if parent else 0.71,
                "saturated_control_steps": 1 if parent else 0,
                "control_steps": 10,
            }
        )
    return {
        "plan_hash": _digest("plan"),
        "policy_hash": _digest("parent" if parent else "candidate"),
        "reference_policy_hash": _digest("parent"),
        "report_hash": _digest("parent-report" if parent else "candidate-report"),
        "episode_reports": reports,
        "suite_summary": {
            "retention": {"fall_count": 0, "success_rate": 1.0, "recovery_rate": 1.0},
            "acquisition": {
                "fall_count": 1 if parent else 0,
                "success_rate": 0.5 if parent else 0.75,
                "recovery_rate": 0.5 if parent else 0.75,
            },
        },
        "residual_output_rms": 0.02,
        "residual_scale": 0.25,
        "passed": not parent,
        "reasons": [] if not parent else ["tracking_success_below_floor"],
    }


def _isolation() -> dict[str, object]:
    return {
        "frozen_base_unchanged": True,
        "frozen_base_hash_before": _digest("frozen"),
        "frozen_base_hash_after": _digest("frozen"),
        "examined_frozen_parameter_count": 100,
        "examined_trainable_parameter_count": 20,
        "candidate_world_steps": 100,
        "maximum_frozen_parameter_drift": 0.0,
    }


def test_matched_candidate_promotes_only_after_absolute_physics_pass() -> None:
    decision = evaluate_opentrack_residual_candidate(
        contract=_contract(),
        parent_report=_report(parent=True),
        candidate_report=_report(parent=False),
        isolation_report=_isolation(),
    )

    assert decision.verdict == "PROMOTED"
    assert decision.parameter_isolation_passed


def test_relative_gain_is_not_misreported_as_champion() -> None:
    candidate = _report(parent=False)
    candidate["passed"] = False
    candidate["reasons"] = ["foot_slip_above_ceiling"]
    decision = evaluate_opentrack_residual_candidate(
        contract=_contract(),
        parent_report=_report(parent=True),
        candidate_report=candidate,
        isolation_report=_isolation(),
    )

    assert decision.verdict == "DEVELOPMENT_ADVANCE_NOT_PROMOTED"
    assert "foot_slip_above_ceiling" in decision.reasons


def test_critical_retention_fall_rejects_candidate() -> None:
    candidate = _report(parent=False)
    candidate["episode_reports"][0]["fell"] = True  # type: ignore[index]
    decision = evaluate_opentrack_residual_candidate(
        contract=_contract(),
        parent_report=_report(parent=True),
        candidate_report=candidate,
        isolation_report=_isolation(),
    )

    assert decision.verdict == "REJECTED"
    assert decision.critical_safety_regressions == 1


def test_training_over_sealed_budget_rejects_candidate() -> None:
    isolation = _isolation()
    isolation["candidate_world_steps"] = 101
    decision = evaluate_opentrack_residual_candidate(
        contract=_contract(),
        parent_report=_report(parent=True),
        candidate_report=_report(parent=False),
        isolation_report=isolation,
    )

    assert decision.verdict == "REJECTED"
    assert "training_steps_above_sealed_ceiling" in decision.reasons
