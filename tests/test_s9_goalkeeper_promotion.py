from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoverageTrial,
    aggregate_coverage_time,
)
from rosclaw_soccer.skills.goalkeeper_v2.promotion import evaluate_goalkeeper_promotion


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _trial(deadline: float, seed: int, *, candidate: bool) -> GoalkeeperCoverageTrial:
    parent_save = seed == 0
    return GoalkeeperCoverageTrial(
        scenario_hash=_hash(f"scenario:{deadline}:{seed}"),
        frozen_shooter_policy_hash=_hash("frozen-shooter"),
        numerical_contract_hash=_hash("cpu"),
        seed=seed,
        target_region="upper_left",
        target_y_m=0.9,
        target_z_m=1.6,
        deadline_sec=deadline,
        observed_flight_start_sec=0.02,
        first_action_sec=0.06 if candidate else 0.10,
        ball_contact=candidate or parent_save,
        true_save=candidate or parent_save,
        intercept_error_m=0.02 if candidate else 0.20,
        recovery_time_sec=0.4 if candidate else 0.8,
        second_save_success=candidate,
        idle_ratio=0.01,
        human_motion_score=0.8 if candidate else 0.5,
        safety_cost=0.0,
        actor_observation_contract_hash=_hash("actor-contract"),
        evaluated_actor_policy_hash=_hash("candidate" if candidate else "parent"),
    )


def _suite(*, candidate: bool) -> tuple[GoalkeeperCoverageTrial, ...]:
    return tuple(
        _trial(deadline, seed, candidate=candidate)
        for deadline in (1.0, 0.8, 0.6, 0.5, 0.4)
        for seed in range(5)
    )


def test_promotion_passes_only_complete_matched_evidence() -> None:
    parent = _suite(candidate=False)
    candidate = _suite(candidate=True)
    parent_report = aggregate_coverage_time(parent, strict_replay=True, sealed_holdout=True)
    candidate_report = aggregate_coverage_time(candidate, strict_replay=True, sealed_holdout=True)

    decision = evaluate_goalkeeper_promotion(
        parent_report=parent_report,
        parent_trials=parent,
        candidate_report=candidate_report,
        candidate_trials=candidate,
        historical_mean_regression=0.01,
    )

    assert decision.verdict == "PROMOTED"
    assert not decision.failed_metrics


def test_promotion_rejects_safety_and_missing_evidence() -> None:
    parent = _suite(candidate=False)
    candidate = _suite(candidate=True)
    candidate = (
        replace(
            candidate[0],
            safety_cost=1.0,
            safety_failure_codes=("GOALKEEPER_PELVIS_BELOW_0_60_M",),
        ),
        *candidate[1:],
    )
    parent_report = aggregate_coverage_time(parent, strict_replay=True, sealed_holdout=False)
    candidate_report = aggregate_coverage_time(candidate, strict_replay=True, sealed_holdout=False)

    decision = evaluate_goalkeeper_promotion(
        parent_report=parent_report,
        parent_trials=parent,
        candidate_report=candidate_report,
        candidate_trials=candidate,
        historical_mean_regression=None,
    )

    assert decision.verdict == "REJECTED"
    assert {"safety_regression", "historical_keeper_regression", "sealed_holdout"} <= set(
        decision.failed_metrics
    )


def test_promotion_rejects_unmatched_exam() -> None:
    parent = _suite(candidate=False)
    candidate = _suite(candidate=True)
    candidate = (replace(candidate[0], scenario_hash=_hash("drift")), *candidate[1:])
    parent_report = aggregate_coverage_time(parent, strict_replay=True, sealed_holdout=True)
    candidate_report = aggregate_coverage_time(candidate, strict_replay=True, sealed_holdout=True)

    with pytest.raises(ValueError, match="scenario_suite_hash"):
        evaluate_goalkeeper_promotion(
            parent_report=parent_report,
            parent_trials=parent,
            candidate_report=candidate_report,
            candidate_trials=candidate,
            historical_mean_regression=0.0,
        )
