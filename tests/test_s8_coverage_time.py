from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoverageTrial,
    aggregate_coverage_time,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _trial(deadline: float, seed: int, save: bool) -> GoalkeeperCoverageTrial:
    return GoalkeeperCoverageTrial(
        scenario_hash=_hash(f"scenario:{deadline}:{seed}"),
        frozen_shooter_policy_hash=_hash("shooter.v12"),
        numerical_contract_hash=_hash("numerical.v1"),
        seed=seed,
        target_region="upper_left",
        target_y_m=1.1,
        target_z_m=1.7,
        deadline_sec=deadline,
        observed_flight_start_sec=0.10,
        first_action_sec=0.18,
        ball_contact=save,
        true_save=save,
        intercept_error_m=0.05 if save else 0.4,
        recovery_time_sec=0.6 if save else None,
        second_save_success=False,
        idle_ratio=0.03,
        human_motion_score=None,
        safety_cost=0.0,
        actor_observation_contract_hash=_hash("keeper.actor.v2"),
    )


def test_coverage_curve_requires_measured_trials_at_every_deadline() -> None:
    trials = tuple(
        _trial(deadline, index, index % 2 == 0)
        for deadline in (1.0, 0.8, 0.6, 0.5, 0.4)
        for index in (1, 2)
    )

    report = aggregate_coverage_time(
        trials,
        strict_replay=True,
        sealed_holdout=False,
    )

    assert len(report.points) == 5
    assert report.points[0].coverage_fraction == 0.5
    assert report.points[0].reaction_latency_p50_sec == pytest.approx(0.08)
    assert not report.sealed_holdout


def test_coverage_trial_rejects_hidden_policy_state_and_visual_scoring() -> None:
    trial = _trial(1.0, 1, True)
    with pytest.raises(ValueError, match="hidden state"):
        replace(trial, actor_reads_other_policy_state=True)
    with pytest.raises(ValueError, match="numerical CPU MuJoCo"):
        replace(trial, pixels_used_for_scoring=True)
    with pytest.raises(ValueError, match="must agree"):
        replace(trial, safety_cost=1.0)


def test_coverage_curve_rejects_opponent_drift_or_missing_deadlines() -> None:
    trials = tuple(_trial(deadline, 1, True) for deadline in (1.0, 0.8, 0.6, 0.5, 0.4))
    with pytest.raises(ValueError, match="frozen_shooter_policy_hash"):
        aggregate_coverage_time(
            (*trials[:-1], replace(trials[-1], frozen_shooter_policy_hash=_hash("weak"))),
            strict_replay=True,
            sealed_holdout=True,
        )
    with pytest.raises(ValueError, match="no measured trials"):
        aggregate_coverage_time(
            trials[:-1],
            strict_replay=True,
            sealed_holdout=True,
        )


def test_scenario_suite_identity_does_not_depend_on_policy_outcome() -> None:
    parent = tuple(_trial(deadline, 1, True) for deadline in (1.0, 0.8, 0.6, 0.5, 0.4))
    candidate = tuple(
        replace(
            trial,
            evaluated_actor_policy_hash=_hash("candidate"),
            true_save=False,
            ball_contact=False,
        )
        for trial in parent
    )

    parent_report = aggregate_coverage_time(parent, strict_replay=True, sealed_holdout=False)
    candidate_report = aggregate_coverage_time(candidate, strict_replay=True, sealed_holdout=False)

    assert parent_report.scenario_suite_hash == candidate_report.scenario_suite_hash
