"""Fail-closed individual promotion gate for Goalkeeper V2.

The gate compares a candidate and its frozen parent on the same numerical
CPU-MuJoCo exam.  Missing recovery, second-save, human-motion, historical, or
sealed-holdout evidence is a rejection, never an implicit pass.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoverageTimeReport,
    GoalkeeperCoverageTrial,
)


@dataclass(frozen=True)
class GoalkeeperPromotionThresholds:
    median_reaction_reduction: float = 0.30
    p90_reaction_reduction: float = 0.20
    coverage_relative_gain: float = 0.30
    contact_rate_gain: float = 0.20
    true_save_rate_gain: float = 0.15
    recovery_time_reduction: float = 0.30
    minimum_second_save_rate: float = 0.01
    minimum_human_motion_gain: float = 0.05
    maximum_historical_mean_regression: float = 0.03
    schema_version: str = "rosclaw_soccer.goalkeeper_promotion_thresholds.v1"

    def __post_init__(self) -> None:
        values = tuple(value for key, value in asdict(self).items() if key != "schema_version")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("goalkeeper promotion thresholds must be in [0, 1]")

    @property
    def thresholds_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class GoalkeeperGateMetric:
    name: str
    parent_value: float | bool | None
    candidate_value: float | bool | None
    required_change: str
    passed: bool
    reason: str
    schema_version: str = "rosclaw_soccer.goalkeeper_gate_metric.v1"


@dataclass(frozen=True)
class GoalkeeperPromotionDecision:
    parent_policy_hash: str
    candidate_policy_hash: str
    scenario_suite_hash: str
    numerical_contract_hash: str
    thresholds_hash: str
    metrics: tuple[GoalkeeperGateMetric, ...]
    verdict: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_promotion_decision.v1"

    def __post_init__(self) -> None:
        for value in (
            self.parent_policy_hash,
            self.candidate_policy_hash,
            self.scenario_suite_hash,
            self.numerical_contract_hash,
            self.thresholds_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("goalkeeper promotion identities must be content hashes")
        expected = "PROMOTED" if all(metric.passed for metric in self.metrics) else "REJECTED"
        if self.verdict != expected:
            raise ValueError("goalkeeper promotion verdict disagrees with its metrics")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("goalkeeper promotion cannot authorize hardware")

    @property
    def failed_metrics(self) -> tuple[str, ...]:
        return tuple(metric.name for metric in self.metrics if not metric.passed)

    @property
    def decision_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["failed_metrics"] = list(self.failed_metrics)
        if include_hash:
            value["decision_hash"] = self.decision_hash
        return value


def evaluate_goalkeeper_promotion(
    *,
    parent_report: GoalkeeperCoverageTimeReport,
    parent_trials: tuple[GoalkeeperCoverageTrial, ...],
    candidate_report: GoalkeeperCoverageTimeReport,
    candidate_trials: tuple[GoalkeeperCoverageTrial, ...],
    historical_mean_regression: float | None,
    thresholds: GoalkeeperPromotionThresholds | None = None,
) -> GoalkeeperPromotionDecision:
    """Compare one candidate to its frozen parent and reject incomplete proof."""

    active = thresholds or GoalkeeperPromotionThresholds()
    _require_matched_exam(parent_report, parent_trials, candidate_report, candidate_trials)
    parent_reaction = _values(parent_trials, "reaction_latency_sec")
    candidate_reaction = _values(candidate_trials, "reaction_latency_sec")
    parent_recovery = _values(parent_trials, "recovery_time_sec")
    candidate_recovery = _values(candidate_trials, "recovery_time_sec")
    parent_human = _values(parent_trials, "human_motion_score")
    candidate_human = _values(candidate_trials, "human_motion_score")
    parent_coverage = float(np.mean([trial.covered for trial in parent_trials]))
    candidate_coverage = float(np.mean([trial.covered for trial in candidate_trials]))
    parent_contact = float(np.mean([trial.ball_contact for trial in parent_trials]))
    candidate_contact = float(np.mean([trial.ball_contact for trial in candidate_trials]))
    parent_save = float(np.mean([trial.true_save for trial in parent_trials]))
    candidate_save = float(np.mean([trial.true_save for trial in candidate_trials]))
    candidate_second = float(np.mean([trial.second_save_success for trial in candidate_trials]))
    parent_safety = max(trial.safety_cost for trial in parent_trials)
    candidate_safety = max(trial.safety_cost for trial in candidate_trials)

    metrics = (
        _reduction_metric(
            "median_reaction_latency",
            _quantile(parent_reaction, 0.50),
            _quantile(candidate_reaction, 0.50),
            active.median_reaction_reduction,
        ),
        _reduction_metric(
            "p90_reaction_latency",
            _quantile(parent_reaction, 0.90),
            _quantile(candidate_reaction, 0.90),
            active.p90_reaction_reduction,
        ),
        _relative_gain_metric(
            "fixed_deadline_coverage",
            parent_coverage,
            candidate_coverage,
            active.coverage_relative_gain,
        ),
        _absolute_gain_metric(
            "save_contact_rate",
            parent_contact,
            candidate_contact,
            active.contact_rate_gain,
        ),
        _absolute_gain_metric(
            "true_save_rate",
            parent_save,
            candidate_save,
            active.true_save_rate_gain,
        ),
        _reduction_metric(
            "recovery_time",
            None if not parent_recovery.size else float(np.mean(parent_recovery)),
            None if not candidate_recovery.size else float(np.mean(candidate_recovery)),
            active.recovery_time_reduction,
        ),
        GoalkeeperGateMetric(
            name="second_save_success",
            parent_value=float(np.mean([trial.second_save_success for trial in parent_trials])),
            candidate_value=candidate_second,
            required_change=f">={active.minimum_second_save_rate:.6f}",
            passed=candidate_second >= active.minimum_second_save_rate,
            reason=(
                "threshold_met"
                if candidate_second >= active.minimum_second_save_rate
                else "second_save_evidence_missing_or_zero"
            ),
        ),
        _absolute_gain_metric(
            "human_motion_prior_score",
            None if not parent_human.size else float(np.mean(parent_human)),
            None if not candidate_human.size else float(np.mean(candidate_human)),
            active.minimum_human_motion_gain,
        ),
        GoalkeeperGateMetric(
            name="safety_regression",
            parent_value=parent_safety,
            candidate_value=candidate_safety,
            required_change="candidate==0 and candidate<=parent",
            passed=candidate_safety == 0.0 and candidate_safety <= parent_safety,
            reason=(
                "threshold_met"
                if candidate_safety == 0.0 and candidate_safety <= parent_safety
                else "safety_regression"
            ),
        ),
        GoalkeeperGateMetric(
            name="historical_keeper_regression",
            parent_value=0.0,
            candidate_value=historical_mean_regression,
            required_change=f"<={active.maximum_historical_mean_regression:.6f}",
            passed=(
                historical_mean_regression is not None
                and historical_mean_regression <= active.maximum_historical_mean_regression
            ),
            reason=(
                "threshold_met"
                if historical_mean_regression is not None
                and historical_mean_regression <= active.maximum_historical_mean_regression
                else "historical_evidence_missing_or_regressed"
            ),
        ),
        GoalkeeperGateMetric(
            name="strict_replay",
            parent_value=parent_report.strict_replay,
            candidate_value=candidate_report.strict_replay,
            required_change="parent==true and candidate==true",
            passed=parent_report.strict_replay and candidate_report.strict_replay,
            reason=(
                "threshold_met"
                if parent_report.strict_replay and candidate_report.strict_replay
                else "strict_replay_missing"
            ),
        ),
        GoalkeeperGateMetric(
            name="sealed_holdout",
            parent_value=parent_report.sealed_holdout,
            candidate_value=candidate_report.sealed_holdout,
            required_change="candidate==true",
            passed=candidate_report.sealed_holdout,
            reason=("threshold_met" if candidate_report.sealed_holdout else "holdout_not_sealed"),
        ),
    )
    verdict = "PROMOTED" if all(metric.passed for metric in metrics) else "REJECTED"
    if parent_report.evaluated_actor_policy_hash is None:
        raise ValueError("parent report is missing its policy identity")
    if candidate_report.evaluated_actor_policy_hash is None:
        raise ValueError("candidate report is missing its policy identity")
    return GoalkeeperPromotionDecision(
        parent_policy_hash=parent_report.evaluated_actor_policy_hash,
        candidate_policy_hash=candidate_report.evaluated_actor_policy_hash,
        scenario_suite_hash=parent_report.scenario_suite_hash,
        numerical_contract_hash=parent_report.numerical_contract_hash,
        thresholds_hash=active.thresholds_hash,
        metrics=metrics,
        verdict=verdict,
    )


def _require_matched_exam(
    parent_report: GoalkeeperCoverageTimeReport,
    parent_trials: tuple[GoalkeeperCoverageTrial, ...],
    candidate_report: GoalkeeperCoverageTimeReport,
    candidate_trials: tuple[GoalkeeperCoverageTrial, ...],
) -> None:
    if not parent_trials or len(parent_trials) != len(candidate_trials):
        raise ValueError("goalkeeper promotion requires equal non-empty matched trials")
    report_identities = (
        "scenario_suite_hash",
        "frozen_shooter_policy_hash",
        "numerical_contract_hash",
        "actor_observation_contract_hash",
    )
    for identity in report_identities:
        if getattr(parent_report, identity) != getattr(candidate_report, identity):
            raise ValueError(f"goalkeeper promotion changed {identity}")
    parent_scenarios = sorted(trial.scenario_hash for trial in parent_trials)
    candidate_scenarios = sorted(trial.scenario_hash for trial in candidate_trials)
    if parent_scenarios != candidate_scenarios:
        raise ValueError("goalkeeper promotion changed matched scenario seeds")


def _values(trials: tuple[GoalkeeperCoverageTrial, ...], field: str) -> np.ndarray:
    values = [getattr(trial, field) for trial in trials]
    return np.asarray([value for value in values if value is not None], dtype=np.float64)


def _quantile(values: np.ndarray, quantile: float) -> float | None:
    return None if not values.size else float(np.quantile(values, quantile))


def _reduction_metric(
    name: str,
    parent: float | None,
    candidate: float | None,
    minimum_reduction: float,
) -> GoalkeeperGateMetric:
    passed = bool(
        parent is not None
        and candidate is not None
        and parent > 0.0
        and candidate <= parent * (1.0 - minimum_reduction)
    )
    return GoalkeeperGateMetric(
        name=name,
        parent_value=parent,
        candidate_value=candidate,
        required_change=f"relative_reduction>={minimum_reduction:.6f}",
        passed=passed,
        reason="threshold_met" if passed else "missing_or_insufficient_reduction",
    )


def _relative_gain_metric(
    name: str,
    parent: float,
    candidate: float,
    minimum_gain: float,
) -> GoalkeeperGateMetric:
    passed = parent > 0.0 and candidate >= parent * (1.0 + minimum_gain)
    return GoalkeeperGateMetric(
        name=name,
        parent_value=parent,
        candidate_value=candidate,
        required_change=f"relative_gain>={minimum_gain:.6f}",
        passed=passed,
        reason="threshold_met" if passed else "insufficient_relative_gain",
    )


def _absolute_gain_metric(
    name: str,
    parent: float | None,
    candidate: float | None,
    minimum_gain: float,
) -> GoalkeeperGateMetric:
    passed = bool(
        parent is not None and candidate is not None and candidate - parent >= minimum_gain
    )
    return GoalkeeperGateMetric(
        name=name,
        parent_value=parent,
        candidate_value=candidate,
        required_change=f"absolute_gain>={minimum_gain:.6f}",
        passed=passed,
        reason="threshold_met" if passed else "missing_or_insufficient_absolute_gain",
    )


__all__ = [
    "GoalkeeperGateMetric",
    "GoalkeeperPromotionDecision",
    "GoalkeeperPromotionThresholds",
    "evaluate_goalkeeper_promotion",
]
