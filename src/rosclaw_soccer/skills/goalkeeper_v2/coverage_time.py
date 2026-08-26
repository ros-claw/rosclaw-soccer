"""Coverage--time benchmark contracts for independently evaluated keepers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class GoalkeeperCoverageTrial:
    scenario_hash: str
    frozen_shooter_policy_hash: str
    numerical_contract_hash: str
    seed: int
    target_region: str
    target_y_m: float
    target_z_m: float
    deadline_sec: float
    observed_flight_start_sec: float | None
    first_action_sec: float | None
    ball_contact: bool
    true_save: bool
    intercept_error_m: float | None
    recovery_time_sec: float | None
    second_save_success: bool
    idle_ratio: float
    human_motion_score: float | None
    safety_cost: float
    actor_observation_contract_hash: str
    safety_failure_codes: tuple[str, ...] = ()
    minimum_pelvis_height_m: float | None = None
    evaluated_actor_policy_hash: str | None = None
    actor_reads_other_policy_state: bool = False
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_coverage_trial.v2"

    def __post_init__(self) -> None:
        for label, value in (
            ("scenario_hash", self.scenario_hash),
            ("frozen_shooter_policy_hash", self.frozen_shooter_policy_hash),
            ("numerical_contract_hash", self.numerical_contract_hash),
            ("actor_observation_contract_hash", self.actor_observation_contract_hash),
        ):
            if not value.startswith("sha256:"):
                raise ValueError(f"{label} must be a content hash")
        if self.evaluated_actor_policy_hash is not None and not (
            self.evaluated_actor_policy_hash.startswith("sha256:")
        ):
            raise ValueError("evaluated_actor_policy_hash must be a content hash")
        if not self.target_region.strip() or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("coverage trial requires a region and unsigned seed")
        values = (
            self.target_y_m,
            self.target_z_m,
            self.deadline_sec,
            self.idle_ratio,
            self.safety_cost,
        )
        if any(not math.isfinite(value) for value in values) or self.deadline_sec <= 0.0:
            raise ValueError("coverage trial metrics must be finite")
        if not 0.0 <= self.idle_ratio <= 1.0 or not 0.0 <= self.safety_cost <= 1.0:
            raise ValueError("coverage trial ratios must be in [0, 1]")
        for optional in (
            self.observed_flight_start_sec,
            self.first_action_sec,
            self.intercept_error_m,
            self.recovery_time_sec,
            self.human_motion_score,
            self.minimum_pelvis_height_m,
        ):
            if optional is not None and (not math.isfinite(optional) or optional < 0.0):
                raise ValueError("optional coverage metrics must be finite and non-negative")
        if self.actor_reads_other_policy_state:
            raise ValueError("keeper actor cannot read another policy's hidden state")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.physics_authority != "CPU_MUJOCO"
            or self.hardware_command_sent
            or self.pixels_used_for_scoring
        ):
            raise ValueError("coverage trials must remain numerical CPU MuJoCo evidence")
        if len(set(self.safety_failure_codes)) != len(self.safety_failure_codes) or any(
            not value.strip() for value in self.safety_failure_codes
        ):
            raise ValueError("safety failure codes must be non-empty and unique")
        if bool(self.safety_failure_codes) != bool(self.safety_cost):
            raise ValueError("safety cost and failure codes must agree")

    @property
    def reaction_latency_sec(self) -> float | None:
        if self.observed_flight_start_sec is None or self.first_action_sec is None:
            return None
        return max(0.0, self.first_action_sec - self.observed_flight_start_sec)

    @property
    def covered(self) -> bool:
        return self.true_save and self.safety_cost == 0.0

    @property
    def trial_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reaction_latency_sec"] = self.reaction_latency_sec
        value["covered"] = self.covered
        return value


@dataclass(frozen=True)
class GoalkeeperCoveragePoint:
    deadline_sec: float
    trial_count: int
    coverage_fraction: float
    save_contact_rate: float
    true_save_rate: float
    reaction_latency_p50_sec: float | None
    reaction_latency_p90_sec: float | None
    mean_intercept_error_m: float | None
    mean_recovery_time_sec: float | None
    second_save_rate: float
    mean_idle_ratio: float
    mean_human_motion_score: float | None
    maximum_safety_cost: float
    schema_version: str = "rosclaw_soccer.goalkeeper_coverage_point.v1"


@dataclass(frozen=True)
class GoalkeeperCoverageTimeReport:
    requested_deadlines_sec: tuple[float, ...]
    points: tuple[GoalkeeperCoveragePoint, ...]
    scenario_suite_hash: str
    frozen_shooter_policy_hash: str
    numerical_contract_hash: str
    actor_observation_contract_hash: str
    evaluated_actor_policy_hash: str | None
    strict_replay: bool
    sealed_holdout: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_coverage_time_report.v2"

    def __post_init__(self) -> None:
        if (
            tuple(sorted(self.requested_deadlines_sec, reverse=True))
            != self.requested_deadlines_sec
        ):
            raise ValueError("coverage deadlines must be ordered from easiest to hardest")
        if len(set(self.requested_deadlines_sec)) != len(self.requested_deadlines_sec):
            raise ValueError("coverage deadlines must be unique")
        if tuple(point.deadline_sec for point in self.points) != self.requested_deadlines_sec:
            raise ValueError("coverage points do not match requested deadlines")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("coverage report is SIM_ONLY")

    @property
    def report_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_deadlines_sec": list(self.requested_deadlines_sec),
            "points": [asdict(point) for point in self.points],
            "scenario_suite_hash": self.scenario_suite_hash,
            "frozen_shooter_policy_hash": self.frozen_shooter_policy_hash,
            "numerical_contract_hash": self.numerical_contract_hash,
            "actor_observation_contract_hash": self.actor_observation_contract_hash,
            "evaluated_actor_policy_hash": self.evaluated_actor_policy_hash,
            "strict_replay": self.strict_replay,
            "sealed_holdout": self.sealed_holdout,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }


def aggregate_coverage_time(
    trials: tuple[GoalkeeperCoverageTrial, ...],
    *,
    deadlines_sec: tuple[float, ...] = (1.0, 0.8, 0.6, 0.5, 0.4),
    strict_replay: bool,
    sealed_holdout: bool,
) -> GoalkeeperCoverageTimeReport:
    """Aggregate measured trials without extrapolating missing deadline cells."""

    if not trials:
        raise ValueError("coverage benchmark requires measured trials")
    identities = (
        "frozen_shooter_policy_hash",
        "numerical_contract_hash",
        "actor_observation_contract_hash",
        "evaluated_actor_policy_hash",
    )
    for identity in identities:
        if len({getattr(trial, identity) for trial in trials}) != 1:
            raise ValueError(f"coverage trials changed {identity}")
    points: list[GoalkeeperCoveragePoint] = []
    for deadline in deadlines_sec:
        selected = tuple(
            trial for trial in trials if math.isclose(trial.deadline_sec, deadline, abs_tol=1e-12)
        )
        if not selected:
            raise ValueError(f"coverage deadline {deadline} has no measured trials")
        reaction = np.asarray(
            [value for trial in selected if (value := trial.reaction_latency_sec) is not None],
            dtype=np.float64,
        )
        errors = np.asarray(
            [trial.intercept_error_m for trial in selected if trial.intercept_error_m is not None],
            dtype=np.float64,
        )
        recovery = np.asarray(
            [trial.recovery_time_sec for trial in selected if trial.recovery_time_sec is not None],
            dtype=np.float64,
        )
        human = np.asarray(
            [
                trial.human_motion_score
                for trial in selected
                if trial.human_motion_score is not None
            ],
            dtype=np.float64,
        )
        points.append(
            GoalkeeperCoveragePoint(
                deadline_sec=deadline,
                trial_count=len(selected),
                coverage_fraction=float(np.mean([trial.covered for trial in selected])),
                save_contact_rate=float(np.mean([trial.ball_contact for trial in selected])),
                true_save_rate=float(np.mean([trial.true_save for trial in selected])),
                reaction_latency_p50_sec=(
                    None if not reaction.size else float(np.quantile(reaction, 0.50))
                ),
                reaction_latency_p90_sec=(
                    None if not reaction.size else float(np.quantile(reaction, 0.90))
                ),
                mean_intercept_error_m=None if not errors.size else float(np.mean(errors)),
                mean_recovery_time_sec=(None if not recovery.size else float(np.mean(recovery))),
                second_save_rate=float(np.mean([trial.second_save_success for trial in selected])),
                mean_idle_ratio=float(np.mean([trial.idle_ratio for trial in selected])),
                mean_human_motion_score=None if not human.size else float(np.mean(human)),
                maximum_safety_cost=max(trial.safety_cost for trial in selected),
            )
        )
    # This is the immutable exam identity, not an outcome hash.  Binding the
    # whole trial would make matched parent/candidate reports look like they
    # used different exams merely because their actions differed.
    scenario_suite_hash = str(
        hash_json(
            {
                "scenario_hashes": sorted(trial.scenario_hash for trial in trials),
                "deadlines_sec": list(deadlines_sec),
            }
        )
    )
    return GoalkeeperCoverageTimeReport(
        requested_deadlines_sec=deadlines_sec,
        points=tuple(points),
        scenario_suite_hash=scenario_suite_hash,
        frozen_shooter_policy_hash=trials[0].frozen_shooter_policy_hash,
        numerical_contract_hash=trials[0].numerical_contract_hash,
        actor_observation_contract_hash=trials[0].actor_observation_contract_hash,
        evaluated_actor_policy_hash=trials[0].evaluated_actor_policy_hash,
        strict_replay=strict_replay,
        sealed_holdout=sealed_holdout,
    )


__all__ = [
    "GoalkeeperCoveragePoint",
    "GoalkeeperCoverageTimeReport",
    "GoalkeeperCoverageTrial",
    "aggregate_coverage_time",
]
