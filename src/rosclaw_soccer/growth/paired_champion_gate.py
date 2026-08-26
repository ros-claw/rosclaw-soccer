"""Domain-neutral parent/candidate gate for evidence-backed self evolution.

Passing a safety or baseline exam qualifies a candidate, but it does not prove
that the candidate should erase the current champion.  This module performs a
second, content-bound Pareto audit on an identical sealed scenario suite.  A
safe but non-dominating candidate remains a useful lineage branch.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_METRIC = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class ChampionMetricSpec:
    """One auditable objective and its replacement tolerance."""

    name: str
    direction: Literal["MAXIMIZE", "MINIMIZE"]
    maximum_regression: float = 0.0
    minimum_improvement: float = 0.0
    hard_lower_bound: float | None = None
    hard_upper_bound: float | None = None
    schema_version: str = "rosclaw_soccer.champion_metric_spec.v1"

    def __post_init__(self) -> None:
        if not _METRIC.fullmatch(self.name):
            raise ValueError("champion metric name is invalid")
        if self.direction not in ("MAXIMIZE", "MINIMIZE"):
            raise ValueError("champion metric direction is invalid")
        values = (self.maximum_regression, self.minimum_improvement)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("champion metric tolerances must be finite and nonnegative")
        for bound in (self.hard_lower_bound, self.hard_upper_bound):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("champion metric hard bounds must be finite")
        if (
            self.hard_lower_bound is not None
            and self.hard_upper_bound is not None
            and self.hard_lower_bound > self.hard_upper_bound
        ):
            raise ValueError("champion metric hard bounds are reversed")


@dataclass(frozen=True)
class ChampionSnapshot:
    """Metrics for one immutable artifact on one sealed paired suite."""

    artifact_hash: str
    parent_artifact_hash: str | None
    scenario_suite_hash: str
    episode_count: int
    metrics: tuple[tuple[str, float], ...]
    qualified: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.champion_snapshot.v1"

    def __post_init__(self) -> None:
        for value in (self.artifact_hash, self.scenario_suite_hash):
            if not _HASH.fullmatch(value):
                raise ValueError("champion snapshot hash is invalid")
        if self.parent_artifact_hash is not None and not _HASH.fullmatch(self.parent_artifact_hash):
            raise ValueError("champion parent hash is invalid")
        if not 8 <= self.episode_count <= 1_000_000:
            raise ValueError("champion snapshot episode count is invalid")
        names = tuple(name for name, _ in self.metrics)
        if not self.metrics or len(set(names)) != len(names):
            raise ValueError("champion snapshot metrics must be non-empty and unique")
        if any(not _METRIC.fullmatch(name) for name in names):
            raise ValueError("champion snapshot metric name is invalid")
        if any(not math.isfinite(value) for _, value in self.metrics):
            raise ValueError("champion snapshot metrics must be finite")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("champion snapshot must remain SIM_ONLY")

    @property
    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)


@dataclass(frozen=True)
class PairedChampionDecision:
    replace_champion: bool
    status: Literal["REPLACE_CHAMPION", "RETAIN_PARENT_ARCHIVE_CANDIDATE"]
    reasons: tuple[str, ...]
    metric_deltas: tuple[tuple[str, float], ...]
    parent_artifact_hash: str
    candidate_artifact_hash: str
    scenario_suite_hash: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.paired_champion_decision.v1"

    @property
    def decision_hash(self) -> str:
        return str(hash_json(asdict(self)))


def evaluate_paired_champion(
    *,
    parent: ChampionSnapshot,
    candidate: ChampionSnapshot,
    metrics: tuple[ChampionMetricSpec, ...],
    minimum_improved_metrics: int = 1,
) -> PairedChampionDecision:
    """Choose replacement only when a qualified child Pareto-dominates."""

    if candidate.parent_artifact_hash != parent.artifact_hash:
        raise ValueError("champion candidate is not bound to its evaluated parent")
    if candidate.scenario_suite_hash != parent.scenario_suite_hash:
        raise ValueError("champion comparison must use an identical scenario suite")
    if candidate.episode_count != parent.episode_count:
        raise ValueError("champion comparison must use an identical episode count")
    if not metrics or len({item.name for item in metrics}) != len(metrics):
        raise ValueError("champion gate metric specs must be non-empty and unique")
    if not 1 <= minimum_improved_metrics <= len(metrics):
        raise ValueError("champion gate improved-metric floor is invalid")
    expected = {item.name for item in metrics}
    if set(parent.metric_map) != expected or set(candidate.metric_map) != expected:
        raise ValueError("champion snapshot metrics do not match the gate")

    reasons: list[str] = []
    if not parent.qualified:
        reasons.append("parent_not_qualified")
    if not candidate.qualified:
        reasons.append("candidate_not_qualified")
    deltas: list[tuple[str, float]] = []
    improved = 0
    for spec in metrics:
        parent_value = parent.metric_map[spec.name]
        candidate_value = candidate.metric_map[spec.name]
        signed_delta = (
            candidate_value - parent_value
            if spec.direction == "MAXIMIZE"
            else parent_value - candidate_value
        )
        deltas.append((spec.name, signed_delta))
        if signed_delta < -spec.maximum_regression:
            reasons.append(f"{spec.name}_regression_exceeds_budget")
        if signed_delta >= spec.minimum_improvement and (
            signed_delta > 0.0 or spec.minimum_improvement == 0.0
        ):
            improved += 1
        if spec.hard_lower_bound is not None and candidate_value < spec.hard_lower_bound:
            reasons.append(f"{spec.name}_below_hard_floor")
        if spec.hard_upper_bound is not None and candidate_value > spec.hard_upper_bound:
            reasons.append(f"{spec.name}_above_hard_ceiling")
    if improved < minimum_improved_metrics:
        reasons.append("insufficient_improved_metrics")
    unique_reasons = tuple(dict.fromkeys(reasons))
    replace = not unique_reasons
    return PairedChampionDecision(
        replace_champion=replace,
        status=("REPLACE_CHAMPION" if replace else "RETAIN_PARENT_ARCHIVE_CANDIDATE"),
        reasons=unique_reasons,
        metric_deltas=tuple(deltas),
        parent_artifact_hash=parent.artifact_hash,
        candidate_artifact_hash=candidate.artifact_hash,
        scenario_suite_hash=parent.scenario_suite_hash,
    )


def decision_payload(
    decision: PairedChampionDecision,
    *,
    parent: ChampionSnapshot,
    candidate: ChampionSnapshot,
    metrics: tuple[ChampionMetricSpec, ...],
) -> dict[str, Any]:
    """Return a content-addressable evidence payload without writing state."""

    payload: dict[str, Any] = {
        "decision": asdict(decision),
        "decision_hash": decision.decision_hash,
        "parent": asdict(parent),
        "candidate": asdict(candidate),
        "metric_specs": [asdict(item) for item in metrics],
        "selection_semantics": "SAFE_BRANCH_MAY_QUALIFY_WITHOUT_REPLACING_CHAMPION",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "schema_version": "rosclaw_soccer.paired_champion_evidence.v1",
    }
    payload["report_hash"] = hash_json(payload)
    return payload


__all__ = [
    "ChampionMetricSpec",
    "ChampionSnapshot",
    "PairedChampionDecision",
    "decision_payload",
    "evaluate_paired_champion",
]
