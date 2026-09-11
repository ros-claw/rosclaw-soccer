"""Outcome-labelled entry retrieval; never motor authority or a safety certificate.

Successful examples alone cannot distinguish familiar failures. This memory
uses both classes in the same standardized feature space and abstains on ties,
nearby failures, or extrapolation. Distances are not success probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.training.entry_support import EntrySupportGate, calibrate_entry_support


@dataclass(frozen=True)
class OutcomeEntryAssessment:
    suggested: bool
    success_distance: float
    failure_distance: float
    support_threshold: float
    reason: str


@dataclass(frozen=True)
class OutcomeEntryMemory:
    success: EntrySupportGate
    failure: EntrySupportGate
    separation_ratio: float = 0.5

    def __post_init__(self) -> None:
        if (
            not isinstance(self.success, EntrySupportGate)
            or not isinstance(self.failure, EntrySupportGate)
            or self.success.scales != self.failure.scales
            or self.success.threshold != self.failure.threshold
            or len(self.success.centers) + len(self.failure.centers) > 512
            or type(self.separation_ratio) is not float
            or not math.isfinite(self.separation_ratio)
            or not 0 < self.separation_ratio < 1
        ):
            raise ValueError("bounded outcome memories must share a feature metric")

    def assess(self, query: tuple[float, ...]) -> OutcomeEntryAssessment:
        positive = self.success.assess(query)
        negative = self.failure.assess(query)
        if not positive.supported:
            reason = "outside_success_support"
        elif positive.squared_distance >= self.separation_ratio * negative.squared_distance:
            reason = "near_failure_or_ambiguous"
        else:
            reason = "empirical_success_neighbor"
        return OutcomeEntryAssessment(
            reason == "empirical_success_neighbor",
            positive.squared_distance,
            negative.squared_distance,
            positive.threshold,
            reason,
        )


def fit_outcome_entry_memory(
    examples: Any,
    outcomes: Any,
    *,
    separation_ratio: float = 0.5,
    quantile: float = 0.99,
    expansion: float = 4.0,
    scale_floor: float = 0.01,
) -> OutcomeEntryMemory:
    """Fit immutable calibration examples with at least two of each outcome.

    Callers own feature semantics and train/exam separation. Do not include
    future outcomes, clocks, or sample IDs as runtime proprioceptive features.
    A suggestion still needs independently qualified initiation and execution.
    """
    import numpy as np

    values = np.asarray(examples)
    labels = np.asarray(outcomes)
    if (
        values.ndim != 2
        or labels.ndim != 1
        or len(labels) != len(values)
        or labels.dtype.kind != "b"
        or labels.sum() < 2
        or (~labels).sum() < 2
    ):
        raise ValueError("matched boolean outcomes with at least two of each class required")
    coverage = calibrate_entry_support(
        values, quantile=quantile, expansion=expansion, scale_floor=scale_floor
    )
    successes = tuple(row for row, label in zip(coverage.centers, labels, strict=True) if label)
    failures = tuple(row for row, label in zip(coverage.centers, labels, strict=True) if not label)
    return OutcomeEntryMemory(
        EntrySupportGate(successes, coverage.scales, coverage.threshold),
        EntrySupportGate(failures, coverage.scales, coverage.threshold),
        separation_ratio,
    )
