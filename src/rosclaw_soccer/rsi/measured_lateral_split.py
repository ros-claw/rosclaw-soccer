"""Fit a conservative one-dimensional skill-router boundary from measured outcomes."""

from __future__ import annotations

import math


def fit_measured_lateral_split(
    beneficial_features: tuple[float, ...], harmful_features: tuple[float, ...]
) -> float:
    """Midpoint of observed benefit/risk with a positive separation margin.

    Labels must come from paired, safe physical rollouts; this function has no
    access to course IDs, future observations, robot handles, or actuators.
    """
    if (
        type(beneficial_features) is not tuple
        or type(harmful_features) is not tuple
        or not beneficial_features
        or not harmful_features
        or any(
            type(value) is not float or not math.isfinite(value) or not 0.70 <= value <= 0.85
            for value in beneficial_features + harmful_features
        )
    ):
        raise ValueError("finite measured benefit and harm features required")
    benefit_max = max(beneficial_features)
    harm_min = min(harmful_features)
    if harm_min - benefit_max < 0.003:
        raise ValueError("skill benefit and harm have no robust measured gap")
    threshold = (benefit_max + harm_min) / 2.0
    if not 0.760 <= threshold <= 0.775:
        raise ValueError("learned skill boundary exceeds SIM_ONLY lateral envelope")
    return threshold
