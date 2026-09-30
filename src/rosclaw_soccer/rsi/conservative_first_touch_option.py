"""Causal, bounded SIM_ONLY option-value model for G1 first-touch development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_COUNT = 9
ACTION_COUNT = 3
RIDGE_STRENGTH = 10.0
GAIN_MARGIN = 0.25
MAX_NEAREST_DISTANCE = 2.5


@dataclass(frozen=True)
class ConservativeOptionValue:
    """Learned option values; out-of-support states fall back to a fixed arm."""

    mean: np.ndarray[Any, Any]
    scale: np.ndarray[Any, Any]
    train_features: np.ndarray[Any, Any]
    coefficients: np.ndarray[Any, Any]
    fixed: int

    def choose(self, query: np.ndarray[Any, Any]) -> int:
        x = np.asarray(query, dtype=float)
        if x.shape != (FEATURE_COUNT,) or not np.isfinite(x).all():
            raise ValueError("finite nine-feature precontact context required")
        normalized = (x - self.mean) / self.scale
        if (
            float(np.min(np.linalg.norm(self.train_features - normalized, axis=1)))
            > MAX_NEAREST_DISTANCE
        ):
            return self.fixed
        value = np.r_[1.0, normalized] @ self.coefficients
        winner = int(np.argmax(value))
        return winner if value[winner] >= value[self.fixed] + GAIN_MARGIN else self.fixed


def fit_conservative_option_value(
    features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any]
) -> ConservativeOptionValue:
    """Fit only supplied independent audited courses; caller owns split integrity."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(rewards, dtype=float)
    if (
        x.ndim != 2
        or x.shape[0] < 24
        or x.shape[1] != FEATURE_COUNT
        or y.shape != (len(x), ACTION_COUNT)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("at least 24 independent audited contexts required")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 0.1)
    normalized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), normalized))
    penalty = np.diag([0.0] + [RIDGE_STRENGTH] * FEATURE_COUNT)
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    fixed = int(np.argmax(np.mean(y, axis=0)))
    return ConservativeOptionValue(mean, scale, normalized, coefficients, fixed)
