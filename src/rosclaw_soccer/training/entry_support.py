"""Empirical entry coverage for candidate routing, never a safety certificate.

A supported query resembles the calibration examples under one standardized
distance. It does not prove task success, hardware safety or permission to
activate a policy. An unsupported entry needs a separately validated fallback
or new learning evidence, not silent extrapolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntrySupportAssessment:
    supported: bool
    squared_distance: float
    threshold: float


@dataclass(frozen=True)
class EntrySupportGate:
    centers: tuple[tuple[float, ...], ...]
    scales: tuple[float, ...]
    threshold: float

    def __post_init__(self) -> None:
        if (
            type(self.centers) is not tuple
            or not 2 <= len(self.centers) <= 512
            or type(self.scales) is not tuple
            or not 1 <= len(self.scales) <= 512
            or any(type(x) is not float or not math.isfinite(x) or x < 0.01 for x in self.scales)
            or any(
                type(row) is not tuple
                or len(row) != len(self.scales)
                or any(type(x) is not float or not math.isfinite(x) or abs(x) > 10 for x in row)
                for row in self.centers
            )
            or type(self.threshold) is not float
            or not math.isfinite(self.threshold)
            or not 0 < self.threshold <= 1000
        ):
            raise ValueError("immutable finite bounded entry calibration required")

    def assess(self, query: tuple[float, ...]) -> EntrySupportAssessment:
        if (
            type(query) is not tuple
            or len(query) != len(self.scales)
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 10 for x in query
            )
        ):
            raise ValueError("finite bounded entry query with matching features required")
        distance = min(
            math.fsum(
                ((x - y) / scale) ** 2 for x, y, scale in zip(query, row, self.scales, strict=True)
            )
            / len(self.scales)
            for row in self.centers
        )
        return EntrySupportAssessment(distance <= self.threshold, distance, self.threshold)


def calibrate_entry_support(
    examples: Any, *, quantile: float = 0.99, expansion: float = 4.0, scale_floor: float = 0.01
) -> EntrySupportGate:
    """Use leave-one-out nearest-example distances; no evaluation queries fitted.

    Immutable copies prevent callers mutating the gate by changing the input.
    Hyperparameters are explicit experimental choices, not calibrated risk
    probabilities. Duplicate examples receive a small nonzero radius floor.
    """
    import numpy as np

    values = np.asarray(examples)
    if (
        values.ndim != 2
        or not 2 <= values.shape[0] <= 512
        or not 1 <= values.shape[1] <= 512
        or values.shape[0] ** 2 * values.shape[1] > 16_777_216
        or values.dtype.kind not in "fi"
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 10)
        or any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in (quantile, expansion, scale_floor)
        )
        or not 0.9 <= quantile <= 1
        or not 1 <= expansion <= 8
        or not 0.01 <= scale_floor <= 1
    ):
        raise ValueError("finite bounded calibration matrix and parameters required")
    centers = values.astype(np.float64, copy=True)
    scales = np.maximum(centers.std(axis=0, ddof=1), scale_floor)
    distance = np.square((centers[:, None] - centers[None, :]) / scales).mean(axis=2)
    np.fill_diagonal(distance, np.inf)
    threshold = max(0.01, float(np.quantile(distance.min(axis=1), quantile)) * expansion)
    return EntrySupportGate(
        tuple(tuple(float(x) for x in row) for row in centers),
        tuple(float(x) for x in scales),
        threshold,
    )
