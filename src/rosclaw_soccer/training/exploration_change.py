"""Validate an intended exploration increase against the actual checkpoint.

This reports effective Gaussian scale changes, not observed task improvement.
It neither mutates a model nor counts configuration changes as learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExplorationIncrease:
    requested_log_std: float
    minimum_log_std: float
    maximum_log_std: float
    previous_effective_log_std: tuple[float, ...]
    resulting_effective_log_std: float
    standard_deviation_multipliers: tuple[float, ...]


def plan_exploration_increase(
    previous_log_std: NDArray[np.floating],
    *,
    requested_log_std: float,
    minimum_log_std: float,
    maximum_log_std: float,
) -> ExplorationIncrease:
    """Reject no-op/decreasing changes after applying the sampler's clipping.

    The caller supplies the checkpoint's actual logstd and the actual sampler
    bounds, applies the planned change explicitly, and verifies sampled logp.
    A report alone cannot establish that a collector uses those parameters.
    """
    if (
        not isinstance(previous_log_std, np.ndarray)
        or previous_log_std.ndim != 1
        or not 1 <= previous_log_std.size <= 4096
        or previous_log_std.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(previous_log_std).all()
        or np.any(np.abs(previous_log_std) > 20)
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (requested_log_std, minimum_log_std, maximum_log_std)
        )
        or not -20 <= minimum_log_std < maximum_log_std <= 2
        or not -20 <= requested_log_std <= 2
    ):
        raise ValueError("finite bounded checkpoint logstd and explicit sampler bounds required")
    previous = np.clip(previous_log_std.astype(np.float64), minimum_log_std, maximum_log_std)
    resulting = float(np.clip(requested_log_std, minimum_log_std, maximum_log_std))
    change = resulting - previous
    if np.any(change < 0) or not np.any(change > 0):
        raise ValueError("requested exploration increase is a no-op or decreases a component")
    return ExplorationIncrease(
        requested_log_std=float(requested_log_std),
        minimum_log_std=float(minimum_log_std),
        maximum_log_std=float(maximum_log_std),
        previous_effective_log_std=tuple(float(v) for v in previous),
        resulting_effective_log_std=resulting,
        standard_deviation_multipliers=tuple(float(v) for v in np.exp(change)),
    )
