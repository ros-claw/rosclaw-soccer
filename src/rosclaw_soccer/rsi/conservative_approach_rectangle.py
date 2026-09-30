"""Learn a causal, abstaining SIM_ONLY approach gate from independent outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ApproachRectangle:
    """No motor authority; caller may request only a bounded simulation option."""

    x_max_m: float | None
    y_min_m: float | None
    support: int
    training_gain: float

    def choose(self, forward_gap_m: float, lateral_gap_m: float) -> bool:
        if not np.isfinite(forward_gap_m) or not np.isfinite(lateral_gap_m):
            raise ValueError("finite measured frame-zero ball context required")
        return bool(
            self.x_max_m is not None
            and self.y_min_m is not None
            and forward_gap_m <= self.x_max_m
            and self.y_min_m <= lateral_gap_m < 0.0
        )


def _cuts(values: np.ndarray[Any, Any]) -> tuple[float, ...]:
    unique = np.unique(values)
    cuts = np.r_[unique[0] - 1e-6, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1e-6]
    return tuple(float(value) for value in cuts)


def fit_approach_rectangle(
    context: np.ndarray[Any, Any],
    reward_gain: np.ndarray[Any, Any],
    clean_loss: np.ndarray[Any, Any],
    new_out_of_play: np.ndarray[Any, Any],
) -> ApproachRectangle:
    """Enumerate two-coordinate rules using training-only rewards and safety labels."""
    x = np.asarray(context, dtype=float)
    gain = np.asarray(reward_gain, dtype=float)
    clean = np.asarray(clean_loss, dtype=bool)
    out = np.asarray(new_out_of_play, dtype=bool)
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or len(x) < 12
        or gain.shape != (len(x),)
        or clean.shape != (len(x),)
        or out.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(gain).all()
    ):
        raise ValueError("finite independent causal paired courses required")
    negative = x[:, 1] < 0.0
    if np.count_nonzero(negative) < 3:
        return ApproachRectangle(None, None, 0, 0.0)
    best = ApproachRectangle(None, None, 0, 0.0)
    for x_max in _cuts(x[negative, 0]):
        for y_min in _cuts(x[negative, 1]):
            selected = negative & (x[:, 0] <= x_max) & (x[:, 1] >= y_min)
            support = int(np.count_nonzero(selected))
            if support < 3 or np.any(clean[selected]) or np.any(out[selected]):
                continue
            total_gain = float(np.sum(gain[selected]))
            if total_gain < 1.0:
                continue
            if (
                total_gain > best.training_gain + 1e-9
                or (
                    abs(total_gain - best.training_gain) <= 1e-9
                    and (best.support == 0 or support < best.support)
                )
                or (
                    abs(total_gain - best.training_gain) <= 1e-9
                    and support == best.support
                    and best.x_max_m is not None
                    and x_max < best.x_max_m
                )
            ):
                best = ApproachRectangle(x_max, y_min, support, total_gain)
    return best
