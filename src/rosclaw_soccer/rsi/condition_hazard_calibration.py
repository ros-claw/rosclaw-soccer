"""Calibrate per-episode contact warning thresholds; not a motor controller."""

import math

import numpy as np


def condition_hazard_summary(
    scores: np.ndarray, labels: np.ndarray, condition: np.ndarray
) -> list[dict]:
    """One maximum score and one outcome per independently identified condition."""
    if (
        not isinstance(scores, np.ndarray)
        or scores.ndim != 1
        or len(scores) == 0
        or not isinstance(labels, np.ndarray)
        or labels.shape != scores.shape
        or not isinstance(condition, np.ndarray)
        or condition.shape != scores.shape
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or not np.isin(labels, (0, 1, 2)).all()
        or condition.dtype.kind not in "iu"
    ):
        raise ValueError("finite probability and exact labeled condition identities required")
    rows = []
    for key in np.unique(condition):
        mask = condition == key
        rows.append(
            dict(
                condition=int(key),
                max_risk=float(scores[mask].max()),
                contains_nonfoot=bool((labels[mask] == 2).any()),
            )
        )
    return rows


def calibrate_negative_maxima(rows: list[dict], *, alpha: float) -> tuple[float, int]:
    """Conservative upper order statistic; insufficient negatives disables alarms.

    This is a calibration rule, not a population guarantee: robot conditions
    can be correlated and must not be treated as exchangeable without evidence.
    """
    if (
        not isinstance(rows, list)
        or not 0 < alpha < 1
        or not math.isfinite(alpha)
        or any(
            type(r.get("condition")) is not int
            or type(r.get("contains_nonfoot")) is not bool
            or type(r.get("max_risk")) not in (float, int)
            or not math.isfinite(r["max_risk"])
            or not 0 <= r["max_risk"] <= 1
            for r in rows
        )
        or len({r["condition"] for r in rows}) != len(rows)
    ):
        raise ValueError("unique finite condition-level calibration records required")
    negative = sorted(float(r["max_risk"]) for r in rows if not r["contains_nonfoot"])
    n = len(negative)
    rank = math.ceil((n + 1) * (1 - alpha))
    return (negative[rank - 1] if rank <= n else 1.0), n
