"""Causal prefix and outcomes for paired post-contact relay studies."""

import numpy as np


def untouched_prefix_length(force: np.ndarray) -> tuple[int, bool]:
    if (
        force.ndim != 4
        or force.shape[1:] != (10, 2, 3)
        or len(force) < 1
        or not np.isfinite(force).all()
        or (force < 0).any()
    ):
        raise ValueError("finite actual native 10-step ball forces required")
    touched = np.flatnonzero((force[:, :, 1, :] > 1).any(axis=(1, 2)))
    return (int(touched[0]) + 1, True) if len(touched) else (len(force), False)


def equal_prefix(left: dict[str, np.ndarray], right: dict[str, np.ndarray], count: int) -> bool:
    if type(count) is not int or count < 1 or set(left) != set(right):
        raise ValueError("positive aligned trace and same fields required")
    return all(
        left[k].ndim > 0
        and right[k].ndim > 0
        and len(left[k]) >= count
        and len(right[k]) >= count
        and np.array_equal(left[k][:count], right[k][:count])
        for k in left
    )


def pair_metrics(baseline: dict, candidate: dict) -> dict:
    fields = (
        "handoff",
        "body_failure",
        "A_failed",
        "B_failed",
        "outgoing_foot_strike",
        "return_0p1m_and_healthy",
    )
    if any(type(x.get(k)) is not bool for x in (baseline, candidate) for k in fields):
        raise ValueError("actual boolean outcomes required")
    return {
        name: bool(not baseline[key] and candidate[key])
        if positive
        else bool(baseline[key] and not candidate[key])
        for name, key, positive in (
            ("handoff_rescue", "handoff", True),
            ("handoff_regression", "handoff", False),
            ("body_failure_rescue", "body_failure", False),
            ("body_failure_regression", "body_failure", True),
            ("precise_rescue", "return_0p1m_and_healthy", True),
            ("precise_regression", "return_0p1m_and_healthy", False),
        )
    }
