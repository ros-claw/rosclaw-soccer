"""Bounded local feasibility restoration; never treat linear contact models as truth."""

from typing import Any

import numpy as np


def task_violation(features: np.ndarray, *, force_scale: float) -> float:
    if (
        features.shape != (3,)
        or features.dtype != np.float64
        or not np.isfinite(features).all()
        or not np.isfinite(force_scale)
        or force_scale < 1
    ):
        raise ValueError(
            "finite physical/model features and fixed initial force normalization required"
        )
    forward, lateral, force = features
    return float(
        max(1.05 - forward, 0)
        + max(lateral - 0.27 * forward, 0)
        + max(-lateral - 0.27 * forward, 0)
        + max(force - 0.5, 0) / force_scale
    )


def restore_joint_feasibility(
    baseline: np.ndarray,
    jacobian: np.ndarray,
    center: np.ndarray,
    valid_axes: list[int],
    *,
    force_scale: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Take a small improvement step even when the complete task is not yet locally feasible.

    The caller MUST run a fresh full NN physical forecast and accept the new
    center only if its observed violation decreases with the SAME force scale.
    This function neither accepts a center nor changes a real/simulated body.
    """
    before = task_violation(baseline, force_scale=force_scale)
    if (
        jacobian.shape != (3, 29)
        or center.shape != (29,)
        or jacobian.dtype != np.float64
        or center.dtype != np.float64
        or not np.isfinite(jacobian).all()
        or not np.isfinite(center).all()
        or abs(center).max() > 0.06
        or sorted(set(valid_axes)) != valid_axes
        or any(type(i) is not int or not 0 <= i < 29 for i in valid_axes)
    ):
        raise ValueError(
            "finite local measured response and bounded named joint reference required"
        )
    report: dict[str, Any] = dict(
        schema="soccer.joint_task_feasibility_restoration.v1",
        baseline_features=baseline.tolist(),
        center=center.tolist(),
        response_jacobian=jacobian.tolist(),
        valid_axes=valid_axes,
        force_scale=force_scale,
        current_violation=before,
        maximum_step_axis_rad=0.03,
        maximum_step_l1_rad=0.06,
        maximum_reference_axis_rad=0.06,
        requires_new_complete_healthy_NN_physics=True,
        model_improvement_is_not_physical_improvement=True,
        hardware_authorized=False,
    )
    if not valid_axes:
        return None, dict(report, status="ABSTAIN_NO_COMPLETE_LOCAL_RESPONSE_PAIRS")
    from scipy.optimize import linprog  # type: ignore[import-untyped]

    f, lateral, n = baseline
    j = jacobian
    m = np.stack((-j[0], j[1] - 0.27 * j[0], -j[1] - 0.27 * j[0], j[2], -j[2]))
    # x=(29 signed local steps,29 absolute-step variables,4 feasibility slacks).
    a = np.zeros((64, 62))
    a[:5, :29] = m
    a[:4, 58:] = -np.eye(4)
    a[5:34, :29], a[5:34, 29:58] = np.eye(29), -np.eye(29)
    a[34:63, :29], a[34:63, 29:58] = -np.eye(29), -np.eye(29)
    a[63, 29:58] = 1
    rhs = np.r_[f - 1.05, 0.27 * f - lateral, 0.27 * f + lateral, 0.5 - n, n, np.zeros(58), 0.06]
    cap = float(np.nextafter(0.06, 0.0))
    bounds = [
        (max(-0.03, -cap - center[i]), min(0.03, cap - center[i])) if i in valid_axes else (0, 0)
        for i in range(29)
    ]
    if any(lo > hi for lo, hi in bounds):
        return None, dict(report, status="ABSTAIN_LOCAL_TRUST_INTERSECTION")
    solved = linprog(
        np.r_[np.zeros(29), np.full(29, 0.001 / 0.06), [1.0, 1.0, 1.0, 1.0 / force_scale]],
        A_ub=a,
        b_ub=rhs,
        bounds=bounds + [(0, 0.03)] * 29 + [(0, None)] * 4,
        method="highs",
    )
    if not solved.success or not np.isfinite(solved.x).all():
        return None, dict(report, status="ABSTAIN_RESTORATION_SOLVER")
    step = solved.x[:29]
    candidate = center + step
    if (
        (a @ solved.x - rhs > 1e-7).any()
        or abs(step).max() > 0.03000001
        or abs(step).sum() > 0.06000001
        or abs(candidate).max() > 0.06
    ):
        return None, dict(report, status="ABSTAIN_RESTORATION_TRUST_MISMATCH")
    predicted = baseline + j @ step
    after = task_violation(predicted, force_scale=force_scale)
    if after >= before - 1e-6:
        return None, dict(report, status="ABSTAIN_NO_LOCAL_MODEL_PROGRESS")
    return candidate, dict(
        report,
        status="UNVERIFIED_RESTORATION_CANDIDATE",
        step=step.tolist(),
        candidate=candidate.tolist(),
        predicted_features=predicted.tolist(),
        predicted_violation=after,
    )
