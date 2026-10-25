"""Fit a local action-to-ball response from native NN counterfactual probes.

These are observed simulation windows, including failures, not FK predictions
or neural actor updates. A coordinated numeric proposal ALWAYS needs a new
native closed-loop forecast and the eventual unchanged physical examination.
"""

from typing import Any

import numpy as np


def propose_contact_response(
    nominal: np.ndarray,
    limits: np.ndarray,
    parent_window: dict[str, Any],
    probes: list[tuple[np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if (
        nominal.shape != (29,)
        or limits.shape != (29, 2)
        or nominal.dtype != np.float64
        or limits.dtype != np.float64
        or not np.isfinite(nominal).all()
        or not np.isfinite(limits).all()
        or (limits[:, 0] >= limits[:, 1]).any()
    ):
        raise ValueError("finite actual nominal target and physical limits required")
    if not parent_window["quality_observed"]:
        return None, dict(status="ABSTAIN_NO_OBSERVED_PARENT_WINDOW", physically_verified=False)
    baseline = np.array([parent_window["forward_60_m"], parent_window["lateral_60_m"]], np.float64)
    if not np.isfinite(baseline).all():
        raise ValueError("finite observed parent ball displacement required")
    numerator, denominator = np.zeros((2, 12)), np.zeros(12)
    lower, upper = np.zeros(12), np.zeros(12)
    observed = []
    for target, window in probes:
        if target.shape != (29,) or target.dtype != np.float64 or not np.isfinite(target).all():
            raise ValueError("finite recorded native probe target required")
        changed = np.flatnonzero(target != nominal)
        if len(changed) != 1 or changed[0] >= 12 or np.max(abs(target - nominal)) > 0.030001:
            raise ValueError("declared single-axis bounded probe required")
        if not window["quality_observed"]:
            continue  # No zero-filled outcome for a censored or early-failed branch.
        axis = int(changed[0])
        if not limits[axis, 0] <= nominal[axis] <= limits[axis, 1]:
            continue  # No coordinated automatic repair of an inadmissible parent axis.
        value = np.array([window["forward_60_m"], window["lateral_60_m"]], np.float64)
        if not np.isfinite(value).all():
            raise ValueError("finite observed probe displacement required")
        delta = float(target[axis] - nominal[axis])
        numerator[:, axis] += delta * (value - baseline)
        denominator[axis] += delta * delta
        lower[axis] = min(lower[axis], delta)
        upper[axis] = max(upper[axis], delta)
        observed.append((axis, delta, value))
    active = np.flatnonzero(denominator > 0)
    if len(active) < 2:
        return None, dict(
            status="ABSTAIN_INSUFFICIENT_OBSERVED_AXES",
            observed_axes=active.tolist(),
            physically_verified=False,
        )
    response = numerator[:, active] / denominator[active]
    if not np.isfinite(response).all() or abs(response).max() > 1e4:
        return None, dict(status="ABSTAIN_ILL_SCALED_RESPONSE", physically_verified=False)
    desired_forward = max(1.2, float(baseline[0]))
    desired = np.array(
        [desired_forward, np.clip(baseline[1], -0.25 * desired_forward, 0.25 * desired_forward)]
    )
    # Standard bounded least squares; explicit .1 ridge chooses a small
    # coordinated increment rather than claiming the fitted response is exact.
    from scipy.optimize import lsq_linear  # type: ignore[import-untyped]

    matrix = np.vstack((response, np.sqrt(0.1) * np.eye(len(active))))
    target = np.r_[desired - baseline, np.zeros(len(active))]
    lo = np.maximum(lower[active], limits[active, 0] - nominal[active])
    hi = np.minimum(upper[active], limits[active, 1] - nominal[active])
    if np.any(lo >= hi):
        return None, dict(status="ABSTAIN_RESPONSE_BOUND_INTERSECTION", physically_verified=False)
    solved = lsq_linear(matrix, target, bounds=(lo, hi), method="bvls", tol=1e-10, max_iter=300)
    if not solved.success or not np.isfinite(solved.x).all():
        return None, dict(status="ABSTAIN_RESPONSE_SOLVER", physically_verified=False)
    proposal = nominal.copy()
    proposal[active] += solved.x
    fit_errors = [
        float(np.linalg.norm((numerator[:, axis] / denominator[axis]) * delta - (value - baseline)))
        for axis, delta, value in observed
    ]
    return proposal, dict(
        status="LOCAL_NATIVE_RESPONSE_NUMERIC_PROPOSAL",
        observed_axes=active.tolist(),
        observed_probe_count=len(observed),
        response_m_per_target_rad=response.tolist(),
        observed_parent_window_m=baseline.tolist(),
        desired_window_m=desired.tolist(),
        linear_predicted_window_m=(baseline + response @ solved.x).tolist(),
        maximum_probe_fit_error_m=max(fit_errors, default=0.0),
        proposed_target_delta_rad=(proposal - nominal).tolist(),
        per_axis_trust_region_is_observed_probe_range=True,
        physically_verified=False,
        neural_actor_updates=0,
        hardware_authorized=False,
    )
