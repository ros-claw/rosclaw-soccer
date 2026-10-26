"""One current-state response model from four bounded positive SIM probes."""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.small_strict_qp import solve_small_strict_qp
from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan

AXES = (7, 8, 11, 12)
PROBE_FRACTION = 0.125


def propose_multi_axis_torque_response(
    parent: dict[str, Any],
    supports: dict[int, dict[str, Any]],
    *,
    body_hash: str,
    parent_policy_hash: str,
) -> tuple[CFtrackTorqueIntentPlan, dict[str, Any]]:
    if (
        parent.get("private_safe") is not True
        or parent.get("post60_observed") is not True
        or type(supports) is not dict
        or set(supports) != set(AXES)
    ):
        raise ValueError("safe complete parent and all four declared probe results required")
    first = parent.get("first_foot_control")
    if type(first) is not int or not 25 <= first <= 99:
        raise ValueError("bounded current parent contact window required")
    base = np.array([parent["forward_60_m"], parent["lateral_60_m"]], np.float64)
    if not np.isfinite(base).all():
        raise ValueError("finite parent physical outcome required")
    jacobian = np.zeros((2, 4), np.float64)
    upper = np.zeros(4, np.float64)
    excluded = []
    for i, axis in enumerate(AXES):
        support = supports[axis]
        if support.get("private_safe") is not True or support.get("post60_observed") is not True:
            excluded.append(axis)
            continue
        value = np.array([support["forward_60_m"], support["lateral_60_m"]], np.float64)
        if not np.isfinite(value).all():
            raise ValueError("finite safe support outcome required")
        jacobian[:, i] = (value - base) / PROBE_FRACTION
        upper[i] = 0.25 - 1e-7
    if not upper.any():
        raise ValueError("all current positive support probes unsafe or censored; abstain")
    weights = np.diag([1.0, 4.0])
    target = np.array([max(1.5, float(base[0])), 0.0], np.float64)
    h = jacobian.T @ weights @ jacobian + 0.1 * np.eye(4)
    g = jacobian.T @ weights @ (base - target) + 0.25
    jx, jy = jacobian
    inequalities = np.vstack([-np.eye(4), np.eye(4), -jx, jy - 0.2 * jx, -jy - 0.2 * jx])
    bounds = np.r_[
        np.zeros(4), upper, base[0] - 1.2, 0.2 * base[0] - base[1], 0.2 * base[0] + base[1]
    ]
    solution, certificate = solve_small_strict_qp(h, g, inequalities, bounds)
    solution[abs(solution) <= 1e-10] = 0.0
    if (
        (solution < 0).any()
        or (solution > upper).any()
        or (inequalities @ solution - bounds > 1e-9).any()
    ):
        raise ValueError("encoded solution violates strict torque or planning constraints")
    fractions = [0.0] * 29
    for axis, value in zip(AXES, solution, strict=True):
        fractions[axis] = float(value)
    plan = CFtrackTorqueIntentPlan(first - 25, 24, tuple(fractions), body_hash, parent_policy_hash)
    plan.offsets()
    return plan, dict(
        axes=AXES,
        probe_fraction=PROBE_FRACTION,
        current_parent_xy=base.tolist(),
        current_jacobian_per_effort_fraction=jacobian.tolist(),
        excluded_unsafe_or_censored_axes=excluded,
        nonnegative_only=True,
        predicted_xy=(base + jacobian @ solution).tolist(),
        solution_fractions=solution.tolist(),
        certificate=certificate,
        physical_feasibility_not_implied=True,
        neural_optimizer_updates=0,
    )
