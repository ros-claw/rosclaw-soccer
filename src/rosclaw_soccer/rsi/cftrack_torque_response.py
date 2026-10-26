"""One state-local positive torque probe, one constrained numeric proposal.

This fits a two-output, one-input response, not neural weights. The local
linearization requires a new physical veto and cannot certify contact physics.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.small_strict_qp import solve_small_strict_qp
from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan


def propose_torque_response(
    parent: dict[str, Any],
    support: dict[str, Any],
    *,
    body_hash: str,
    parent_policy_hash: str,
) -> tuple[CFtrackTorqueIntentPlan, dict[str, Any]]:
    if any(
        m.get("private_safe") is not True or m.get("post60_observed") is not True
        for m in (parent, support)
    ):
        raise ValueError("safe complete current parent and positive torque support required")
    first = parent.get("first_foot_control")
    if type(first) is not int or not 25 <= first <= 99:
        raise ValueError("current parent contact must leave a bounded C2 control window")
    base = np.array([parent["forward_60_m"], parent["lateral_60_m"]], np.float64)
    observed = np.array([support["forward_60_m"], support["lateral_60_m"]], np.float64)
    if not np.isfinite(base).all() or not np.isfinite(observed).all():
        raise ValueError("finite measured physical response required")
    jacobian = (observed - base) / 0.125
    jx, jy = jacobian
    # Fixed cost: .5*4*lateral^2 + .5*.1*beta^2 + .25*beta; beta is nonnegative.
    h = np.array([[4 * jy * jy + 0.1]], np.float64)
    g = np.array([4 * base[1] * jy + 0.25], np.float64)
    inequalities = np.array([[-1.0], [1.0], [-jx], [jy - 0.2 * jx], [-jy - 0.2 * jx]], np.float64)
    bounds = np.array(
        [0.0, 0.25 - 1e-7, base[0] - 1.2, 0.2 * base[0] - base[1], 0.2 * base[0] + base[1]],
        np.float64,
    )
    solution, certificate = solve_small_strict_qp(h, g, inequalities, bounds)
    beta = float(solution[0])
    if not 0 <= beta <= 0.25:
        raise ValueError("strict torque authority bound violated; no clipping")
    fractions = [0.0] * 29
    fractions[7] = beta
    plan = CFtrackTorqueIntentPlan(first - 25, 24, tuple(fractions), body_hash, parent_policy_hash)
    plan.offsets()
    return plan, dict(
        current_parent_xy=base.tolist(),
        current_support_xy=observed.tolist(),
        current_jacobian_per_effort_fraction=jacobian.tolist(),
        fitted_axis_hg=7,
        positive_probe_fraction=0.125,
        predicted_xy=(base + jacobian * beta).tolist(),
        solution_fraction=beta,
        certificate=certificate,
        physical_feasibility_not_implied=True,
        single_proposal=True,
        neural_optimizer_updates=0,
    )
