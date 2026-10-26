"""Fit one bounded reference-aim proposal from current SIM physical feedback."""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.small_strict_qp import solve_small_strict_qp


def propose_reference_aim(parent: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    if any(
        m.get("private_safe") is not True or m.get("post60_observed") is not True
        for m in (parent, support)
    ):
        raise ValueError("safe complete current parent and aim probe required")
    base = np.array([parent["forward_60_m"], parent["lateral_60_m"]], np.float64)
    observed = np.array([support["forward_60_m"], support["lateral_60_m"]], np.float64)
    if not np.isfinite(base).all() or not np.isfinite(observed).all():
        raise ValueError("finite measured physical outcomes required")
    # Beta is the magnitude of a negative reference-only goal-Y offset.
    jacobian = (observed - base) / 0.4
    jx, jy = jacobian
    target = np.array([max(1.5, float(base[0])), 0.0], np.float64)
    weight = np.diag([1.0, 4.0])
    h = np.array([[float(jacobian @ weight @ jacobian) + 0.1]], np.float64)
    g = np.array([float(jacobian @ weight @ (base - target)) + 0.25], np.float64)
    a = np.array([[-1.0], [1.0], [-jx], [jy - 0.2 * jx], [-jy - 0.2 * jx]], np.float64)
    b = np.array(
        [0.0, 0.7 - 1e-6, base[0] - 1.2, 0.2 * base[0] - base[1], 0.2 * base[0] + base[1]],
        np.float64,
    )
    solution, witness = solve_small_strict_qp(h, g, a, b)
    beta = float(solution[0])
    if not 0 <= beta <= 0.7 or (a @ solution - b > 1e-9).any():
        raise ValueError("reference aim would violate bounded planning constraints")
    return dict(
        reference_goal_offset_y_m=-beta,
        source_probe_goal_offset_y_m=-0.4,
        parent_xy=base.tolist(),
        probe_xy=observed.tolist(),
        current_jacobian_per_negative_goal_offset_m=jacobian.tolist(),
        predicted_xy=(base + jacobian * beta).tolist(),
        certificate=witness,
        actual_task_goal_and_h300_unchanged=True,
        physical_feasibility_not_implied=True,
        neural_optimizer_updates=0,
    )
