"""Bounded active-set enumeration for tiny strictly convex quadratic programs.

Returns a numerically checked KKT witness, not physical feasibility evidence.
No robot, policy, reward, simulator, optimizer training or hardware knowledge.
"""

from itertools import combinations
from typing import Any

import numpy as np


def solve_small_strict_qp(
    hessian: np.ndarray, gradient: np.ndarray, inequalities: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        any(
            type(x) is not np.ndarray or x.dtype != np.float64 or not np.isfinite(x).all()
            for x in (hessian, gradient, inequalities, bounds)
        )
        or gradient.ndim != 1
    ):
        raise ValueError("finite float64 quadratic program arrays required")
    n = len(gradient)
    if (
        not 1 <= n <= 4
        or hessian.shape != (n, n)
        or inequalities.ndim != 2
        or inequalities.shape[1] != n
        or not 1 <= len(inequalities) <= 16
        or bounds.shape != (len(inequalities),)
        or not np.allclose(hessian, hessian.T, rtol=0, atol=1e-12)
        or np.linalg.eigvalsh(hessian).min() <= 1e-10
        or np.linalg.cond(hessian) > 1e8
    ):
        raise ValueError("bounded tiny well-conditioned strictly convex problem required")
    tolerance = 1e-9
    best = None
    best_certificate = None
    best_value = float("inf")
    enumerated = 0
    for size in range(min(n, len(inequalities)) + 1):
        for active in combinations(range(len(inequalities)), size):
            enumerated += 1
            a = inequalities[list(active)]
            kkt = np.block([[hessian, a.T], [a, np.zeros((size, size))]])
            rhs = np.r_[-gradient, bounds[list(active)]]
            try:
                solution = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue
            x, multiplier = solution[:n], solution[n:]
            if not np.isfinite(solution).all():
                continue
            primal = float(np.max(inequalities @ x - bounds))
            stationarity = float(np.max(abs(hessian @ x + gradient + a.T @ multiplier)))
            equality = float(np.max(abs(a @ x - bounds[list(active)]))) if size else 0.0
            minimum = float(multiplier.min()) if size else 0.0
            if (
                primal > tolerance
                or stationarity > tolerance
                or equality > tolerance
                or minimum < -tolerance
            ):
                continue
            value = float(0.5 * x @ hessian @ x + gradient @ x)
            if value < best_value:
                best = x.copy()
                best_value = value
                best_certificate = dict(
                    active_constraints=active,
                    objective_without_constant=value,
                    maximum_primal_violation=max(0.0, primal),
                    maximum_stationarity_residual=stationarity,
                    maximum_active_equality_residual=equality,
                    minimum_active_multiplier=minimum,
                    numerical_tolerance=tolerance,
                )
    if best is None or best_certificate is None:
        raise ValueError("no numerically certified feasible KKT point; no constraint relaxation")
    best_certificate["enumerated_active_sets"] = enumerated
    best_certificate["physical_feasibility_not_implied"] = True
    return best, best_certificate
