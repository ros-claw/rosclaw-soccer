"""Finite constrained reference-path solve, not neural-policy optimization.

Only reference samples enter the NN. Joint ranges constrain all 49 samples;
the B-spline derivative control hull bounds additional velocity continuously.
Degree five with interior multiplicity at most three retains C2 continuity.
"""

from typing import Any

import numpy as np


def bounded_contact_curve(
    nominal_hg: np.ndarray,
    waypoint_hg: np.ndarray,
    limits_hg: np.ndarray,
    peak: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.interpolate import BSpline  # type: ignore[import-untyped]
    from scipy.optimize import linprog  # type: ignore[import-untyped]

    from rosclaw_soccer.rsi.contact_reference_plan import _c2_offsets

    if (
        type(peak) is not int
        or not 1 <= peak < 48
        or any(
            type(x) is not np.ndarray or x.dtype != np.float64
            for x in (nominal_hg, waypoint_hg, limits_hg)
        )
        or nominal_hg.shape != (49, 29)
        or waypoint_hg.shape != (29,)
        or limits_hg.shape != (29, 2)
        or any(not np.isfinite(x).all() for x in (nominal_hg, waypoint_hg, limits_hg))
        or (limits_hg[:, 0] >= limits_hg[:, 1]).any()
    ):
        raise ValueError("finite HG49 reference, absolute waypoint and physical bounds required")
    # Fixed spatially neutral knot rule, not a search over failure outcomes.
    interior = sorted([i for i in range(4, 48, 4) if i != peak] + [peak] * 3)
    knots = np.array([0.0] * 6 + interior + [48.0] * 6) * 0.02
    count = len(knots) - 6
    basis = BSpline(knots, np.eye(count), 5)
    times = np.arange(49) * 0.02
    b, db = basis(times), basis.derivative()(times)
    derivative_controls = np.zeros((count - 1, count))
    for i in range(count - 1):
        scale = 5 / (knots[i + 6] - knots[i + 1])
        derivative_controls[i, i : i + 2] = [-scale, scale]
    # Offset, velocity, acceleration vanish at both boundaries. Offset at
    # contact is committed, and its derivative is zero as in the old curve.
    equal = np.vstack(
        [
            b[0],
            db[0],
            basis.derivative(2)(0.0),
            b[-1],
            db[-1],
            basis.derivative(2)(0.96),
            b[peak],
            db[peak],
        ]
    )
    delta = np.zeros(29)
    delta[:12] = waypoint_hg[:12] - nominal_hg[peak, :12]
    preferred, _ = _c2_offsets(delta, tuple(range(29)), peak)
    pos, vel = np.zeros((49, 29)), np.zeros((49, 29))
    objectives = []
    coefficients = []
    derivative_peaks = []
    for joint in range(12):
        # Minimize maximum departure from the original C2 offset curve.
        # LP coefficients use only own reference, own IK and body bounds.
        ub = np.vstack(
            [
                np.c_[b, np.zeros(49)],
                np.c_[-b, np.zeros(49)],
                np.c_[derivative_controls, np.zeros(count - 1)],
                np.c_[-derivative_controls, np.zeros(count - 1)],
                np.c_[b, -np.ones(49)],
                np.c_[-b, -np.ones(49)],
            ]
        )
        rhs = np.r_[
            limits_hg[joint, 1] - nominal_hg[:, joint],
            nominal_hg[:, joint] - limits_hg[joint, 0],
            np.full(2 * (count - 1), 6.0),
            preferred[:, joint],
            -preferred[:, joint],
        ]
        commitment = np.r_[np.zeros(6), delta[joint], 0.0]
        solved = linprog(
            np.r_[np.zeros(count), 1.0],
            A_ub=ub,
            b_ub=rhs,
            A_eq=np.c_[equal, np.zeros(8)],
            b_eq=commitment,
            bounds=[(None, None)] * count + [(0, None)],
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        if not solved.success or solved.x is None or not np.isfinite(solved.x).all():
            raise ValueError(f"bounded reference curve infeasible at HG joint {joint}")
        coeff = solved.x[:count]
        if (
            abs(equal @ coeff - commitment).max() > 1e-8
            or (ub @ solved.x - rhs).max() > 1e-8
            or abs(derivative_controls @ coeff).max() > 6.0 + 1e-8
        ):
            raise ValueError("bounded reference curve failed independent constraint check")
        pos[:, joint], vel[:, joint] = b @ coeff, db @ coeff
        objectives.append(float(solved.fun))
        coefficients.append(tuple(float(v) for v in coeff))
        derivative_peaks.append(float(abs(derivative_controls @ coeff).max()))
    # Exact boundary zeros avoid changing the entry or the post-window branch
    # by tiny linear solver residuals. These values are equality constraints.
    pos[[0, 48]], vel[[0, 48]] = 0.0, 0.0
    return (
        pos,
        vel,
        dict(
            schema="soccer.bounded_contact_reference_curve.v1",
            degree=5,
            interior_knots_controls=interior,
            linear_programs=12,
            maximum_departure_from_fixed_offset_rad=max(objectives),
            continuous_additional_velocity_bound_rad_s=6.0,
            maximum_derivative_control_rad_s=max(derivative_peaks),
            knots_seconds=tuple(float(v) for v in knots),
            coefficients_hg12=tuple(coefficients),
            physical_joint_bounds_scope="49_consumed_reference_samples",
            neural_optimizer_updates=0,
        ),
    )
