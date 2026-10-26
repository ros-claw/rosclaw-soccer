"""Finite C2 reference correction under vector-speed and sampled workspace bounds.

No robot, policy, simulator, task reward or execution API lives here. A quintic
B-spline correction has a continuously bounded derivative by constraining all
derivative control vectors. Workspace/floor bounds apply at consumed samples.
This is a numerical reference-path proposal, not physical feasibility evidence.
"""

from typing import Any

import numpy as np


def solve_bounded_taskspace_reference(
    nominal: np.ndarray,
    anchor: np.ndarray,
    waypoint: np.ndarray,
    preferred_offset: np.ndarray,
    *,
    peak: int,
    dt: float,
    speed_limit: float,
    workspace_radius: float,
    floor_z: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.interpolate import BSpline  # type: ignore[import-untyped]
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    if (
        any(
            type(a) is not np.ndarray or a.dtype != np.float64 or not np.isfinite(a).all()
            for a in (nominal, anchor, waypoint, preferred_offset)
        )
        or nominal.ndim != 2
        or nominal.shape[1] != 3
        or not 9 <= len(nominal) <= 101
        or anchor.shape != nominal.shape
        or preferred_offset.shape != nominal.shape
        or waypoint.shape != (3,)
        or type(peak) is not int
        or not 1 <= peak < len(nominal) - 1
        or any(abs(a).max() > 100 for a in (nominal, anchor, waypoint, preferred_offset))
        or any(
            type(v) is not float or not np.isfinite(v)
            for v in (dt, speed_limit, workspace_radius, floor_z)
        )
        or not 0 < dt <= 0.1
        or not 0 < speed_limit <= 10
        or not 0 < workspace_radius <= 10
        or abs(floor_z) > 100
    ):
        raise ValueError("finite bounded typed 3D reference-path problem required")
    n, end = len(nominal), len(nominal) - 1
    if np.linalg.norm(waypoint - anchor[peak]) > workspace_radius or waypoint[2] < floor_z:
        raise ValueError("absolute waypoint outside declared reference workspace")
    interior = sorted([i for i in range(4, end, 4) if i != peak] + [peak] * 3)
    knots = np.array([0.0] * 6 + interior + [float(end)] * 6) * dt
    count = len(knots) - 6
    basis = BSpline(knots, np.eye(count), 5)
    times = np.arange(n) * dt
    b: np.ndarray = basis(times)
    db: np.ndarray = basis.derivative()(times)
    derivative = np.zeros((count - 1, count))
    for i in range(count - 1):
        scale = 5 / (knots[i + 6] - knots[i + 1])
        derivative[i, i : i + 2] = [-scale, scale]
    equal = np.vstack(
        [
            b[0],
            db[0],
            basis.derivative(2)(0.0),
            b[-1],
            db[-1],
            basis.derivative(2)(end * dt),
            b[peak],
            db[peak],
        ]
    )
    rhs = np.zeros((8, 3))
    rhs[6] = waypoint - nominal[peak]
    # Zero OFFSET velocity at peak preserves nominal follow-through velocity;
    # it does not assert zero total reference/body velocity at contact.
    scaling = np.linalg.norm(equal, axis=1)
    e = np.kron(equal / scaling[:, None], np.eye(3))
    r = (rhs / scaling[:, None]).ravel()
    delta = waypoint - nominal[peak]
    u = np.arange(n, dtype=np.float64)
    up = u <= peak
    tau = np.where(up, u / peak, (u - peak) / (end - peak))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    w = np.where(up, s, 1 - s)
    initial = np.linalg.lstsq(b, w[:, None] * delta, rcond=None)[0].ravel()
    floor_jac = np.zeros((n, count, 3))
    floor_jac[:, :, 2] = b

    def objective(x: np.ndarray) -> float:
        c = x.reshape(count, 3)
        error = b @ c - preferred_offset
        return float(0.5 * np.sum(error * error) + 5e-9 * np.sum(c * c))

    def gradient(x: np.ndarray) -> np.ndarray:
        c = x.reshape(count, 3)
        result: np.ndarray = (b.T @ (b @ c - preferred_offset) + 1e-8 * c).ravel()
        return result

    def speed(x: np.ndarray) -> np.ndarray:
        v = derivative @ x.reshape(count, 3)
        result: np.ndarray = speed_limit**2 - np.sum(v * v, axis=1)
        return result

    def speed_jac(x: np.ndarray) -> np.ndarray:
        v = derivative @ x.reshape(count, 3)
        result: np.ndarray = (-2 * derivative[:, :, None] * v[:, None, :]).reshape(count - 1, -1)
        return result

    def workspace(x: np.ndarray) -> np.ndarray:
        position = nominal + b @ x.reshape(count, 3) - anchor
        result: np.ndarray = workspace_radius**2 - np.sum(position * position, axis=1)
        return result

    def workspace_jac(x: np.ndarray) -> np.ndarray:
        position = nominal + b @ x.reshape(count, 3) - anchor
        result: np.ndarray = (-2 * b[:, :, None] * position[:, None, :]).reshape(n, -1)
        return result

    solved = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda x: e @ x - r, "jac": lambda x: e},
            {"type": "ineq", "fun": speed, "jac": speed_jac},
            {"type": "ineq", "fun": workspace, "jac": workspace_jac},
            {
                "type": "ineq",
                "fun": lambda x: nominal[:, 2] + (b @ x.reshape(count, 3))[:, 2] - floor_z,
                "jac": lambda x: floor_jac.reshape(n, -1),
            },
        ],
        options={"maxiter": 200, "ftol": 1e-11, "disp": False},
    )
    if not solved.success or solved.x is None or not np.isfinite(solved.x).all():
        raise ValueError("bounded C2 taskspace solver abstained: " + str(solved.message))
    coeff = solved.x.reshape(count, 3)
    offset = b @ coeff
    position = nominal + offset
    equality_error = float(abs(equal @ coeff - rhs).max())
    control_speed = float(np.linalg.norm(derivative @ coeff, axis=1).max())
    max_workspace = float(np.linalg.norm(position - anchor, axis=1).max())
    if (
        equality_error > 1e-7
        or control_speed > speed_limit + 1e-8
        or max_workspace > workspace_radius + 1e-9
        or position[:, 2].min() < floor_z - 1e-9
    ):
        raise ValueError("bounded C2 taskspace proposal failed independent constraint check")
    # Restore equality-constrained boundary samples exactly, not clipping.
    position[[0, end]] = nominal[[0, end]]
    return position, dict(
        schema="soccer.bounded_taskspace_reference.v1",
        solver="SLSQP",
        solver_iterations=int(solved.nit),
        maximum_solver_iterations=200,
        degree=5,
        knots_seconds=tuple(float(v) for v in knots),
        offset_coefficients=tuple(tuple(float(v) for v in row) for row in coeff),
        maximum_equality_error=equality_error,
        maximum_derivative_control_vector_speed_m_s=control_speed,
        continuous_additional_reference_speed_bound_m_s=speed_limit,
        maximum_sampled_workspace_m=max_workspace,
        position_workspace_is_sampled_not_continuous=True,
        zero_offset_velocity_at_peak_preserves_nominal_follow_through=True,
        objective=float(solved.fun),
        physical_feasibility_proven=False,
        neural_optimizer_updates=0,
    )
