"""Bounded actuated-joint IK proposals, NOT a dynamics/balance controller.

Soft geometric objectives share the same joint variables. No floating-base or
ball degrees of freedom, physics stepping, driver or runtime is available here.
Joint-position, per-observation motion, command-continuity and nominal-trust
bounds are intersected before solving. Infeasible proposals abstain.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinearJointTask:
    name: str
    jacobian: np.ndarray[Any, Any]
    displacement_m: np.ndarray[Any, Any]
    cost: float


@dataclass(frozen=True)
class JointTaskProposal:
    status: str
    target_rad: tuple[float, ...] | None
    task_residual_m: dict[str, tuple[float, ...]]
    weighted_cost: float | None
    physically_qualified: bool = field(init=False, default=False)


def solve_bounded_joint_tasks(
    current_rad: np.ndarray[Any, Any],
    nominal_rad: np.ndarray[Any, Any],
    previous_target_rad: np.ndarray[Any, Any],
    limits_rad: np.ndarray[Any, Any],
    tasks: tuple[LinearJointTask, ...],
    *,
    observation_step_cap_rad: float,
    command_step_cap_rad: float,
    nominal_trust_rad: float,
    posture_cost: float,
) -> JointTaskProposal:
    """Solve a 29-joint bounded least-squares proposal against actual current q.

    Unlike a local delta blindly added to an unrelated nominal pose, each geometry
    Jacobian acts on (target-current); the nominal pose is a separate soft objective.
    Task weights do not imply contact stability. Physics must test the proposal.
    """
    for raw, shape in (
        (current_rad, (29,)),
        (nominal_rad, (29,)),
        (previous_target_rad, (29,)),
        (limits_rad, (29, 2)),
    ):
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != shape
            or raw.dtype != np.float64
            or not np.isfinite(raw).all()
            or np.max(abs(raw)) > 1000
        ):
            raise ValueError("finite canonical actuated-only float64 joint arrays required")
    if np.any(limits_rad[:, 0] >= limits_rad[:, 1]):
        raise ValueError("ordered physical joint limits required")
    for value, maximum in (
        (observation_step_cap_rad, 0.2),
        (command_step_cap_rad, 0.1),
        (nominal_trust_rad, 0.35),
        (posture_cost, 10.0),
    ):
        if type(value) is not float or not np.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("explicit positive bounded research solver parameters required")
    if (
        type(tasks) is not tuple
        or not 1 <= len(tasks) <= 16
        or any(type(t) is not LinearJointTask or type(t.name) is not str for t in tasks)
        or len({t.name for t in tasks}) != len(tasks)
    ):
        raise ValueError("bounded distinct explicit geometric tasks required")
    design, target = [], []
    for task in tasks:
        if (
            type(task) is not LinearJointTask
            or type(task.name) is not str
            or not task.name
            or len(task.name) > 64
            or not isinstance(task.jacobian, np.ndarray)
            or not isinstance(task.displacement_m, np.ndarray)
            or task.jacobian.shape != (3, 29)
            or task.displacement_m.shape != (3,)
            or any(
                a.dtype != np.float64 or not np.isfinite(a).all()
                for a in (task.jacobian, task.displacement_m)
            )
            or np.max(abs(task.jacobian)) > 1e4
            or np.max(abs(task.displacement_m)) > 0.25
            or type(task.cost) is not float
            or not np.isfinite(task.cost)
            or not 0 < task.cost <= 100
        ):
            raise ValueError("finite bounded 3D objectives on canonical joint variables required")
        scale = np.sqrt(task.cost)
        design.append(scale * task.jacobian)
        target.append(scale * task.displacement_m)
    design.append(np.sqrt(posture_cost) * np.eye(29))
    target.append(np.sqrt(posture_cost) * (nominal_rad - current_rad))
    matrix, desired = np.vstack(design), np.concatenate(target)
    lower = np.maximum.reduce(
        (
            limits_rad[:, 0],
            current_rad - observation_step_cap_rad,
            previous_target_rad - command_step_cap_rad,
            nominal_rad - nominal_trust_rad,
        )
    )
    upper = np.minimum.reduce(
        (
            limits_rad[:, 1],
            current_rad + observation_step_cap_rad,
            previous_target_rad + command_step_cap_rad,
            nominal_rad + nominal_trust_rad,
        )
    )
    if np.any(lower > upper):
        return JointTaskProposal("ABSTAIN_INFEASIBLE_BOUND_INTERSECTION", None, {}, None)
    lo, hi = lower - current_rad, upper - current_rad
    fixed = lo == hi
    delta = np.zeros(29, dtype=np.float64)
    delta[fixed] = lo[fixed]
    if np.any(~fixed):
        from scipy.optimize import lsq_linear  # type: ignore[import-untyped]

        solved = lsq_linear(
            matrix[:, ~fixed],
            desired - matrix[:, fixed] @ delta[fixed],
            bounds=(lo[~fixed], hi[~fixed]),
            method="bvls",
            tol=1e-10,
            max_iter=300,
        )
        if not solved.success or not np.isfinite(solved.x).all():
            return JointTaskProposal("ABSTAIN_SOLVER_NONCONVERGENCE", None, {}, None)
        delta[~fixed] = solved.x
    # Exact projection of floating-point roundoff back into the original absolute
    # intersection, never expansion of any bound or a repair of infeasible input.
    result = np.clip(current_rad + delta, lower, upper)
    delta = result - current_rad
    residual = matrix @ delta - desired
    return JointTaskProposal(
        "NUMERIC_PROPOSAL_NOT_PHYSICALLY_QUALIFIED",
        tuple(float(v) for v in result),
        {t.name: tuple(float(v) for v in t.jacobian @ delta - t.displacement_m) for t in tasks},
        float(0.5 * np.dot(residual, residual)),
    )
