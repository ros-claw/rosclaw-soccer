"""Bounded geometric contact proposals for simulation-only future evaluation.

This module does not control a robot, integrate physics, or establish contact.
Its input may contain privileged simulated future state; the result is only a
candidate for a separately measured dynamics experiment.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactReachProposal:
    delta: np.ndarray
    raw_delta: np.ndarray
    scale: float
    predicted_displacement: np.ndarray


def bounded_contact_reach(
    jacobian: np.ndarray,
    displacement: np.ndarray,
    *,
    damping: float,
    max_joint_delta: float,
) -> ContactReachProposal:
    """Solve a damped least-squares point reach with a global joint bound.

    The bound preserves the solution's direction. Input validation is strict:
    this candidate generator must never convert invalid evidence into a plan.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    goal = np.asarray(displacement, dtype=np.float64)
    if (
        jac.ndim != 2
        or jac.shape[0] != 3
        or jac.shape[1] < 1
        or goal.shape != (3,)
        or not np.all(np.isfinite(jac))
        or not np.all(np.isfinite(goal))
        or not np.isfinite(damping)
        or not np.isfinite(max_joint_delta)
        or damping <= 0
        or max_joint_delta <= 0
    ):
        raise ValueError("finite 3D Jacobian, displacement and positive bounds required")
    raw = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), goal)
    peak = float(np.max(np.abs(raw)))
    scale = min(1.0, max_joint_delta / peak) if peak else 1.0
    delta = raw * scale
    if not np.all(np.isfinite(delta)):
        raise ValueError("non-finite damped reach candidate")
    return ContactReachProposal(delta, raw, scale, jac @ delta)
