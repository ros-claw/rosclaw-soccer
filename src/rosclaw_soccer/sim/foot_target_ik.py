"""Damped task-space foot correction for SIM_ONLY contact development."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def bounded_foot_target_delta(
    jacobian_xz_m_per_rad: NDArray[np.float64],
    desired_xz_m: NDArray[np.float64],
    *,
    knee_jacobian_x_m_per_rad: NDArray[np.float64] | None = None,
    knee_retreat_m: float = 0.0,
    damping_m2: float = 0.0025,
    max_joint_delta_rad: float = 0.12,
) -> NDArray[np.float64]:
    """Solve a two-axis/three-joint local foot displacement, then clip authority.

    The caller still applies physical joint-limit projection.  This function is
    a bounded *target proposal*, not a claim that the commanded foot motion
    or ball contact occurred.
    """
    jacobian = np.asarray(jacobian_xz_m_per_rad, dtype=np.float64)
    desired = np.asarray(desired_xz_m, dtype=np.float64)
    if (
        jacobian.shape != (2, 3)
        or desired.shape != (2,)
        or not np.isfinite(jacobian).all()
        or not np.isfinite(desired).all()
        or not math.isfinite(damping_m2)
        or not 0 < damping_m2 <= 0.1
        or not math.isfinite(max_joint_delta_rad)
        or not 0 < max_joint_delta_rad <= 0.12
        or np.linalg.norm(desired) > 0.2
        or not math.isfinite(knee_retreat_m)
        or not 0.0 <= knee_retreat_m <= 0.05
        or (knee_jacobian_x_m_per_rad is None and knee_retreat_m != 0.0)
    ):
        raise ValueError("bounded finite two-axis foot target and Jacobian required")
    if knee_jacobian_x_m_per_rad is not None:
        knee_jacobian = np.asarray(knee_jacobian_x_m_per_rad, dtype=np.float64)
        if knee_jacobian.shape != (3,) or not np.isfinite(knee_jacobian).all():
            raise ValueError("finite three-joint knee Jacobian required")
        jacobian = np.vstack((jacobian, knee_jacobian))
        desired = np.concatenate((desired, np.array((-knee_retreat_m,))))
    gram = jacobian @ jacobian.T + damping_m2 * np.eye(len(desired))
    delta = jacobian.T @ np.linalg.solve(gram, desired)
    result = np.clip(delta, -max_joint_delta_rad, max_joint_delta_rad)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite IK joint target")
    return result
