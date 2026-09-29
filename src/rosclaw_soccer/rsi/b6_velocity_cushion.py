"""Bounded SIM_ONLY precontact foot-velocity matching on immutable kinematics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def velocity_match_joint_delta(
    *,
    jacobian: NDArray[np.float64],
    foot_velocity_mps: NDArray[np.float64],
    ball_velocity_mps: NDArray[np.float64],
    gain: float,
    horizon_sec: float = 0.04,
    maximum_joint_delta_rad: float = 0.03,
) -> NDArray[np.float64]:
    """Return a small six-joint target delta; no simulation or torque authority."""
    if (
        jacobian.shape != (3, 6)
        or foot_velocity_mps.shape != (3,)
        or ball_velocity_mps.shape != (3,)
        or any(
            not np.isfinite(value).all()
            for value in (jacobian, foot_velocity_mps, ball_velocity_mps)
        )
        or type(gain) not in (int, float)
        or type(horizon_sec) not in (int, float)
        or type(maximum_joint_delta_rad) not in (int, float)
        or not all(math.isfinite(value) for value in (gain, horizon_sec, maximum_joint_delta_rad))
        or not 0.0 <= gain <= 2.0
        or not 0.02 <= horizon_sec <= 0.06
        or not 0.005 <= maximum_joint_delta_rad <= 0.04
    ):
        raise ValueError("finite bounded SIM_ONLY velocity-matching action required")
    horizontal_jacobian = jacobian[:2]
    relative = ball_velocity_mps[:2] - foot_velocity_mps[:2]
    gram = horizontal_jacobian @ horizontal_jacobian.T + 0.04 * np.eye(2)
    delta = gain * horizon_sec * horizontal_jacobian.T @ np.linalg.solve(gram, relative)
    return np.asarray(
        np.clip(delta, -maximum_joint_delta_rad, maximum_joint_delta_rad), dtype=np.float64
    )


__all__ = ["velocity_match_joint_delta"]
