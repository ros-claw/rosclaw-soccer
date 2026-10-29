"""Bounded SIM cold-start ball ablation, never an active-world mutation."""

import numpy as np


def initial_ball_factorial(position, angular_local, *, ground_ball, zero_spin):
    """Copy a declared initial ball state and apply at most two explicit factors."""
    if (
        type(ground_ball) is not bool
        or type(zero_spin) is not bool
        or not isinstance(position, np.ndarray)
        or not isinstance(angular_local, np.ndarray)
        or position.shape != (3,)
        or angular_local.shape != (3,)
        or position.dtype.kind != "f"
        or angular_local.dtype.kind != "f"
        or not np.isfinite(position).all()
        or not np.isfinite(angular_local).all()
        or not 0.11 <= position[2] <= 0.21
        or np.linalg.norm(angular_local) > 100
    ):
        raise ValueError("bounded finite initial ball state and explicit boolean factors required")
    adjusted_position = position.copy()
    adjusted_spin = angular_local.copy()
    if ground_ball:
        adjusted_position[2] = 0.11
    if zero_spin:
        adjusted_spin[:] = 0
    return adjusted_position, adjusted_spin
