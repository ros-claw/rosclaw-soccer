"""Finite-episode football approach shaping with an explicit terminal boundary.

The discounted sum is -Phi(initial), independent of whether the episode ends
with the ball beside the player. This algebra is not a PPO convergence claim.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

REWARD_SHAPING_MODES = ("legacy", "terminal_potential_v1", "contact_safety_v1")


def joint_safety_penalty(margins: NDArray[np.float64]) -> NDArray[np.float64]:
    """Measured 50 Hz joint-limit risk, not a substitute for the physical guard."""
    if margins.ndim != 3 or margins.shape[1:] != (8, 29) or not np.isfinite(margins).all():
        raise ValueError("joint safety rewards require finite eight-player margins")
    closest = margins.min(axis=2)
    return np.asarray(
        -0.25 * (closest < 0) - 0.02 * np.clip((0.03 - closest) / 0.03, 0, 1), dtype=np.float64
    )


def terminal_approach_shaping(
    distance: NDArray[np.float64], *, gamma: float
) -> NDArray[np.float64]:
    if (
        type(gamma) is bool
        or not math.isfinite(gamma)
        or not 0.9 <= gamma < 1.0
        or distance.ndim != 2
        or distance.shape[0] < 1
        or distance.shape[1] != 8
        or not np.all(np.isfinite(distance))
        or np.any(distance < 0)
    ):
        raise ValueError("finite nonnegative eight-player potential and bounded discount required")
    potential = 2.0 * np.exp(-4.0 * distance)
    following = np.zeros_like(potential)
    following[:-1] = potential[1:]
    return gamma * following - potential
