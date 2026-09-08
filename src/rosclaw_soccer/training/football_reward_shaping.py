"""Finite-episode football approach shaping with an explicit terminal boundary.

The discounted sum is -Phi(initial), independent of whether the episode ends
with the ball beside the player. This algebra is not a PPO convergence claim.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

REWARD_SHAPING_MODES = ("legacy", "terminal_potential_v1")


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
