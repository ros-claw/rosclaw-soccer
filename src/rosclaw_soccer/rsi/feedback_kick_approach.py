"""A new measured-state approach hypothesis, not another constant-speed sweep."""

import math

import numpy as np


def velocity_matched_approach(
    *,
    root_position: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    cruise_speed: float,
    entry_gap: float,
    lateral_offset: float,
) -> np.ndarray:
    """Cruise at distance, then match ball motion while closing the entry gap.

    Fixed gains 1.5/s longitudinal and2/s lateral; never hold a timed stop.
    Aim .20m inside admission so a static ball cannot create an asymptotic
    equilibrium just outside the skill's handoff boundary.
    No backward component is requested in this first hypothesis. The caller
    retains its exact body-domain, slew and torque checks and skill handoff.
    """
    vectors = (root_position, ball_position, ball_velocity)
    if (
        any(
            type(x) is not np.ndarray
            or x.shape != (3,)
            or x.dtype != np.float64
            or not np.isfinite(x).all()
            for x in vectors
        )
        or any(
            type(x) not in (float, int) or not math.isfinite(x)
            for x in (cruise_speed, entry_gap, lateral_offset)
        )
        or not (
            0.5 <= cruise_speed <= 2 and 0.45 <= entry_gap <= 1.1 and abs(lateral_offset) <= 0.15
        )
    ):
        raise ValueError("finite measured own state and fixed bounded approach parameters required")
    longitudinal = ball_velocity[0] + 1.5 * (
        ball_position[0] - root_position[0] - (entry_gap - 0.2)
    )
    lateral = ball_velocity[1] + 2 * (ball_position[1] + lateral_offset - root_position[1])
    return np.array([np.clip(longitudinal, 0, cruise_speed), np.clip(lateral, -0.4, 0.4)])
