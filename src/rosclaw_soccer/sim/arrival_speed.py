"""Bounded arrival-speed cues for a delayed motion-reference planner.

This numerical cue is not a stopping-distance or collision-safety guarantee.
Actual tracking, clearance, actuator limits and skill admission remain external.
"""

from __future__ import annotations

import math


def delayed_arrival_speed(
    *,
    remaining_distance_m: float,
    maximum_speed_mps: float,
    assumed_deceleration_mps2: float,
    response_delay_sec: float,
    arrival_margin_m: float,
) -> float:
    """Solve d = delay*v + v²/(2*a), subtracting an explicit arrival margin.

    The caller supplies assumed response parameters, not certified physical
    capabilities. This produces a reference-planner speed hint, not a velocity
    override or permission to exceed an existing motor/navigation contract.
    No robot, simulator, goal coordinate, policy or runtime dependency is used.
    """
    values = (
        remaining_distance_m,
        maximum_speed_mps,
        assumed_deceleration_mps2,
        response_delay_sec,
        arrival_margin_m,
    )
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values) or (
        not 0 <= remaining_distance_m <= 1000
        or not 0 < maximum_speed_mps <= 10
        or not 0 < assumed_deceleration_mps2 <= 20
        or not 0 <= response_delay_sec <= 5
        or not 0 <= arrival_margin_m <= 5
    ):
        raise ValueError("finite bounded distance, speed, deceleration, delay and margin required")
    distance = max(0.0, remaining_distance_m - arrival_margin_m)
    delay_term = assumed_deceleration_mps2 * response_delay_sec
    cue = math.sqrt(delay_term**2 + 2 * assumed_deceleration_mps2 * distance) - delay_term
    return float(min(maximum_speed_mps, max(0.0, cue)))
