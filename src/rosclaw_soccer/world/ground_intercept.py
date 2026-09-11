"""Read-only short-horizon interception geometry, not a ball-control policy."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundIntercept:
    root_target_xy: tuple[float, float]
    predicted_ball_xy: tuple[float, float]
    horizon_sec: float
    kinematically_reachable: bool
    activation_ceiling: str = "SIM_ONLY"


def propose_ground_intercept(
    *,
    root_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    ball_velocity_xy: tuple[float, float],
    root_speed_mps: float = 0.7,
    contact_standoff_m: float = 0.28,
    maximum_horizon_sec: float = 0.8,
) -> GroundIntercept:
    """Earliest constant-velocity contact-region intercept or bounded lookahead.

    Assumes the root can immediately travel at the declared speed, the ball
    maintains velocity, and no obstacles exist. Restricts the ball to slower
    than the root so reachability is monotone. The caller must retain collision
    clearance, body limits, ownership, actual foot contact and skill admission.
    This does not place a ball, move a root, establish possession or learn.
    """
    for vector in (root_xy, ball_xy, ball_velocity_xy):
        if (
            type(vector) is not tuple
            or len(vector) != 2
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000 for x in vector
            )
        ):
            raise ValueError("finite bounded planar observations required")
    if any(
        type(x) not in (int, float) or not math.isfinite(x)
        for x in (root_speed_mps, contact_standoff_m, maximum_horizon_sec)
    ) or not (
        0.1 <= root_speed_mps <= 2
        and 0.15 <= contact_standoff_m <= 0.5
        and 0.1 <= maximum_horizon_sec <= 2
        and math.hypot(*ball_velocity_xy) < root_speed_mps
    ):
        raise ValueError("bounded slow-ball interception domain required")
    dx, dy = ball_xy[0] - root_xy[0], ball_xy[1] - root_xy[1]
    vx, vy = ball_velocity_xy

    def gap(t: float) -> float:
        return math.hypot(dx + vx * t, dy + vy * t) - contact_standoff_m - root_speed_mps * t

    reachable = gap(maximum_horizon_sec) <= 0
    if gap(0) <= 0:
        horizon = 0.0
    elif reachable:
        lower, upper = 0.0, maximum_horizon_sec
        for _ in range(48):
            middle = (lower + upper) / 2
            if gap(middle) <= 0:
                upper = middle
            else:
                lower = middle
        horizon = upper
    else:
        horizon = maximum_horizon_sec
    predicted = (ball_xy[0] + vx * horizon, ball_xy[1] + vy * horizon)
    rx, ry = predicted[0] - root_xy[0], predicted[1] - root_xy[1]
    distance = math.hypot(rx, ry)
    fraction = max(0.0, distance - contact_standoff_m) / max(distance, 1e-12)
    target = (root_xy[0] + fraction * rx, root_xy[1] + fraction * ry)
    return GroundIntercept(target, predicted, horizon, reachable)
