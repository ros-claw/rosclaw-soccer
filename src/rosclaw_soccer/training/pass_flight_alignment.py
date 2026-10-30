"""SIM-only instantaneous pass-ray diagnostic; never a receipt of reception."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PassFlightAlignment:
    forward_speed_mps: float
    lateral_speed_mps: float
    closest_time_sec: float
    closest_distance_m: float
    within_capture_corridor: bool
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False


def inspect_pass_flight_alignment(
    *,
    ball_xy_m: tuple[float, float],
    ball_velocity_xy_mps: tuple[float, float],
    receiver_xy_m: tuple[float, float],
    capture_radius_m: float,
    horizon_sec: float,
) -> PassFlightAlignment:
    """Compare a constant-velocity ray to a fixed observed receiver point.

    This is a diagnostic prediction, not evidence of a launch, intention,
    collision-free path or reception. It ignores friction, bounce, spin and
    receiver motion. The caller authenticates observations and chooses the
    explicit corridor and horizon before seeing evaluation results.
    """
    vectors = (ball_xy_m, ball_velocity_xy_mps, receiver_xy_m)
    if any(
        type(value) is not tuple
        or len(value) != 2
        or any(type(x) not in (float, int) or not math.isfinite(x) for x in value)
        for value in vectors
    ) or any(
        type(value) not in (float, int) or not math.isfinite(value) or value <= 0.0
        for value in (capture_radius_m, horizon_sec)
    ):
        raise ValueError("finite planar observations and positive explicit bounds required")
    dx, dy = receiver_xy_m[0] - ball_xy_m[0], receiver_xy_m[1] - ball_xy_m[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        raise ValueError("receiver direction is undefined at the ball")
    vx, vy = ball_velocity_xy_mps
    speed_sq = vx * vx + vy * vy
    if not math.isfinite(speed_sq) or speed_sq <= 1e-12:
        raise ValueError("finite nonzero ball velocity required")
    forward = (vx * dx + vy * dy) / distance
    lateral = (vx * -dy + vy * dx) / distance
    closest_time = max(0.0, min(horizon_sec, (vx * dx + vy * dy) / speed_sq))
    miss = math.hypot(dx - vx * closest_time, dy - vy * closest_time)
    if not all(math.isfinite(v) for v in (forward, lateral, closest_time, miss)):
        raise ValueError("geometry overflow")
    return PassFlightAlignment(
        forward_speed_mps=forward,
        lateral_speed_mps=lateral,
        closest_time_sec=closest_time,
        closest_distance_m=miss,
        within_capture_corridor=bool(forward > 0 and miss <= capture_radius_m),
    )
