"""Auxiliary learning cue from observed launch velocity, NOT goal evidence.

The estimate ignores air effects and later collisions. Callers separately
measure goal-plane accuracy and retain physical failure guards. No actuator,
ball-force command, scoring decision or candidate activation is produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BallisticLaunchGuidance:
    prediction_available: bool
    estimated_flight_time_s: float | None
    estimated_required_lateral_speed_m_s: float | None
    estimated_required_vertical_speed_m_s: float | None
    velocity_gap_m_s: float | None
    auxiliary_penalty: float


def ballistic_launch_guidance(
    *,
    launch_position_m: tuple[float, float, float],
    launch_velocity_m_s: tuple[float, float, float],
    target_position_m: tuple[float, float, float],
    complete_clean_episode: bool,
) -> BallisticLaunchGuidance:
    """Bounded terminal cue in a goal-aligned coordinate frame (+X forward).

    Use the measured state at the end of the first contiguous foot-contact
    event; the caller must prove that event and clean complete-episode status.
    Do not substitute a requested launch velocity or predicted ball position.
    Missing contact is handled by the caller as a failed/missing observation,
    not by inventing a launch. Rejected episodes get the maximum cost (-4).

    Estimate flight time from current forward speed. For 0.05..3 seconds and
    forward speed >=1 m/s, derive lateral and vertical velocity needed under
    gravity 9.81 m/s². Cost is -0.5 * min(velocity gap, 8). Different outcome
    velocities therefore remain distinguishable even when all balls land
    before the goal and their measured crossing heights are nearly identical.
    This intentionally changes the learning objective; it is not potential
    shaping, proof of a future goal, or a replacement for actual precision.
    """
    for vector in (launch_position_m, launch_velocity_m_s, target_position_m):
        if (
            type(vector) is not tuple
            or len(vector) != 3
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 100 for v in vector
            )
        ):
            raise ValueError("bounded finite three-component launch/target tuples required")
    if type(complete_clean_episode) is not bool:
        raise ValueError("explicit complete clean-episode flag required")
    unavailable = BallisticLaunchGuidance(False, None, None, None, None, -4.0)
    if not complete_clean_episode or launch_velocity_m_s[0] < 1.0:
        return unavailable
    flight = (target_position_m[0] - launch_position_m[0]) / launch_velocity_m_s[0]
    if not 0.05 <= flight <= 3.0:
        return unavailable
    lateral = (target_position_m[1] - launch_position_m[1]) / flight
    vertical = (target_position_m[2] - launch_position_m[2]) / flight + 4.905 * flight
    gap = math.hypot(lateral - launch_velocity_m_s[1], vertical - launch_velocity_m_s[2])
    return BallisticLaunchGuidance(True, flight, lateral, vertical, gap, -0.5 * min(gap, 8.0))
