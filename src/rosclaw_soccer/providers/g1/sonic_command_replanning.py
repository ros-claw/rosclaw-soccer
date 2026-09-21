"""Opt-in simulation scheduling for the frozen low-latency SONIC planner.

This decides when to refresh a motion reference, not whether motion is safe.
Commands must already have passed the world's navigation guards. It neither
changes command limits nor owns a simulator, body, permit or executor.
"""

import math


def command_event_requires_replan(
    *,
    frame: int,
    last_plan_frame: int,
    command: tuple[float, float, float],
    last_planned_command: tuple[float, float, float],
) -> bool:
    """Fixed research protocol: meaningful change, at least 100 ms since plan.

    Compare with the last successfully planned command, not the previous tick:
    gradual changes must accumulate. This is not a physical latency guarantee.
    """
    if (
        type(frame) is not int
        or type(last_plan_frame) is not int
        or not 0 <= last_plan_frame <= frame <= 3000
    ):
        raise ValueError("ordered bounded local planner frames required")
    for value in (command, last_planned_command):
        if (
            type(value) is not tuple
            or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or math.hypot(value[0], value[1]) > 0.700000001
            or abs(value[2]) > 1.500000001
        ):
            raise ValueError("finite immutable commands inside the original envelope required")
    return frame - last_plan_frame >= 5 and (
        math.hypot(command[0] - last_planned_command[0], command[1] - last_planned_command[1])
        >= 0.15
        or abs(command[2] - last_planned_command[2]) >= 0.3
    )
