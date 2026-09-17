"""Admission to the existing SIM receiving stabilization, never pass credit."""

import math

from rosclaw_soccer.growth.role_self_model import TacticalIntent


def admit_loose_ball_capture(
    *,
    intent: TacticalIntent,
    effector_code: int,
    contact_force_n: float,
    fresh_contact: bool,
    receive_lease_present: bool,
    motor_busy: bool,
    capture_busy: bool,
) -> bool:
    """Require a new measured foot touch during an actual acquisition task.

    Caller owns contact history and all motor guards. Never interrupt a live
    teammate reception, full-body/strike option or ongoing capture, and never
    restart the capture clock on every frame of persistent foot contact.
    """
    if (
        not isinstance(intent, TacticalIntent)
        or type(effector_code) is not int
        or effector_code not in range(5)
        or type(contact_force_n) not in (int, float)
        or not math.isfinite(contact_force_n)
        or contact_force_n < 0
        or any(
            type(v) is not bool
            for v in (fresh_contact, receive_lease_present, motor_busy, capture_busy)
        )
    ):
        raise ValueError("explicit measured loose-ball capture admission required")
    return (
        intent in {TacticalIntent.RECEIVE, TacticalIntent.PRESS, TacticalIntent.INTERCEPT}
        and effector_code in (1, 2)
        and contact_force_n > 1
        and fresh_contact
        and not receive_lease_present
        and not motor_busy
        and not capture_busy
    )
