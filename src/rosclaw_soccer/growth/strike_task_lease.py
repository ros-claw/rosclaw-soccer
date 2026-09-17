"""Release stale preparation authority without interrupting an executing skill."""

from rosclaw_soccer.growth.role_self_model import TacticalIntent


def should_release_strike_task(
    *,
    intent: TacticalIntent,
    owns_ball: bool,
    motor_executing: bool,
    phase_active: bool,
) -> bool:
    """A lost-ball reacquisition/recovery task supersedes idle shot preparation.

    Does not grant a new strike, infer contact, or reset motor lifecycle. Active
    whole-body motion and a running strike phase retain their own termination
    contract. A held ball or a continuing SHOOT task keeps the original lease.
    """
    if not isinstance(intent, TacticalIntent) or any(
        type(x) is not bool for x in (owns_ball, motor_executing, phase_active)
    ):
        raise ValueError("explicit current task, ownership and execution state required")
    return (
        not owns_ball
        and not motor_executing
        and not phase_active
        and intent
        in {
            TacticalIntent.RECEIVE,
            TacticalIntent.PRESS,
            TacticalIntent.INTERCEPT,
            TacticalIntent.RECOVER,
        }
    )
