"""Explicit nested-interface warm starts, not a new motion authority.

The A1 interface contains A0's twelve leg coordinates with seventeen zero
upper-body residuals. This algebraic mapping does not by itself prove physical
equivalence: the caller must replay the same state, controller and physics.
No equivalent mapping is assumed between unrelated locomotion foundations.
"""

from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def embed_leg_oracle_in_body(schedule: ReceivingOracleSchedule) -> ReceivingOracleSchedule:
    """Preserve timing, identity and leg knots; append exactly zero upper residual.

    Use the physically qualified embedded solution as an incumbent, so a
    larger-dimensional random search cannot erase a known feasible action.
    This changes search initialization, never the capture/safety examination.
    """
    if not isinstance(schedule, ReceivingOracleSchedule):
        raise ValueError("typed bounded leg schedule required")
    schedule.__post_init__()
    if schedule.substrate != "A0_leg12":
        raise ValueError("only the explicit A0-to-A1 nested interface is supported")
    return ReceivingOracleSchedule(
        schedule.agent_id,
        "A1_body29",
        schedule.start_frame,
        schedule.knot_frames,
        tuple(row + (0.0,) * 17 for row in schedule.knots),
    )
