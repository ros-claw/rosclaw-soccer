"""SIM contact ordering, measured over one common observed physical horizon."""

import numpy as np


def return_prepare_contact(ball_forces, body_failed, *, handoff, outgoing_foot_strike):
    n = len(ball_forces)
    if (
        not isinstance(ball_forces, np.ndarray)
        or ball_forces.shape != (n, 10, 2, 3)
        or n == 0
        or not np.isfinite(ball_forces).all()
        or (ball_forces < 0).any()
        or not isinstance(body_failed, np.ndarray)
        or body_failed.shape != (n, 2)
        or body_failed.dtype != np.bool_
        or type(outgoing_foot_strike) is not bool
        or (handoff is not None and (type(handoff) is not int or not 0 <= handoff < n))
    ):
        raise ValueError("bounded native horizon, handoff and foot evidence required")
    f = ball_forces.reshape(-1, 2, 3)
    b_foot = a_foot = a_other = None
    if handoff is not None:
        candidates = np.flatnonzero(np.max(f[handoff * 10 :, 1, :2], axis=1) > 10)
        if len(candidates):
            b_foot = handoff * 10 + int(candidates[0])
            feet = np.flatnonzero(np.max(f[b_foot:, 0, :2], axis=1) > 10)
            others = np.flatnonzero(f[b_foot:, 0, 2] > 1)
            a_foot = b_foot + int(feet[0]) if len(feet) else None
            a_other = b_foot + int(others[0]) if len(others) else None
    return {
        "observed_controls": n,
        "a_failed": bool(body_failed[:, 0].any()),
        "b_failed": bool(body_failed[:, 1].any()),
        "post_handoff_b_foot_substep": b_foot,
        "actual_outgoing_foot_strike_reported": outgoing_foot_strike,
        "a_foot_substep": a_foot,
        "a_nonfoot_substep": a_other,
        "a_foot_before_nonfoot": bool(a_foot is not None and (a_other is None or a_foot < a_other)),
        "a_clean_foot_contact": bool(
            a_foot is not None and a_other is None and not body_failed.any()
        ),
        "possession_proven": False,
    }
