"""Offline force-window measurement of a passer's actual ball impulse.

The lookahead is for evaluation only: do not feed this result to a live actor.
"""

import numpy as np


def measure_first_pass_impulse(
    forces: np.ndarray, ball_velocity: np.ndarray, *, quiet_steps: int = 5
) -> dict:
    """Measure after a quiet interval, or report censoring/other-body interference.

    ``forces`` is native-step x (passer, receiver) x (left foot, right foot,
    other body), with the same grouping used by shared native world. A quiet
    interval is an operational boundary, not proof that no later impact occurs.
    """
    f, v = np.asarray(forces), np.asarray(ball_velocity)
    if (
        f.ndim != 3
        or f.shape[1:] != (2, 3)
        or v.shape != (len(f), 3)
        or len(f) == 0
        or not np.isfinite(f).all()
        or not np.isfinite(v).all()
        or np.any(f < 0)
        or type(quiet_steps) is not int
        or not 1 <= quiet_steps <= 50
    ):
        raise ValueError("finite native two-player forces and ball velocity required")
    own = np.max(f[:, 0, :2], axis=1)
    starts = np.flatnonzero(own > 10.0)
    if not len(starts):
        return {"status": "NO_PASSER_FOOT_CONTACT", "first_strong": None}
    first = int(starts[0])
    start = first
    while start > 0 and own[start - 1] > 1.0:
        start -= 1
    quiet = 0
    end = None
    for i in range(first + 1, len(own)):
        quiet = quiet + 1 if own[i] <= 1.0 else 0
        if quiet == quiet_steps:
            end = i - quiet_steps
            break
    if end is None:
        return {"status": "RIGHT_CENSORED", "first_strong": first, "start": start}
    confirmation = end + quiet_steps
    interfered = bool(np.any(f[start : confirmation + 1, 0, 2] > 1.0))
    interfered |= bool(np.any(f[start : confirmation + 1, 1, :] > 1.0))
    return {
        "status": "OTHER_BODY_OR_RECEIVER_CONTACT" if interfered else "QUIET_EXIT_OBSERVED",
        "start": start,
        "first_strong": first,
        "last_foot_force_step": end,
        "quiet_confirmation_step": confirmation,
        "first_strong_ball_speed_mps": float(np.linalg.norm(v[first])),
        "quiet_confirmation_ball_speed_mps": float(np.linalg.norm(v[confirmation])),
        "quiet_confirmation_ball_velocity_mps": v[confirmation].tolist(),
        "other_body_or_receiver_contact": interfered,
    }
