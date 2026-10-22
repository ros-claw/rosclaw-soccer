"""Native incoming-pass diagnostics, separate from return-pass success.

Measures the ball after actual A foot contact and BEFORE B's first contact,
so B cannot improve the attributed delivery error by subsequently dribbling.
This geometry diagnostic alone never qualifies a received pass or full game.
"""

from typing import Any

import numpy as np


def incoming_delivery(
    ball_positions: np.ndarray, ball_forces: np.ndarray, receiving_target: np.ndarray
) -> dict[str, Any]:
    n = len(ball_positions)
    if (
        ball_positions.shape != (n, 3)
        or n < 1
        or ball_forces.shape != (n, 2, 3)
        or receiving_target.shape != (3,)
        or any(not np.isfinite(a).all() for a in (ball_positions, ball_forces, receiving_target))
        or (ball_forces < 0).any()
    ):
        raise ValueError(
            "finite native ball positions, per-player force magnitudes and target required"
        )
    source = np.flatnonzero(ball_forces[:, 0, :2].max(axis=1) > 10)
    receiver = np.flatnonzero(ball_forces[:, 1, :].max(axis=1) > 1)
    first_a = int(source[0]) if len(source) else None
    first_b = int(receiver[0]) if len(receiver) else None
    end = first_b if first_b is not None else n
    observable = first_a is not None and first_a < end
    error = (
        float(np.linalg.norm(ball_positions[first_a:end] - receiving_target, axis=1).min())
        if observable
        else None
    )
    return dict(
        A_first_foot_strike_substep=first_a,
        B_first_ball_contact_substep=first_b,
        incoming_error_observed=observable,
        incoming_target_error_3d_before_B_contact_m=error,
        incoming_window_end_exclusive=end,
        incoming_target=receiving_target.tolist(),
        incoming_diagnostic_is_received_pass_qualification=False,
    )
