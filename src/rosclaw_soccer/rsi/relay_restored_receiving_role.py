"""Reconstruct a SIM receiver's causal role decision from its own history."""

import numpy as np

from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent


def restored_receiving_role(
    *,
    initial_ball: np.ndarray,
    initial_home: np.ndarray,
    pass_goal: np.ndarray,
    current_q: np.ndarray,
    current_v: np.ndarray,
    past_ball_velocities: np.ndarray,
    lateral_offset_m: float = 0.12,
):
    arrays = (
        (initial_ball, (2,)),
        (initial_home, (2,)),
        (pass_goal, (3,)),
        (current_q, (43,)),
        (current_v, (41,)),
    )
    if (
        any(
            not isinstance(x, np.ndarray)
            or x.shape != shape
            or x.dtype != np.float64
            or not np.isfinite(x).all()
            for x, shape in arrays
        )
        or not isinstance(past_ball_velocities, np.ndarray)
        or past_ball_velocities.ndim != 2
        or past_ball_velocities.shape[1] != 2
        or not len(past_ball_velocities)
        or past_ball_velocities.dtype != np.float64
        or not np.isfinite(past_ball_velocities).all()
        or abs(np.linalg.norm(current_q[3:7]) - 1) > 1e-5
        or type(lateral_offset_m) is not float
        or not np.isfinite(lateral_offset_m)
        or abs(lateral_offset_m) > 0.4
    ):
        raise ValueError(
            "finite actual own state and completed causal ball-velocity history required"
        )
    forward = pass_goal[:2] - initial_ball
    length = float(np.linalg.norm(forward))
    if length < 1e-8:
        raise ValueError("nondegenerate actual pass direction required")
    forward = forward / length
    lateral = np.array([-forward[1], forward[0]])
    # The recorded shared setup's ready anchor remains fixed at +0.12m;
    # the course's interception offset is a separate, explicit role input.
    anchor = initial_ball + 4 * forward + 0.12 * lateral
    seen = bool(np.any(past_ball_velocities @ forward > 0.4))
    qw, x, y, z = current_q[3:7]
    yaw = float(np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z)))
    return receiving_intent(
        root_xy=current_q[:2],
        yaw=yaw,
        ball_xy=current_q[36:38],
        ball_velocity_xy=current_v[35:37],
        home_xy=initial_home,
        pass_direction=forward,
        launch_seen=seen,
        ready_anchor_xy=anchor,
        lateral_offset_m=lateral_offset_m,
    )
