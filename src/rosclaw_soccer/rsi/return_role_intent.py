"""Causal SIM-only role signal for a passer facing a teammate's return."""

import numpy as np

from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent


def return_strike_seen(forces, qvel, *, ball_v, return_direction):
    """Use simultaneous measured receiver-foot force and ball velocity only."""
    f, v, direction = map(np.asarray, (forces, qvel, return_direction))
    if (
        f.ndim != 3
        or f.shape[1:] != (2, 3)
        or len(f) == 0
        or v.ndim != 2
        or len(v) != len(f)
        or type(ball_v) is not int
        or ball_v < 0
        or ball_v + 2 > v.shape[1]
        or direction.shape != (2,)
        or not all(np.isfinite(x).all() for x in (f, v, direction))
        or np.any(f < 0)
        or abs(float(np.linalg.norm(direction)) - 1) > 1e-10
    ):
        raise ValueError("finite aligned measured return-force and velocity evidence required")
    return bool(
        np.any((np.max(f[:, 1, :2], axis=1) > 10) & (v[:, ball_v : ball_v + 2] @ direction >= 1.0))
    )


def return_receive_command(q, v, *, return_direction, launch_seen):
    """Return a bounded body-frame command from current own state, not future data."""
    q, v, direction = map(np.asarray, (q, v, return_direction))
    if (
        q.shape != (43,)
        or v.shape != (41,)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or type(launch_seen) is not bool
    ):
        raise ValueError("finite canonical own state and explicit causal role history required")
    if not launch_seen:
        return np.zeros(3, np.float32)
    w, x, y, z = q[3:7]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    result = receiving_intent(
        root_xy=q[:2],
        yaw=yaw,
        ball_xy=q[36:38],
        ball_velocity_xy=v[35:37],
        home_xy=q[:2].copy(),
        pass_direction=direction,
        launch_seen=True,
        ready_anchor_xy=q[:2].copy(),
        mode="velocity-match-v4",
    )
    return np.asarray(result.command_body, np.float32)
