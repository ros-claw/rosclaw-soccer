"""Causal ball-relative features from a measured canonical SIM state.

This is observation construction only: it does not command a robot, predict
future contact, or attest to a reception. The state is ordered according to
``NativeVelocityLayout.canonical`` and must describe one instant.
"""

import numpy as np


def receiver_ball_observation(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return body-frame ball position and its rotating-frame time derivative.

    Native free-joint translation velocities are world-frame; the robot's
    angular velocity is body-frame. Subtracting omega cross position is needed
    for the derivative in the rotating body frame, not just for a rotated
    world-relative velocity.
    """
    if (
        type(q) is not np.ndarray
        or type(v) is not np.ndarray
        or q.shape != (43,)
        or v.shape != (41,)
        or not np.issubdtype(q.dtype, np.floating)
        or not np.issubdtype(v.dtype, np.floating)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("finite floating canonical receiver and ball state required")
    w, x, y, z = (float(n) for n in q[3:7])
    if abs(w * w + x * x + y * y + z * z - 1.0) > 2e-5:
        raise ValueError("unit receiver quaternion required")
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    position = rotation.T @ (q[36:39] - q[:3])
    velocity = rotation.T @ (v[35:38] - v[:3]) - np.cross(v[3:6], position)
    result = np.concatenate((position, velocity))
    if not np.isfinite(result).all():
        raise ValueError("finite relative observation required")
    return result
