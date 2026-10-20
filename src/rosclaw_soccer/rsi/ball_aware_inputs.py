"""Explicit SIM perception ABI: velocity98 + reference70 + current ball/goal9.

The last nine values are metres, m/s, metres expressed in the current pelvis
basis: ball minus pelvis position, ball minus pelvis world linear velocity,
and target minus pelvis position. Relative velocity is not a rotating-frame
coordinate derivative. No clipping, cached launch position, or future state.
"""

import numpy as np

BALL_AWARE_ABI = "soccer.velocity98.reference70.current_ball_goal9.v1"
BALL_AWARE_DIM = 177


def ball_goal9(
    *,
    pelvis_rotation: np.ndarray,
    pelvis_position: np.ndarray,
    pelvis_linear_velocity: np.ndarray,
    ball_position: np.ndarray,
    ball_linear_velocity: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    for value, shape in (
        (pelvis_rotation, (3, 3)),
        (pelvis_position, (3,)),
        (pelvis_linear_velocity, (3,)),
        (ball_position, (3,)),
        (ball_linear_velocity, (3,)),
        (target_position, (3,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype not in (np.float32, np.float64)
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite typed measured pelvis/ball/target geometry required")
    if (
        not np.allclose(pelvis_rotation.T @ pelvis_rotation, np.eye(3), rtol=0, atol=1e-5)
        or abs(np.linalg.det(pelvis_rotation) - 1) > 1e-5
    ):
        raise ValueError("proper current pelvis rotation required")
    value = np.concatenate(
        [
            pelvis_rotation.T @ delta
            for delta in (
                ball_position - pelvis_position,
                ball_linear_velocity - pelvis_linear_velocity,
                target_position - pelvis_position,
            )
        ]
    ).astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("representable finite perception9 required")
    return value
