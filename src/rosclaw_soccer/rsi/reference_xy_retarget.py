"""Training-only rigid reference retarget, never a motor/state correction.

Choose an exogenous reference strike from qualified FK, then translate pelvis
and torso references together. Measured states, joints, NN weights and PD stay
untouched. The initial target and its velocity are exactly unchanged.
"""

import numpy as np


def ball_strike_xy_plan(
    *,
    capsule_centers: np.ndarray,
    capsule_axes: np.ndarray,
    capsule_half_length: float,
    capsule_radius: float,
    ball_radius: float,
    initial_ball: np.ndarray,
    goal: np.ndarray,
) -> dict[str, object]:
    if not isinstance(capsule_centers, np.ndarray) or capsule_centers.ndim != 2:
        raise ValueError("typed reference capsule trajectory required")
    n = len(capsule_centers)
    for a, shape in (
        (capsule_centers, (n, 3)),
        (capsule_axes, (n, 3)),
        (initial_ball, (3,)),
        (goal, (3,)),
    ):
        if (
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype.kind != "f"
            or not np.isfinite(a).all()
        ):
            raise ValueError("finite typed FK capsule/ball/goal geometry required")
    if not 100 <= n <= 300 or not np.allclose(
        np.linalg.norm(capsule_axes, axis=1), 1, atol=1e-6, rtol=0
    ):
        raise ValueError("100..300 reference frames and unit capsule axes required")
    if any(
        type(v) is not float or not np.isfinite(v) or not 0 < v < 1
        for v in (capsule_half_length, capsule_radius, ball_radius)
    ):
        raise ValueError("explicit positive bounded compiled capsule/ball sizes required")
    direction = goal[:2] - initial_ball[:2]
    distance = float(np.linalg.norm(direction))
    if distance < 1e-6:
        raise ValueError("nondegenerate target direction required")
    direction /= distance
    signs = np.where(capsule_axes[:, :2] @ direction >= 0, 1.0, -1.0)
    tips = capsule_centers + capsule_axes * (signs * capsule_half_length)[:, None]
    velocity = np.gradient(tips[:, :2] @ direction, 0.02)
    eligible = (
        (np.arange(n) >= 50)
        & (np.arange(n) < n - 30)
        & (abs(tips[:, 2] - initial_ball[2]) <= ball_radius + capsule_radius)
        & (velocity >= 0.1)
    )
    if not eligible.any():
        raise ValueError("no admitted forward low-foot strike in the declared reference window")
    strike = int(np.argmax(np.where(eligible, velocity, -np.inf)))
    contact_center = tips[strike, :2] + direction * (capsule_radius + ball_radius)
    requested = initial_ball[:2] - contact_center
    norm = float(np.linalg.norm(requested))
    applied = requested * min(1.0, 0.3 / max(norm, 1e-12))
    u = np.minimum(np.arange(n, dtype=np.float64) * 0.02, 1.0)
    fraction = 3 * u**2 - 2 * u**3
    rate = 6 * u * (1 - u)  # one-second ramp; derivative in 1/s
    offset = np.zeros((n, 3), dtype=np.float64)
    offset_velocity = np.zeros_like(offset)
    offset[:, :2] = fraction[:, None] * applied
    offset_velocity[:, :2] = rate[:, None] * applied
    return dict(
        strike_reference_frame=strike,
        strike_reference_forward_velocity_mps=float(velocity[strike]),
        requested_offset_xy_m=requested.tolist(),
        applied_offset_xy_m=applied.tolist(),
        translation_norm_limited=norm > 0.3,
        maximum_translation_m=0.3,
        ramp_seconds=1.0,
        offset=offset,
        offset_velocity=offset_velocity,
    )
