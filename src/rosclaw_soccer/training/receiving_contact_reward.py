"""Measured receiving reward adjustment, not proof of a completed team pass."""

import numpy as np


def receiving_contact_adjustment(
    *,
    time: np.ndarray,
    receiving: np.ndarray,
    foot_contact: np.ndarray,
    owns_ball: np.ndarray,
    speed_before: np.ndarray,
    speed_after: np.ndarray,
    foot_distance: np.ndarray,
    directed_reward: np.ndarray,
) -> np.ndarray:
    """Replace send-ball credit by slowing/briefly retaining a contacted ball.

    Inputs describe contiguous 50 Hz measured control frames. A possession label
    alone earns nothing: require a recent actual foot contact, a nearby slow
    ball and a receiving task. Credit expires after 0.3 s without another contact.
    No physics, collisions, ownership labels or safety penalties are modified.
    """
    if time.ndim != 1 or len(time) == 0:
        raise ValueError("nonempty control timeline required")
    for value in (receiving, foot_contact, owns_ball):
        if value.shape != time.shape or value.dtype != np.bool_:
            raise ValueError("explicit boolean per-frame contact/task/ownership required")
    for value in (time, speed_before, speed_after, foot_distance, directed_reward):
        if (
            value.shape != time.shape
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite aligned receiving measurements required")
    if time[0] < 0 or not np.allclose(np.diff(time), 0.02, rtol=0, atol=1e-7):
        raise ValueError("contiguous 50 Hz control measurements required")
    if any(np.any(x < 0) for x in (speed_before, speed_after, foot_distance)):
        raise ValueError("speed magnitudes and foot distance must be nonnegative")
    last = -np.inf
    held = np.zeros(len(time), bool)
    for i, contact in enumerate(foot_contact):
        if contact:
            last = float(time[i])
        held[i] = (
            owns_ball[i]
            and time[i] - last <= 0.3
            and foot_distance[i] <= 0.35
            and speed_after[i] <= 0.8
        )
    return np.where(
        receiving,
        -directed_reward
        + 0.04 * foot_contact * np.clip(speed_before - speed_after, -2, 2)
        + 0.01 * held,
        0.0,
    )
