"""Recompute original sustained reception from saved shared-world physics."""

import numpy as np


def original_reception_readback(
    qpos: np.ndarray,
    qvel: np.ndarray,
    forces: np.ndarray,
    body_failed: np.ndarray,
    *,
    receiver_q: int,
    receiver_v: int,
    ball_q: int,
    ball_v: int,
    excluded_contact_ever: bool,
) -> dict:
    """Use post-control samples; excluded contact timing is not reconstructed."""
    n = len(qpos)
    if (
        qpos.shape != (n, 10, 79)
        or qvel.shape != (n, 10, 76)
        or forces.shape != (n, 10, 2, 3)
        or body_failed.shape != (n, 2)
        or body_failed.dtype != np.bool_
        or n < 1
        or any(not np.isfinite(x).all() for x in (qpos, qvel, forces))
        or (forces < 0).any()
        or type(excluded_contact_ever) is not bool
        or any(type(x) is not int for x in (receiver_q, receiver_v, ball_q, ball_v))
        or not 0 <= receiver_q <= 72
        or not 0 <= ball_q <= 72
        or not 0 <= receiver_v <= 70
        or not 0 <= ball_v <= 70
        or set(range(receiver_q, receiver_q + 7)) & set(range(ball_q, ball_q + 7))
        or set(range(receiver_v, receiver_v + 6)) & set(range(ball_v, ball_v + 6))
    ):
        raise ValueError("complete finite native shared-world histories required")
    q, v = qpos[:, -1], qvel[:, -1]
    root = q[:, receiver_q : receiver_q + 7]
    ball = q[:, ball_q : ball_q + 7]
    if not np.allclose(np.linalg.norm(root[:, 3:7], axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("unit observed orientation required")
    delta = ball[:, :3] - root[:, :3]
    w, x, y, z = root[:, 3:7].T
    forward = (
        (1 - 2 * (y * y + z * z)) * delta[:, 0]
        + 2 * (x * y + w * z) * delta[:, 1]
        + 2 * (x * z - w * y) * delta[:, 2]
    )
    contact_seen = np.maximum.accumulate(forces[:, :, 1, :].max(axis=(1, 2)) > 1)
    tests = (
        contact_seen,
        np.linalg.norm(delta[:, :2], axis=1) <= 0.65,
        np.linalg.norm(v[:, ball_v : ball_v + 2] - v[:, receiver_v : receiver_v + 2], axis=1)
        <= 0.6,
        ball[:, 2] <= 0.3,
        forward >= 0.05,
        root[:, 2] >= 0.65,
        1 - 2 * (x * x + y * y) >= 0.9,
        ~body_failed.any(axis=1),
    )
    good = np.logical_and.reduce(tests)
    streak = 0
    first = None
    for index, value in enumerate(good):
        streak = streak + 1 if value else 0
        if streak >= 15:
            first = index + 1
            break
    return {
        "reconstruction_exact": not excluded_contact_ever,
        "first_ready_control": None if excluded_contact_ever else first,
        "excluded_contact_timing_unknown": excluded_contact_ever,
    }
