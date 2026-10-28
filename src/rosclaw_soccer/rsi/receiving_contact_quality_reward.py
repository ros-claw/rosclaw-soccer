"""Research-only causal event feedback from recorded MuJoCo receiving traces."""

import numpy as np


def contact_quality_rewards(
    receiver_xy: np.ndarray,
    ball_xy: np.ndarray,
    ball_forces: np.ndarray,
    body_failed: np.ndarray,
) -> np.ndarray:
    """Score observed first contacts and later retention, not task success.

    Samples are post-control at 50 Hz, with ten ordered 2 ms force samples.
    A foot event exceeds 10 N; any nonfoot event exceeds 1 N. A first foot
    receives positive credit only if it strictly precedes the first nonfoot
    substep. Exactly ten controls after first foot, a small bounded retention
    bonus requires no nonfoot contact, no observed receiver failure, ball
    distance <= 0.5 m and finite-difference relative speed <= 0.6 m/s.
    A missing follow-up is censored, not penalized or credited. This is an
    offline training proxy, not clean-possession or team-success evidence.
    """
    if not isinstance(receiver_xy, np.ndarray) or receiver_xy.ndim != 2:
        raise ValueError("aligned measured histories required")
    n = len(receiver_xy)
    if (
        n < 1
        or receiver_xy.shape != (n, 2)
        or not isinstance(ball_xy, np.ndarray)
        or ball_xy.shape != (n, 2)
        or not isinstance(ball_forces, np.ndarray)
        or ball_forces.shape != (n, 10, 3)
        or not isinstance(body_failed, np.ndarray)
        or body_failed.shape != (n,)
        or body_failed.dtype != np.bool_
        or not all(np.isfinite(x).all() for x in (receiver_xy, ball_xy, ball_forces))
        or (ball_forces < 0).any()
    ):
        raise ValueError("finite aligned post-control physics and contact forces required")
    reward = np.zeros(n, dtype=np.float32)
    forces = ball_forces.reshape(n * 10, 3)
    foot = np.flatnonzero((forces[:, :2] > 10.0).any(axis=1))
    nonfoot = np.flatnonzero(forces[:, 2] > 1.0)
    failed = np.flatnonzero(body_failed)
    if len(failed):
        reward[failed[0]] -= 2.0
    if len(nonfoot):
        reward[nonfoot[0] // 10] -= 1.0
    if not len(foot):
        return reward
    first = int(foot[0])
    first_control = first // 10
    first_clean = not len(nonfoot) or first < int(nonfoot[0])
    if first_clean:
        reward[first_control] += 1.0
    follow = first_control + 10
    if (
        first_clean
        and follow < n
        and (not len(nonfoot) or int(nonfoot[0]) >= (follow + 1) * 10)
        and not body_failed[: follow + 1].any()
    ):
        relative = ball_xy.astype(np.float64) - receiver_xy.astype(np.float64)
        distance = float(np.linalg.norm(relative[follow]))
        speed = float(np.linalg.norm(relative[follow] - relative[follow - 1]) / 0.02)
        if distance <= 0.5 and speed <= 0.6:
            reward[follow] += 1.0
    return reward
