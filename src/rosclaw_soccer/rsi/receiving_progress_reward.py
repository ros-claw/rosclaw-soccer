"""SIM-only dense physical feedback; never a reception-success certificate."""

import numpy as np


def receiving_progress_rewards(
    receiver_xy: np.ndarray,
    ball_xy: np.ndarray,
    ball_forces: np.ndarray,
    body_failed: np.ndarray,
    *,
    discount: float = 1.0,
) -> np.ndarray:
    """Compute potential change, first foot/nonfoot contact and fall penalties.

    Inputs are consecutive post-control measurements, not predictions or
    injected rewards. Distance can change because the ball moves, so this is
    not causal credit for the receiver. The first potential difference is zero because the
    pre-episode state is not supplied. Contact rewards are one-shot, are not
    clean-reception labels, and may coexist with a nonfoot penalty. A fall is
    penalized on its first observed control. This exploratory research signal
    must be evaluated against independent strict task outcomes.
    """
    if not isinstance(receiver_xy, np.ndarray) or receiver_xy.ndim != 2:
        raise ValueError("observed positions must be two-dimensional arrays")
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
        or not all(np.isfinite(v).all() for v in (receiver_xy, ball_xy, ball_forces))
        or (ball_forces < 0).any()
        or type(discount) is not float
        or not 0.0 < discount <= 1.0
    ):
        raise ValueError("finite aligned physical history and explicit discount required")
    distance = np.linalg.norm(ball_xy.astype(np.float64) - receiver_xy, axis=1)
    potential = -np.minimum(distance, 4.0)
    rewards = np.zeros(n, dtype=np.float64)
    rewards[1:] = discount * potential[1:] - potential[:-1]
    foot = np.flatnonzero(np.max(ball_forces[:, :, :2], axis=(1, 2)) > 10.0)
    nonfoot = np.flatnonzero(np.max(ball_forces[:, :, 2], axis=1) > 1.0)
    failures = np.flatnonzero(body_failed)
    if len(foot):
        rewards[foot[0]] += 1.0
    if len(nonfoot):
        rewards[nonfoot[0]] -= 1.0
    if len(failures):
        rewards[failures[0]] -= 5.0
    return rewards.astype(np.float32)
