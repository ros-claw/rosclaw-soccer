"""Sparse contact feedback for offline SIM research, not soccer success."""

import numpy as np


def precontact_feedback(ball_forces: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reward first clean foot and penalize first nonfoot aggregate contact.

    The nonfoot channel contains legal and illegal body parts. No force is
    assigned to individual geometries; no possession or safety is implied.
    """
    force = np.asarray(ball_forces)
    if (
        force.ndim != 3
        or force.shape[1:] != (10, 3)
        or len(force) == 0
        or not np.isfinite(force).all()
        or np.any(force < 0)
    ):
        raise ValueError("finite native 10-substep receiver force history required")
    flat = force.reshape(-1, 3)
    foot = np.flatnonzero((flat[:, :2] > 10).any(axis=1))
    other = np.flatnonzero(flat[:, 2] > 1)
    first_foot = int(foot[0]) if foot.size else None
    first_other = int(other[0]) if other.size else None
    reward = np.zeros(len(force), np.float32)
    if first_foot is not None and (first_other is None or first_foot < first_other):
        reward[first_foot // 10] += 0.5
    if first_other is not None:
        reward[first_other // 10] -= 1.0
    return reward, dict(
        first_foot_substep=first_foot,
        first_nonfoot_substep=first_other,
        nonfoot_is_not_legality=True,
    )
