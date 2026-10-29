"""Index ordered contact events in existing dual-player native trajectories."""

import numpy as np


def first_receiver_nonfoot_after_pass(forces):
    """Return 2 ms event indices; simultaneous contact is not strictly after."""
    forces = np.asarray(forces)
    if (
        forces.ndim != 3
        or forces.shape[1:] != (2, 3)
        or not np.isfinite(forces).all()
        or (forces < 0).any()
    ):
        raise ValueError("finite native forces for two players required")
    passer = np.flatnonzero(forces[:, 0, :2].max(axis=1) > 10)
    if not len(passer):
        return None, None
    candidates = np.flatnonzero(forces[:, 1, 2] > 1)
    candidates = candidates[candidates > passer[0]]
    return int(passer[0]), int(candidates[0]) if len(candidates) else None
