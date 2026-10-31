"""Measured nearest-player context for simulation-only policy experiments.

The caller supplies observations; this module does not connect to robots or
grant control. Perfect simulator state is privileged and cannot be presented
as a real-world perception capability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def nearest_player_context(
    self_position_xy: ArrayLike,
    self_velocity_xy: ArrayLike,
    self_yaw: float,
    other_ids: Sequence[str],
    other_positions_xy: ArrayLike,
    other_velocities_xy: ArrayLike,
) -> NDArray[np.float64]:
    """Nearest player's relative position and velocity in the local yaw frame.

    Returns [present, dx, dy, dvx, dvy]. Exact-distance ties use agent ID,
    independent of input roster order. An empty roster returns all zeros.
    No team identity, learned policy or physical deployment is implied.
    """
    pos = np.asarray(self_position_xy, dtype=np.float64)
    vel = np.asarray(self_velocity_xy, dtype=np.float64)
    others = np.asarray(other_positions_xy, dtype=np.float64)
    other_vel = np.asarray(other_velocities_xy, dtype=np.float64)
    n = len(other_ids)
    if (
        pos.shape != (2,)
        or vel.shape != (2,)
        or others.shape != (n, 2)
        or other_vel.shape != (n, 2)
        or any(not isinstance(name, str) or not name for name in other_ids)
        or len(set(other_ids)) != n
        or not np.isfinite(self_yaw)
        or not all(np.isfinite(x).all() for x in (pos, vel, others, other_vel))
    ):
        raise ValueError("finite aligned local player states and unique IDs required")
    if not n:
        return np.zeros(5, dtype=np.float64)
    delta = others - pos
    distances = np.einsum("ij,ij->i", delta, delta)
    index = min(range(n), key=lambda i: (float(distances[i]), other_ids[i]))
    relative_velocity = other_vel[index] - vel
    c, s = np.cos(self_yaw), np.sin(self_yaw)
    rotation = np.array([[c, s], [-s, c]])
    return np.r_[1.0, rotation @ delta[index], rotation @ relative_velocity]
