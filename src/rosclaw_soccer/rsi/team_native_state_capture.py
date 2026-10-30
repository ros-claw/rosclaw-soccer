"""Read-only per-player canonical states from a shared simulation boundary.

These observations omit other bodies, contact state, warmstarts and policy
memory. They are not restorable checkpoints or evidence of safe execution.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np


def capture_team_native_states(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    controllers: Sequence[Any],
    ball_qpos: int,
    ball_qvel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robot-first, ball-last qpos/qvel copies in controller order."""
    if (
        not isinstance(qpos, np.ndarray)
        or not isinstance(qvel, np.ndarray)
        or qpos.ndim != 1
        or qvel.ndim != 1
        or qpos.dtype.kind != "f"
        or qvel.dtype.kind != "f"
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or type(ball_qpos) is not int
        or type(ball_qvel) is not int
        or not 0 <= ball_qpos <= len(qpos) - 7
        or not 0 <= ball_qvel <= len(qvel) - 6
        or not 1 <= len(controllers) <= 8
    ):
        raise ValueError("bounded finite shared-world state required")
    poses, velocities = [], []
    pos_claimed = set(range(ball_qpos, ball_qpos + 7))
    vel_claimed = set(range(ball_qvel, ball_qvel + 6))
    for c in controllers:
        qi, vi = np.asarray(c.joint_qpos), np.asarray(c.joint_qvel)
        if (
            type(c.qpos_base) is not int
            or type(c.qvel_base) is not int
            or not 0 <= c.qpos_base <= len(qpos) - 7
            or not 0 <= c.qvel_base <= len(qvel) - 6
            or qi.shape != (29,)
            or vi.shape != (29,)
            or qi.dtype.kind not in "iu"
            or vi.dtype.kind not in "iu"
            or (qi < 0).any()
            or (qi >= len(qpos)).any()
            or (vi < 0).any()
            or (vi >= len(qvel)).any()
        ):
            raise ValueError("complete bounded 29-joint player layout required")
        pi = [*range(c.qpos_base, c.qpos_base + 7), *(int(i) for i in qi)]
        di = [*range(c.qvel_base, c.qvel_base + 6), *(int(i) for i in vi)]
        if (
            len(set(pi)) != 36
            or len(set(di)) != 35
            or pos_claimed.intersection(pi)
            or vel_claimed.intersection(di)
        ):
            raise ValueError("disjoint player and ball layout required")
        pos_claimed.update(pi)
        vel_claimed.update(di)
        poses.append(np.concatenate((qpos[pi], qpos[ball_qpos : ball_qpos + 7])))
        velocities.append(np.concatenate((qvel[di], qvel[ball_qvel : ball_qvel + 6])))
    return np.asarray(poses, dtype=np.float64), np.asarray(velocities, dtype=np.float64)
