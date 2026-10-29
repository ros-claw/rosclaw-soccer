"""Bounded potential differences from actual SIM physical receiver states."""

import numpy as np


def receiver_potential_shaping(qpos, qvel, *, receiver_q, receiver_v, ball_q, ball_v, gamma=0.99):
    """Shape a complete action segment; terminal potential is explicitly zero.

    Rows are states BEFORE successive actions, including the actual initial
    boundary. No future outcome, synthetic contact, or motor authority is used.
    """
    q, v = np.asarray(qpos), np.asarray(qvel)
    n = len(q)
    if (
        q.ndim != 2
        or v.ndim != 2
        or n < 1
        or len(v) != n
        or any(type(i) is not int for i in (receiver_q, receiver_v, ball_q, ball_v))
        or not (0 <= receiver_q <= q.shape[1] - 2 and 0 <= ball_q <= q.shape[1] - 2)
        or not (0 <= receiver_v <= v.shape[1] - 2 and 0 <= ball_v <= v.shape[1] - 2)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or not np.isfinite(gamma)
        or not 0 < gamma <= 1
    ):
        raise ValueError("finite actual pre-action physical states and valid layout required")
    distance = np.linalg.norm(q[:, receiver_q : receiver_q + 2] - q[:, ball_q : ball_q + 2], axis=1)
    speed = np.linalg.norm(v[:, receiver_v : receiver_v + 2] - v[:, ball_v : ball_v + 2], axis=1)
    potential = 0.5 * np.exp(-distance / 0.65 - speed / 1.5)
    next_potential = np.r_[potential[1:], 0.0]
    return gamma * next_potential - potential
