"""New finite long-approach credit, not the qualified legacy24 reward/GAE.

No alive bonus or proxy kick reward: only nonpositive measured imitation/rate/
limit costs in the head, then an actual terminal ball outcome in recovery.
Fixed20ms gamma0.995 (half-life~138 controls) and lambda0.95, no task-end value.
This module is learning math, not a skill, physics or promotion certificate.
"""

from typing import Any

import numpy as np

GAMMA = np.float32(0.995)
LAMBDA = np.float32(0.95)


def approach_head_penalty(info: dict[str, Any]) -> np.ndarray:
    valid, tracking = info["valid"], info["tracking"]
    n = len(valid)
    if valid.shape != (n,) or valid.dtype != np.bool_ or tracking.shape != (n, 5):
        raise ValueError("aligned actual tracking/mask required")
    for key in ("tracking", "action_rate", "limits"):
        if not np.isfinite(info[key]).all():
            raise ValueError("finite measured head penalties required")
    if (tracking < 0).any() or (tracking > 1).any():
        raise ValueError("bounded actual exponential imitation scores required")
    for key in ("action_rate", "limits"):
        if info[key].shape != (n,) or (info[key] < 0).any():
            raise ValueError("aligned nonnegative actual action/limit costs required")
    terminal = info["terminated"]
    if terminal.shape != (n,) or terminal.dtype != np.bool_ or (terminal & ~valid).any():
        raise ValueError("failure penalty only at actual active failure required")
    reward = (
        -0.002 * (5 - tracking.sum(axis=1))
        - 0.0001 * np.minimum(info["action_rate"], 100)
        - 0.002 * np.minimum(info["limits"], 10)
        - 20 * terminal
    )
    return np.where(valid, reward, 0).astype(np.float32)


def approach_option_gae(
    reward, value, next_value, terminated, handoff, valid, *, tail_reward, tail_valid, tail_terminal
):
    shape = reward.shape
    if len(shape) != 2 or not 1 <= shape[0] <= 300 or shape[1] < 1:
        raise ValueError("bounded real time-major head required")
    for x in (reward, value, next_value):
        if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("aligned finite float32 head values required")
    for mask in (terminated, handoff, valid):
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError("aligned boolean head masks required")
    if (reward > 0).any() or (reward[~valid] != 0).any():
        raise ValueError("no alive bonus or reward for unexecuted head samples")
    if (terminated & handoff).any() or ((terminated | handoff) & ~valid).any():
        raise ValueError("exactly one actual boundary kind required")
    if (
        tail_reward.ndim != 2
        or not 1 <= len(tail_reward) <= 60
        or tail_reward.shape[1] != shape[1]
        or tail_reward.dtype != np.float32
        or not np.isfinite(tail_reward).all()
    ):
        raise ValueError("bounded actual finite recovery rewards required")
    for mask in (tail_valid, tail_terminal):
        if mask.shape != tail_reward.shape or mask.dtype != np.bool_:
            raise ValueError("aligned boolean recovery masks required")
    if (tail_terminal & ~tail_valid).any() or (tail_reward[~tail_terminal] != 0).any():
        raise ValueError("recovery reward only at actual final terminal sample")
    for w in range(shape[1]):
        n, t = int(valid[:, w].sum()), int(tail_valid[:, w].sum())
        if (
            n == 0
            or not np.array_equal(valid[:, w], np.arange(len(valid)) < n)
            or not np.array_equal(tail_valid[:, w], np.arange(len(tail_valid)) < t)
            or not np.array_equal(np.flatnonzero((terminated | handoff)[:, w]), [n - 1])
            or (t > 0) != bool(handoff[:, w].any())
            or not np.array_equal(np.flatnonzero(tail_terminal[:, w]), [t - 1] if t else [])
        ):
            raise ValueError("contiguous actual head/tail and finite boundary required")
    future = np.zeros(shape[1], dtype=np.float32)
    for t in range(len(tail_reward) - 1, -1, -1):
        future = np.where(tail_valid[t], tail_reward[t] + GAMMA * future, 0).astype(np.float32)
    next_actual = np.where(handoff, future[None, :], next_value).astype(np.float32)
    advantage = np.zeros(shape, dtype=np.float32)
    carry = np.zeros(shape[1], dtype=np.float32)
    for t in range(shape[0] - 1, -1, -1):
        delta = reward[t] + GAMMA * next_actual[t] * ~terminated[t] - value[t]
        carry = (delta + GAMMA * LAMBDA * carry * ~(terminated[t] | handoff[t])) * valid[t]
        advantage[t] = carry
    return advantage, np.where(valid, advantage + value, 0).astype(np.float32)
