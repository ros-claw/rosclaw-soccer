"""Credit for fixed-duration navigation decisions in one continuous world.

One held action is one PPO sample, not ten independent decisions. Reward is
discounted inside each physical block. The block-level learner must use
gamma=frame_gamma**hold_frames; the final partial block is terminal.
"""

from __future__ import annotations

import math

import numpy as np


def navigation_block_credit(
    rewards: np.ndarray,
    decision_frames: np.ndarray,
    applied: np.ndarray,
    *,
    hold_frames: int,
    frame_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate measured frames, keeping guarded-out blocks distinguishable."""
    if (
        not isinstance(rewards, np.ndarray)
        or rewards.ndim != 1
        or not 1 <= len(rewards) <= 4096
        or rewards.dtype.kind != "f"
        or not np.isfinite(rewards).all()
        or not isinstance(applied, np.ndarray)
        or applied.shape != rewards.shape
        or applied.dtype != np.bool_
        or not isinstance(decision_frames, np.ndarray)
        or decision_frames.ndim != 1
        or decision_frames.dtype.kind not in "iu"
        or not len(decision_frames)
        or type(hold_frames) is not int
        or not 1 <= hold_frames <= 25
        or type(frame_gamma) not in (int, float)
        or not math.isfinite(frame_gamma)
        or not 0.99 <= frame_gamma < 1
        or decision_frames[0] < 0
        or decision_frames[0] >= len(rewards)
        or decision_frames[-1] >= len(rewards)
        or not np.all(np.diff(decision_frames.astype(np.int64)) == hold_frames)
        or len(rewards) - decision_frames[-1] > hold_frames
    ):
        raise ValueError("finite continuous physical navigation blocks required")
    credit = np.empty(len(decision_frames), dtype=np.float32)
    influence = np.empty(len(decision_frames), dtype=np.bool_)
    discount = frame_gamma ** np.arange(hold_frames, dtype=np.float64)
    for i, start_value in enumerate(decision_frames):
        start = int(start_value)
        end = min(start + hold_frames, len(rewards))
        credit[i] = np.dot(rewards[start:end].astype(np.float64), discount[: end - start])
        influence[i] = applied[start:end].any()
    if not np.isfinite(credit).all():
        raise ValueError("navigation block credit overflow")
    return credit, influence
