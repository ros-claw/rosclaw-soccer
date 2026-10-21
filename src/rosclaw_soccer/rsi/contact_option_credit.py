"""Measured finite-option tail credit for contact actions; no reward invention.

Tail rewards MUST come from the actual continuously simulated fixed recovery
policy. There is no critic bootstrap after this finite task ends. This does not
change the old contact24 time-limit semantics or original formal H300 exam.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.native_contact_ppo import course_gae


def recovery_terminal_reward(result: dict[str, Any]) -> float:
    """New task definition, from measured whole-option safety and 1.2s outcome.

    The caller places this at the LAST actual recovery control, exactly once.
    Not a reward for an assumed future kick and not the original formal exam.
    """
    if not result["stable"] or result["forbidden"]:
        return -20.0
    post = result["postcontact"]
    if not post["post_contact_quality_observed"] or not post["correct_foot_first"]:
        return 0.0
    progress = float(post["goal_progress_1p2s_m"])
    lateral = float(post["goal_lateral_1p2s_m"])
    if not math.isfinite(progress) or not math.isfinite(lateral):
        raise ValueError("observed actual finite ball displacement required")
    dense = min(max(progress, 0.0), 1.0) * math.exp(-((lateral / (0.3 * max(progress, 0.1))) ** 2))
    quality = progress >= 1.0 and abs(lateral) <= 0.3 * progress
    return 2.0 * dense + 10.0 * quality


def measured_tail_return(
    reward: NDArray[np.float32],
    valid: NDArray[np.bool_],
    terminal: NDArray[np.bool_],
) -> NDArray[np.float32]:
    if reward.ndim != 2 or not 1 <= reward.shape[0] <= 60 or reward.shape[1] < 1:
        raise ValueError("complete bounded time-major recovery tail required")
    if reward.dtype != np.float32 or not np.isfinite(reward).all():
        raise ValueError("finite actual float32 tail rewards required")
    for mask in (valid, terminal):
        if mask.shape != reward.shape or mask.dtype != np.bool_:
            raise ValueError("aligned boolean tail masks required")
    if (terminal & ~valid).any() or (reward[~valid] != 0).any():
        raise ValueError("no rewards or terminal events after physical execution ends")
    for world in range(reward.shape[1]):
        count = int(valid[:, world].sum())
        if not np.array_equal(valid[:, world], np.arange(len(valid)) < count):
            raise ValueError("no in-tail reset, restart, or observation holes")
        events = np.flatnonzero(terminal[:, world])
        if (count == 0 and len(events)) or (count and not np.array_equal(events, [count - 1])):
            raise ValueError("actual finite tail must end exactly once at last valid step")
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    for t in range(len(reward) - 1, -1, -1):
        carry = np.where(valid[t], reward[t] + np.float32(0.99) * carry, 0).astype(np.float32)
    return carry


def contact_option_gae(
    reward: NDArray[np.float32],
    value: NDArray[np.float32],
    next_value: NDArray[np.float32],
    terminated: NDArray[np.bool_],
    handoff: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    tail_reward: NDArray[np.float32],
    tail_valid: NDArray[np.bool_],
    tail_terminal: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    future = measured_tail_return(tail_reward, tail_valid, tail_terminal)
    if handoff.shape != reward.shape or handoff.dtype != np.bool_ or reward.shape[1] != len(future):
        raise ValueError("same contact and recovery world identities required")
    if not np.array_equal(handoff.sum(axis=0), tail_valid.any(axis=0).astype(int)):
        raise ValueError("only completed contact handoffs have actual recovery tails")
    if ((terminated | handoff).sum(axis=0) != 1).any():
        raise ValueError("each finite contact option must terminate or hand off once")
    for world in range(reward.shape[1]):
        count = int(valid[:, world].sum())
        if not np.array_equal(valid[:, world], np.arange(len(valid)) < count):
            raise ValueError("contact execution must be one contiguous physical prefix")
        events = np.flatnonzero((terminated | handoff)[:, world])
        if count == 0 or not np.array_equal(events, [count - 1]):
            raise ValueError("contact boundary must occur at its last executed control")
        if handoff[:, world].any() and count != 24:
            raise ValueError("this option hands off only after all24 contact controls")
    if (reward[~valid] != 0).any():
        raise ValueError("invalid contact samples cannot carry reward")
    boundary_value = next_value.copy()
    for world in range(len(future)):
        boundary_value[handoff[:, world], world] = future[world]
    # The handoff ends the learned contact segment, not the physical task.
    # course_gae bootstraps this one boundary from MEASURED future return and
    # breaks lambda carry there. Internal next values remain critic estimates.
    return course_gae(reward, value, boundary_value, terminated, handoff, valid)
