"""Explicit training-only label scope; no phase/teacher metadata enters actors."""

import numpy as np


def correction_window(
    *,
    first_foot_frame: int | None,
    first_nonfoot_frame: int | None,
    teacher_handoff_frame: int | None,
    parent_qualified: bool,
) -> np.ndarray:
    """Only pre-contact, pre-handoff teacher approach before its deadline.

    A stopped/timed-out shadow teacher is not a successful kicking expert.
    Existing parent successes must use parent actions, not contradictory advice.
    """
    if type(parent_qualified) is not bool:
        raise ValueError("explicit original parent result required")
    events = (first_foot_frame, first_nonfoot_frame, teacher_handoff_frame)
    if any(x is not None and (type(x) is not int or not 0 <= x < 300) for x in events):
        raise ValueError("observed native global frame or absent event required")
    mask = np.zeros(300, dtype=np.bool_)
    if parent_qualified:
        return mask
    end = min([91] + [x for x in events if x is not None])
    if end > 20:
        mask[20:end] = True
    return mask


def success_memory_window(
    *, first_foot_frame: int, qualified: bool, full_retention: bool
) -> np.ndarray:
    """Empirical successful contact memory, or full old successful continuation."""
    if (
        type(first_foot_frame) is not int
        or not 20 <= first_foot_frame <= 239
        or type(qualified) is not bool
        or not qualified
        or type(full_retention) is not bool
    ):
        raise ValueError("complete original successful foot outcome required")
    mask = np.zeros(300, dtype=np.bool_)
    if full_retention:
        mask[20:] = True
    else:
        mask[max(20, first_foot_frame - 20) : min(300, first_foot_frame + 5)] = True
    return mask
