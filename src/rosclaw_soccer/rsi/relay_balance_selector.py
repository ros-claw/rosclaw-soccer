"""Offline development choice and score, not a safety gate or motor policy."""

import numpy as np


def decision_rewards(success: np.ndarray, failure: np.ndarray, kick: np.ndarray) -> np.ndarray:
    if any(a.shape != (4,) or a.dtype != np.bool_ for a in (success, failure, kick)) or np.any(
        success & failure
    ):
        raise ValueError("four paired valid measured outcomes required")
    return (
        2.0 * success.astype(np.float32)
        + 0.25 * kick.astype(np.float32)
        - 2.0 * failure.astype(np.float32)
    )


def select_with_parent_fallback(values: np.ndarray, nearest_distance: float) -> tuple[int, str]:
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or not np.isfinite(nearest_distance)
        or nearest_distance < 0
    ):
        raise ValueError("finite four-action values and measured support distance required")
    if nearest_distance > 2.0:
        return 0, "OUTSIDE_TRAIN_SUPPORT"
    action = int(np.argmax(values))
    if action == 0 or values[action] < values[0] + 0.25:
        return 0, "PARENT_OR_SMALL_MARGIN"
    return action, "RESEARCH_TEACHER_CHOICE"
