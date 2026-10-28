"""Exact unaffected-axis preservation for an experimental SIM target teacher."""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import target29


def teacher_target_to_action(
    action: np.ndarray, proposed_target: np.ndarray, proposed_delta: np.ndarray, params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the native float32 PD target map only on teacher-modified axes.

    Return the actual target difference, not the ideal float64 proposal.
    Float32 rounding may require one or more ULPs toward the parent to keep
    the teacher's hard 0.16 rad bound; no other axis may change.
    """
    if (
        not isinstance(action, np.ndarray)
        or action.shape != (29,)
        or action.dtype != np.float32
        or not isinstance(proposed_target, np.ndarray)
        or proposed_target.shape != (29,)
        or not isinstance(proposed_delta, np.ndarray)
        or proposed_delta.shape != (29,)
        or not np.isfinite(proposed_target).all()
        or not np.isfinite(proposed_delta).all()
        or np.max(np.abs(proposed_delta)) > 0.16 + 1e-12
    ):
        raise ValueError("bounded finite teacher proposal and actual action required")
    nominal = target29(action, params)
    if not np.allclose(proposed_target - nominal, proposed_delta, atol=1e-12, rtol=0):
        raise ValueError("proposal target and delta must bind to the actual parent")
    mask = proposed_delta != 0
    new_action = action.copy()
    scale = np.asarray(params.action_scale, dtype=np.float32)
    offset = np.asarray(params.action_offset, dtype=np.float32)
    new_action[mask] = (
        (proposed_target[mask].astype(np.float32) - offset[mask]) / scale[mask]
    ).astype(np.float32)
    for _ in range(8):
        actual = target29(new_action, params) - nominal
        over = np.abs(actual) > 0.16 + 1e-12
        if not over.any():
            break
        new_action[over] = np.nextafter(new_action[over], action[over])
    actual = target29(new_action, params) - nominal
    if np.any(new_action[~mask] != action[~mask]) or np.max(np.abs(actual)) > 0.16 + 1e-12:
        raise ValueError("actual teacher action exceeded bound or changed parent-only axes")
    if not np.allclose(actual, proposed_delta, atol=1e-6, rtol=0):
        raise ValueError("actual target cannot represent bounded proposal")
    return new_action, actual
