"""Experimental SIM-only absolute target slew filter for native G1 actions."""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import target29

ANKLE_JOINT_INDICES = (4, 5, 10, 11)


def limit_target_rate(
    *, action: np.ndarray, previous_action: np.ndarray, params: Any, limit_rad: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    """Bound actual float32 target step using only the previous executed action.

    This changes the whole policy command where necessary and does not assert
    joint limits, torque limits, balance, or safe hardware behavior. The caller
    must feed the returned action into the policy history and physics.
    """
    if type(limit_rad) is not float or not 0 < limit_rad <= 0.2:
        raise ValueError("finite bounded explicit target step required")
    proposed, previous = target29(action, params), target29(previous_action, params)
    desired = np.clip(proposed, previous - limit_rad, previous + limit_rad)
    affected = desired != proposed
    output = action.copy()
    scale = np.asarray(params.action_scale, dtype=np.float32)
    offset = np.asarray(params.action_offset, dtype=np.float32)
    output[affected] = (
        (desired[affected].astype(np.float32) - offset[affected]) / scale[affected]
    ).astype(np.float32)
    actual = target29(output, params)
    for index in np.flatnonzero(np.abs(actual - previous) > limit_rad + 1e-6):
        safe = np.float32(previous_action[index])
        unsafe = np.float32(output[index])
        for _ in range(40):
            middle = np.float32((float(safe) + float(unsafe)) / 2)
            if middle in (safe, unsafe):
                break
            trial = output.copy()
            trial[index] = middle
            if abs(target29(trial, params)[index] - previous[index]) <= limit_rad + 1e-6:
                safe = middle
            else:
                unsafe = middle
        output[index] = safe
    actual = target29(output, params)
    if (
        np.any(output[~affected] != action[~affected])
        or np.max(np.abs(actual - previous)) > limit_rad + 1e-6
    ):
        raise ValueError("actual target rate filter violated its explicit bound")
    return output, actual - proposed


def limit_ankle_target_rate(
    *, action: np.ndarray, previous_action: np.ndarray, params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Limit only ankle target slew; other joints retain the unmodified policy.

    This experimental intervention is not a full-joint slew or safety bound.
    """
    filtered, _ = limit_target_rate(action=action, previous_action=previous_action, params=params)
    output = action.copy()
    output[list(ANKLE_JOINT_INDICES)] = filtered[list(ANKLE_JOINT_INDICES)]
    return output, target29(output, params) - target29(action, params)
