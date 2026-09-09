"""Re-express previous joint-target actions when a locomotion input is reflected.

Only private inference coordinates change. This neither executes a target nor
transfers policy history across different models or different physical players.
"""

from __future__ import annotations

import math

import numpy as np

from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions


def remap_previous_locomotion_action(
    action: np.ndarray,
    *,
    default_angles: np.ndarray,
    action_scale: float,
    joint_to_motor: np.ndarray,
    previous_reflected: bool,
    next_reflected: bool,
) -> np.ndarray:
    """Preserve the represented physical joint target across a frame switch.

    Normalization/defaults use policy order, reflection uses physical G1 motor
    order. The two cannot be interchanged or reflected as raw array indices.
    The caller owns player identity, episode boundaries and the previous-frame
    marker. Inputs are never mutated; there is no hardware or promotion surface.
    """
    for value in (action, default_angles):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (29,)
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite floating 29-joint action and normalization required")
    if (
        type(previous_reflected) is not bool
        or type(next_reflected) is not bool
        or type(action_scale) not in (int, float)
        or not math.isfinite(action_scale)
        or not 0.001 <= action_scale <= 5.0
        or np.max(np.abs(action.astype(np.float64))) > 100.0
        or np.max(np.abs(default_angles.astype(np.float64))) > 10.0
        or not isinstance(joint_to_motor, np.ndarray)
        or joint_to_motor.shape != (29,)
        or joint_to_motor.dtype.kind not in "iu"
        or not np.array_equal(np.sort(joint_to_motor), np.arange(29))
    ):
        raise ValueError("bounded action, explicit frames and bijective joint order required")
    if previous_reflected == next_reflected:
        return action.copy()
    policy_target = action.astype(np.float64) * action_scale + default_angles
    physical_order = np.empty(29, dtype=np.float64)
    physical_order[joint_to_motor] = policy_target
    reflected = mirror_g1_joint_positions(physical_order)
    result = (reflected[joint_to_motor] - default_angles) / action_scale
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 100.0:
        raise ValueError("reflected previous action exceeds the original policy envelope")
    return np.asarray(result, dtype=action.dtype)
