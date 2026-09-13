"""Re-express previous joint-target actions when a locomotion input is reflected.

Only private inference coordinates change. This neither executes a target nor
transfers policy history across different models or different physical players.
"""

from __future__ import annotations

import math

import numpy as np

from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions


def encode_applied_locomotion_target(
    target: np.ndarray,
    *,
    default_angles: np.ndarray,
    action_scale: float,
    joint_to_motor: np.ndarray,
    reflected: bool,
) -> np.ndarray:
    """Express an externally applied motor target in a shadow policy's frame.

    Float32 policy defaults and float32/64 applied targets preserve inference
    arithmetic before producing a float32 action. Defaults are in
    policy order, while target is in physical motor order. A caller must own
    evidence that this is the previous applied target for this player/tick;
    this pure conversion neither establishes that provenance nor transfers or
    initializes recurrent state. Using it changes the previous-action convention
    and requires separate physical qualification of the consuming controller.
    """
    if (
        type(reflected) is not bool
        or type(action_scale) not in (int, float)
        or not math.isfinite(action_scale)
        or not 0.001 <= action_scale <= 5.0
        or not isinstance(joint_to_motor, np.ndarray)
        or joint_to_motor.shape != (29,)
        or joint_to_motor.dtype.kind not in "iu"
        or not np.array_equal(np.sort(joint_to_motor), np.arange(29))
    ):
        raise ValueError("explicit frame, bounded scale and bijective joint mapping required")
    for value, dtypes in (
        (target, (np.dtype("float32"), np.dtype("float64"))),
        (default_angles, (np.dtype("float32"),)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (29,)
            or value.dtype not in dtypes
            or not np.isfinite(value).all()
            or np.max(np.abs(value)) > 10
        ):
            raise ValueError("bounded float32/64 target and float32 policy defaults required")
    physical = mirror_g1_joint_positions(target).astype(target.dtype) if reflected else target
    result: np.ndarray = (physical[joint_to_motor] - default_angles) / np.asarray(
        action_scale, dtype=target.dtype
    )
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 100:
        raise ValueError("applied target cannot be represented within policy action envelope")
    return result.astype(np.float32, copy=True)


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
