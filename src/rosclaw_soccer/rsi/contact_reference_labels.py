"""Read-only labels for a future pre-parent reference actor, not dataset admission.

Recover the *actual* float64 navigation target, not the similar float32 target
captured inside the Foundation. Every label is from the same executed state and
tick, never subtraction of different policy trajectories. No clipping or fitting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    G1SonicRunupController,
    _sonic_control_parameters,
)


def reference_labels(trace: dict[str, NDArray[np.generic]]) -> dict[str, NDArray[np.float64]]:
    """Reconstruct current pre-motor references without repeating neural inference.

    Source authentication and task/contact qualification belong to the caller.
    This does not export training data or authorize using a rejected teacher.
    """
    raw = trace.get("foundation_neural_raw_action_isaac")
    pre = trace.get("pre_motor_joint_target_rad")
    command = trace.get("navigation_command")
    for array, shape, dtype in (
        (raw, (300, 1, 29), np.float32),
        (pre, (300, 1, 29), np.float64),
        (command, (300, 1, 3), np.float64),
    ):
        if (
            type(array) is not np.ndarray
            or array.shape != shape
            or array.dtype != dtype
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "whole finite original raw-action, pre-motor and command clocks required"
            )
    assert raw is not None and pre is not None and command is not None
    # Exactly sonic_navigation.commit_batched_action arithmetic, not the torch
    # float32 default/action-scale arithmetic captured by BatchedSonicTracker.
    scale = _sonic_control_parameters(1.0, (1.0,) * 29)[2]
    navigation = G1SonicRunupController.default_angles + raw[:, :, ISAACLAB_TO_MUJOCO] * scale
    delta = pre - navigation
    if "foundation_navigation_joint_target_rad" in trace:
        captured = trace["foundation_navigation_joint_target_rad"]
        if (
            captured.shape != navigation.shape
            or captured.dtype != navigation.dtype
            or captured.tobytes() != navigation.tobytes()
        ):
            raise ValueError("actual float64 navigation target must reconstruct byte-exactly")
    if "taskspace_joint_increment_rad" in trace:
        captured_delta = trace["taskspace_joint_increment_rad"]
        if (
            captured_delta.shape != delta.shape
            or captured_delta.dtype != delta.dtype
            or captured_delta.tobytes() != delta.tobytes()
        ):
            raise ValueError("actual reference increment must reconstruct byte-exactly")
    if np.any(delta[:, :, 12:] != 0) or np.max(np.abs(delta)) > 0.35 + 1e-12:
        raise ValueError("observed reference correction must stay in original bounded twelve legs")
    return dict(
        navigation_command=command.copy(),
        navigation_joint_target_rad=navigation.copy(),
        pre_parent_reference_increment_rad=delta[:, :, :12].copy(),
    )
