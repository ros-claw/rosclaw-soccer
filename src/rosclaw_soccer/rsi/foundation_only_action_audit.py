"""Verify a captured null-residual control chain; not neural or physical replay."""

from typing import Any

import numpy as np


def require_foundation_only_actions(trace: dict[str, Any]) -> None:
    """Require measured Foundation targets to be the actually applied targets.

    Caller must independently authenticate the capture, recompute the network,
    reconstruct its measured history, and replay the actual actuator dynamics.
    This check fills only the link between those independently checked targets.
    """
    arrays = {}
    shapes = {
        "foundation_neural_joint_target_mujoco_rad": (300, 1, 29),
        "foundation_navigation_joint_target_rad": (300, 1, 29),
        "pre_motor_joint_target_rad": (300, 1, 29),
        "joint_target_rad": (300, 1, 29),
        "taskspace_joint_increment_rad": (300, 1, 29),
        "motor_delta_rad": (300, 1, 12),
        "motor_pre_projection_requested_rad": (300, 1, 12),
    }
    for key, shape in shapes.items():
        value = np.asarray(trace.get(key))
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("complete finite captured foundation-only action chain required")
        arrays[key] = value
    for key in (
        "taskspace_joint_increment_rad",
        "motor_delta_rad",
        "motor_pre_projection_requested_rad",
    ):
        if np.any(arrays[key] != 0):
            raise ValueError(
                "foundation-only ablation cannot include applied or requested residuals"
            )
    reference = arrays["foundation_navigation_joint_target_rad"]
    if any(
        not np.array_equal(arrays[key], reference)
        for key in ("pre_motor_joint_target_rad", "joint_target_rad")
    ):
        raise ValueError("foundation-only final targets differ from navigation targets")
    if not np.allclose(
        arrays["foundation_neural_joint_target_mujoco_rad"], reference, atol=1e-6, rtol=0
    ):
        raise ValueError("captured Foundation output differs from applied navigation target")
