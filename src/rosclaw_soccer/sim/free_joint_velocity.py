"""Explicit MuJoCo free-joint velocity frames, independent of robot morphology.

MuJoCo free-joint qvel mixes world-frame translation and body-frame rotation.
This adapter does not rewrite observations of existing trained checkpoints.
Adopting it requires an explicit observation-contract migration and evaluation.
"""

from __future__ import annotations

import numpy as np


def free_joint_body_velocity(
    *, world_from_body_quaternion_wxyz: np.ndarray, free_joint_qvel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return body-frame linear and angular velocity without mutating inputs.

    The quaternion must be unit length. Reject corrupt inputs instead of hiding
    a state-contract error by silently normalizing or clipping them.
    """
    quaternion = np.asarray(world_from_body_quaternion_wxyz)
    velocity = np.asarray(free_joint_qvel)
    for value, shape in ((quaternion, (4,)), (velocity, (6,))):
        if value.shape != shape or value.dtype.kind not in "fiu":
            raise ValueError("free-joint state requires real vectors of length 4 and 6")
        if not np.isfinite(value).all():
            raise ValueError("free-joint state must be finite")
    quaternion = quaternion.astype(np.float64)
    velocity = velocity.astype(np.float64)
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("free-joint quaternion must be unit length")
    w = quaternion[0]
    vector = quaternion[1:]
    # Inverse quaternion rotation: world translation -> instantaneous body.
    linear = velocity[:3]
    cross = np.cross(vector, linear)
    body_linear = linear - 2.0 * w * cross + 2.0 * np.cross(vector, cross)
    # Angular free-joint qvel already uses the body frame: do not rotate twice.
    return body_linear, velocity[3:].copy()
