"""Bounded measured-joint planner-input experiment, not a motion controller."""

import numpy as np


def measured_joint_context(
    context: np.ndarray,
    measured_qpos: np.ndarray,
    current_reference: np.ndarray,
    joint_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Only29 reference joints change, uniformly scaled to at most0.12rad.

    No clipping of illegal original or proposed references. Root/quaternions,
    relative joint velocities between context rows, and caller arrays are
    unchanged. Read-only independent arrays do not grant execution authority.
    """
    for a, shape in (
        (context, (4, 36)),
        (measured_qpos, (43,)),
        (current_reference, (36,)),
        (joint_limits, (29, 2)),
    ):
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
        ):
            raise ValueError("finite owned-shape float64 measured/context/limits required")
    if (
        np.any(joint_limits[:, 0] >= joint_limits[:, 1])
        or np.max(abs(np.linalg.norm(context[:, 3:7], axis=1) - 1)) > 1e-4
        or abs(np.linalg.norm(measured_qpos[3:7]) - 1) > 1e-4
        or abs(np.linalg.norm(current_reference[3:7]) - 1) > 1e-4
    ):
        raise ValueError("ordered original joint limits and unit pose quaternions required")
    if np.any(context[:, 7:36] < joint_limits[:, 0]) or np.any(
        context[:, 7:36] > joint_limits[:, 1]
    ):
        raise ValueError("original planner context outside native joint limits; no repair")
    delta = measured_qpos[7:36] - current_reference[7:36]
    delta = delta * min(1.0, 0.12 / max(float(np.max(abs(delta))), 1e-12))
    result = context.copy()
    result[:, 7:36] += delta
    if np.any(result[:, 7:36] < joint_limits[:, 0]) or np.any(result[:, 7:36] > joint_limits[:, 1]):
        raise ValueError("proposed measured-joint context outside native joint limits; no clipping")
    result.setflags(write=False)
    delta.setflags(write=False)
    return result, delta
