"""Read-only reference70 reconstruction from actual native body geometry."""

import numpy as np


def reference_inputs70(
    *,
    root_rotation: np.ndarray,
    anchor_position: np.ndarray,
    anchor_rotation: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
    reference_linear_velocity: np.ndarray,
    reference_angular_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    for value, shape in (
        (root_rotation, (3, 3)),
        (anchor_position, (3,)),
        (anchor_rotation, (3, 3)),
        (joint_position, (29,)),
        (joint_velocity, (29,)),
        (reference_position, (3,)),
        (reference_quaternion, (4,)),
        (reference_linear_velocity, (3,)),
        (reference_angular_velocity, (3,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype not in (np.float32, np.float64)
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite typed actual geometry and explicit reference arrays required")
    for rotation in (root_rotation, anchor_rotation):
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or abs(np.linalg.det(rotation) - 1) > 1e-5
        ):
            raise ValueError("actual proper body rotations required")
    if abs(np.linalg.norm(reference_quaternion) - 1) > 1e-5:
        raise ValueError("unit declared reference orientation required; no repair")
    reference_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(reference_rotation, reference_quaternion.astype(np.float64))
    relative_rotation = anchor_rotation.T @ reference_rotation.reshape(3, 3)
    linear_body = root_rotation.T @ reference_linear_velocity
    angular_body = root_rotation.T @ reference_angular_velocity
    command = np.asarray([linear_body[0], linear_body[1], angular_body[2]], dtype=np.float32)
    # This is the declared velocity-command domain, never state/force clipping.
    command = np.clip(
        command,
        np.array([-0.5, -0.5, -1], dtype=np.float32),
        np.array([1, 0.5, 1], dtype=np.float32),
    )
    suffix = np.concatenate(
        (
            joint_position,
            joint_velocity,
            anchor_rotation.T @ (reference_position - anchor_position),
            relative_rotation[:, :2].reshape(6),
            anchor_rotation.T @ reference_linear_velocity,
        )
    ).astype(np.float32)
    if suffix.shape != (70,) or not np.isfinite(suffix).all():
        raise ValueError("finite semantic reference70 required")
    return command, suffix
