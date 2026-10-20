"""Read-only reference70 reconstruction from actual native body geometry."""

import numpy as np


def point_velocity_command(
    *,
    root_rotation: np.ndarray,
    current_position: np.ndarray,
    reference_position: np.ndarray,
    reference_linear_velocity: np.ndarray,
    reference_angular_velocity: np.ndarray,
    position_gain: float = 0.0,
) -> np.ndarray:
    """The position and twist must refer to the SAME anatomical point.

    Rotation into a coordinate basis does not transport velocity between an
    articulated torso and pelvis. Caller explicitly binds the reference point.
    """
    for value, shape in (
        (root_rotation, (3, 3)),
        (current_position, (3,)),
        (reference_position, (3,)),
        (reference_linear_velocity, (3,)),
        (reference_angular_velocity, (3,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype not in (np.float32, np.float64)
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite point-matched position and twist arrays required")
    if (
        not np.allclose(root_rotation.T @ root_rotation, np.eye(3), rtol=0, atol=1e-5)
        or abs(np.linalg.det(root_rotation) - 1) > 1e-5
    ):
        raise ValueError("proper root rotation required")
    if (
        type(position_gain) is not float
        or not np.isfinite(position_gain)
        or not 0 <= position_gain <= 2
    ):
        raise ValueError("bounded explicit reference position-feedback gain required")
    linear_body = root_rotation.T @ (
        reference_linear_velocity + position_gain * (reference_position - current_position)
    )
    angular_body = root_rotation.T @ reference_angular_velocity
    command = np.asarray([linear_body[0], linear_body[1], angular_body[2]], dtype=np.float32)
    return np.clip(
        command,
        np.array([-0.5, -0.5, -1], dtype=np.float32),
        np.array([1, 0.5, 1], dtype=np.float32),
    )


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
    position_gain: float = 0.0,
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
    command = point_velocity_command(
        root_rotation=root_rotation,
        current_position=anchor_position,
        reference_position=reference_position,
        reference_linear_velocity=reference_linear_velocity,
        reference_angular_velocity=reference_angular_velocity,
        position_gain=position_gain,
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
