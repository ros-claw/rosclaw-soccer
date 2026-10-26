"""Read-only reconstruction of the existing direct-approach geometry guard."""

import numpy as np


def direct_approach_geometry(
    base_position: np.ndarray,
    base_quaternion: np.ndarray,
    reference_ball_position: np.ndarray,
    reference_goal_position: np.ndarray,
) -> dict[str, float | bool]:
    for arr, shape in (
        (base_position, (3,)),
        (base_quaternion, (4,)),
        (reference_ball_position, (3,)),
        (reference_goal_position, (3,)),
    ):
        if type(arr) is not np.ndarray or arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError("finite explicit measured/reference geometry required")
    if abs(float(np.linalg.norm(base_quaternion)) - 1.0) > 1e-6:
        raise ValueError("normalized measured base quaternion required")
    direction = (reference_goal_position[:2] - reference_ball_position[:2]).astype(np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("nondegenerate reference direction required")
    direction /= length
    left = np.array([-direction[1], direction[0]])
    relative = reference_ball_position[:2] - base_position[:2]
    qw, qx, qy, qz = base_quaternion
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    desired = np.arctan2(direction[1], direction[0])
    heading = float(abs(np.arctan2(np.sin(yaw - desired), np.cos(yaw - desired))))
    forward = float(relative @ direction)
    lateral = float(relative @ left)
    return dict(
        heading_error_rad=heading,
        forward_offset_m=forward,
        lateral_offset_m=lateral,
        heading_within_guard=heading <= 0.2,
        forward_within_guard=0.3 <= forward <= 1.5,
        lateral_within_guard=abs(lateral) <= 0.2,
        measured_joint_pose_still_requires_separate_validation=True,
        physical_motion_evaluated=False,
    )
