"""Explicit SIM-only reference velocity conversion for one initial state."""

import numpy as np
from scipy.spatial.transform import Rotation


def reference_initial_velocities(motion: dict, order: np.ndarray) -> dict:
    if (
        not isinstance(order, np.ndarray)
        or order.shape != (29,)
        or sorted(order.tolist()) != list(range(29))
    ):
        raise ValueError("exact 29-joint order required")
    root_q = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
    ball_q = np.asarray(motion["object_quat_w"][0], dtype=np.float64)
    if any(abs(np.linalg.norm(q) - 1) > 1e-3 for q in (root_q, ball_q)):
        raise ValueError("unit initial rotations required")
    root_r = Rotation.from_quat(root_q[[1, 2, 3, 0]])
    ball_r = Rotation.from_quat(ball_q[[1, 2, 3, 0]])
    values = dict(
        root_linear=np.asarray(motion["body_lin_vel_w"][0, 0], dtype=np.float64),
        root_angular_local=root_r.inv().apply(motion["body_ang_vel_w"][0, 0]),
        joints=np.asarray(motion["joint_vel"][0, order], dtype=np.float64),
        ball_linear=np.asarray(motion["object_lin_vel_w"][0], dtype=np.float64),
        ball_angular_local=ball_r.inv().apply(motion["object_ang_vel_w"][0]),
    )
    if (
        any(not np.isfinite(v).all() for v in values.values())
        or np.linalg.norm(values["root_linear"]) > 5
        or np.linalg.norm(values["root_angular_local"]) > 12
        or np.max(np.abs(values["joints"])) > 30
        or np.linalg.norm(values["ball_linear"]) > 5
        or np.linalg.norm(values["ball_angular_local"]) > 100
    ):
        raise ValueError("finite bounded research initial velocities required")
    return values
