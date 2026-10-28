"""Explicit reference-only resampling for a bounded research tracking pilot."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def retarget_window(source: dict, *, event: int, source_fps: int, ball_lift_m: float) -> dict:
    """Extract +/- 1 s and resample to 50 Hz without changing duration.

    The vertical offset is a reference transformation, not an actual state edit
    or a guarantee of collision-free tracking. Input contact flags remain labels.
    """
    if (
        type(event) is not int
        or type(source_fps) is not int
        or not 30 <= source_fps <= 240
        or type(ball_lift_m) is not float
        or not 0.0 <= ball_lift_m <= 0.1
    ):
        raise ValueError("explicit bounded source window and reference-only ball lift required")
    shapes = {
        "joint_pos": (29,),
        "joint_vel": (29,),
        "body_pos_w": (39, 3),
        "body_quat_w": (39, 4),
        "body_lin_vel_w": (39, 3),
        "body_ang_vel_w": (39, 3),
        "ee_pos_w": (5, 3),
        "ee_quat_w": (5, 4),
        "object_pos_w": (3,),
        "object_quat_w": (4,),
        "object_lin_vel_w": (3,),
        "object_ang_vel_w": (3,),
        "contact_info": (4, 1),
    }
    n = len(source["joint_pos"])
    if event - source_fps < 0 or event + source_fps >= n:
        raise ValueError("full two-second source interval required")
    for key, tail in shapes.items():
        value = source[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (n, *tail)
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite aligned source arrays required: " + key)
    if not np.isin(source["contact_info"], (0, 1)).all():
        raise ValueError("binary source reference contacts required")
    indices = np.arange(event - source_fps, event + source_fps + 1)
    times = (indices - indices[0]) / source_fps
    query = np.arange(101) / 50.0
    result = {}
    for key in shapes:
        values = source[key][indices]
        if "quat" in key:
            if not np.allclose(np.linalg.norm(values, axis=-1), 1.0, atol=1e-3, rtol=0):
                raise ValueError("unit source quaternion required: " + key)
            q = values.reshape(len(indices), -1, 4)
            out = np.empty((len(query), q.shape[1], 4), np.float64)
            for body in range(q.shape[1]):
                xyzw = q[:, body, [1, 2, 3, 0]]
                interp = Slerp(times, Rotation.from_quat(xyzw))(query).as_quat()
                out[:, body] = interp[:, [3, 0, 1, 2]]
            result[key] = out.reshape(len(query), *shapes[key]).astype(np.float32)
        elif key == "contact_info":
            result[key] = values[np.rint(query * source_fps).astype(int)].astype(np.float32)
        else:
            flat = values.reshape(len(indices), -1)
            out = np.stack(
                [np.interp(query, times, flat[:, i]) for i in range(flat.shape[1])], axis=1
            )
            result[key] = out.reshape(len(query), *shapes[key]).astype(np.float32)
    result["object_pos_w"][:, 2] += ball_lift_m
    result["base_pos_w"] = result["body_pos_w"][:, 0].copy()
    result["base_quat_w"] = result["body_quat_w"][:, 0].copy()
    # CFtrack requires table arrays even for a ball task; the author's ball
    # reference uses all-zero placeholders, not observed table poses.
    result["table1_pos_w"] = np.zeros((101, 3), dtype=np.float32)
    result["table2_pos_w"] = np.zeros((101, 3), dtype=np.float32)
    result["fps"] = np.array([50.0], dtype=np.float32)
    return result
