"""Compare measured and reference receiving states; not a physics teacher."""

import numpy as np


def yaw_local(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if (
        q.shape != (4,)
        or vector.shape != (3,)
        or not np.isfinite(q).all()
        or not np.isfinite(vector).all()
        or abs(np.linalg.norm(q) - 1) > 1e-3
    ):
        raise ValueError("unit wxyz rotation and finite three-vector required")
    w, x, y, z = q
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1], vector[2]])


def match_error(actual: dict, reference: dict) -> dict:
    for value in (actual, reference):
        for key, shape in (("ball_relative", (3,)), ("velocity_relative", (3,)), ("joints", (29,))):
            array = value[key]
            if (
                not isinstance(array, np.ndarray)
                or array.shape != shape
                or not np.isfinite(array).all()
            ):
                raise ValueError("finite matched physical state feature required: " + key)
        if not np.isfinite(value["root_height"]):
            raise ValueError("finite body height required")
    position = float(np.linalg.norm(actual["ball_relative"] - reference["ball_relative"]))
    velocity = float(np.linalg.norm(actual["velocity_relative"] - reference["velocity_relative"]))
    joint_rms = float(np.sqrt(np.mean((actual["joints"] - reference["joints"]) ** 2)))
    height = float(abs(actual["root_height"] - reference["root_height"]))
    return dict(
        position_m=position,
        relative_velocity_mps=velocity,
        joint_rms_rad=joint_rms,
        root_height_m=height,
        score=position / 0.25 + velocity / 0.8 + joint_rms / 0.35 + height / 0.1,
        geometry_state_compatible=bool(
            position <= 0.25 and velocity <= 0.8 and joint_rms <= 0.35 and height <= 0.1
        ),
        dynamic_feasibility_verified=False,
        actual_contact_verified=False,
    )
