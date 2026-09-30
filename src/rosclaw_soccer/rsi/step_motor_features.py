"""Causal 50 Hz motor observation for teacher distillation and future learning.

No episode ID, role lookup, future collision, terminal reward or future state is
an inference feature. Current-frame forces are unavailable until after physics;
only the previous completed frame's measured forces may be included.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES

FEATURE_NAMES = (
    tuple(f"joint_position:{n}" for n in G1_DDS_JOINT_NAMES)
    + tuple(f"joint_velocity:{n}" for n in G1_DDS_JOINT_NAMES)
    + tuple(f"body_velocity:{kind}:{axis}" for kind in ("linear", "angular") for axis in "xyz")
    + tuple(f"ball_relative_body:{a}" for a in "xyz")
    + tuple(f"ball_relative_velocity_body:{a}" for a in "xyz")
    + tuple(f"geometry_relative_body:{i}:{a}" for i in range(4) for a in "xyz")
    + tuple(f"nominal_target:{n}" for n in G1_DDS_JOINT_NAMES)
    + tuple(f"previous_motor_delta:{n}" for n in JOINT_NAMES)
    + tuple(f"previous_ball_body_force:{i}" for i in range(6))
    + ("gait_sin", "gait_cos")
    + tuple(f"gravity_body:{a}" for a in "xyz")
)


def feature_vector(
    *,
    joint_position: Any,
    joint_velocity: Any,
    root_pose_xyzw: Any,
    root_velocity_world: Any,
    ball_position_world: Any,
    ball_velocity_world: Any,
    geometry_position_world: Any,
    nominal_target: Any,
    previous_motor_delta: Any,
    previous_contact_forces: Any,
    frame: int,
) -> np.ndarray[Any, Any]:
    arrays = [
        np.asarray(v, dtype=np.float64)
        for v in (
            joint_position,
            joint_velocity,
            root_pose_xyzw,
            root_velocity_world,
            ball_position_world,
            ball_velocity_world,
            geometry_position_world,
            nominal_target,
            previous_motor_delta,
            previous_contact_forces,
        )
    ]
    if (
        tuple(v.shape for v in arrays)
        != ((29,), (29,), (7,), (6,), (3,), (3,), (4, 3), (29,), (12,), (6,))
        or not all(np.isfinite(v).all() for v in arrays)
        or type(frame) is not int
        or not 0 <= frame < 3000
    ):
        raise ValueError("aligned finite measured 50 Hz state required")
    joint, velocity, root, root_v, ball, ball_v, geometry, nominal, previous, forces = arrays
    if abs(float(np.linalg.norm(root[3:])) - 1) > 1e-4 or np.any(forces < 0):
        raise ValueError("unit measured orientation and nonnegative prior force required")
    x, y, z, w = root[3:]
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    phase = 2 * np.pi * frame / 40
    result = np.concatenate(
        (
            joint,
            velocity,
            rotation.T @ root_v[:3],
            rotation.T @ root_v[3:],
            rotation.T @ (ball - root[:3]),
            rotation.T @ (ball_v - root_v[:3]),
            ((geometry - root[:3]) @ rotation).ravel(),
            nominal,
            previous,
            forces,
            (np.sin(phase), np.cos(phase)),
            rotation.T @ np.asarray((0.0, 0.0, -1.0)),
        )
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("invalid causal motor feature vector")
    return cast(np.ndarray[Any, Any], result)
