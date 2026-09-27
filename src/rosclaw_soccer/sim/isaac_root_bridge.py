"""Fail-closed Isaac Lab XYZW/world root state to MuJoCo WXYZ/free-joint state."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _real_vector(value: NDArray[np.float64], length: int, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value)
    if vector.shape != (length,) or vector.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real vector of length {length}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector.astype(np.float64)


def _rotate_world_to_body(
    vector_w: NDArray[np.float64], quaternion_wxyz: NDArray[np.float64]
) -> NDArray[np.float64]:
    w = quaternion_wxyz[0]
    xyz = quaternion_wxyz[1:]
    cross = np.cross(xyz, vector_w)
    result: NDArray[np.float64] = vector_w - 2.0 * w * cross + 2.0 * np.cross(xyz, cross)
    return result


def _multiply_wxyz(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def isaac_root_to_mujoco(
    *,
    pose_xyzw: NDArray[np.float64],
    velocity_world: NDArray[np.float64],
    asset_quaternion_xyzw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert root-link pose/velocity, preserving MuJoCo's mixed qvel frames.

    Isaac Lab uses XYZW and world-frame linear/angular velocity. MuJoCo free
    joints use WXYZ, world-frame linear velocity, body-frame angular velocity.
    ``asset_quaternion_xyzw`` removes a fixed asset-facing transform from the
    policy observation; it does not change the measured body's angular frame.
    """
    pose = _real_vector(pose_xyzw, 7, "Isaac root pose")
    velocity = _real_vector(velocity_world, 6, "Isaac root velocity")
    asset = _real_vector(asset_quaternion_xyzw, 4, "Isaac asset quaternion")
    observed_wxyz = pose[[6, 3, 4, 5]]
    asset_wxyz = asset[[3, 0, 1, 2]]
    if not np.isclose(np.linalg.norm(observed_wxyz), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("Isaac root quaternion must be unit length")
    if not np.isclose(np.linalg.norm(asset_wxyz), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("Isaac asset quaternion must be unit length")
    inverse_asset = asset_wxyz * np.asarray((1.0, -1.0, -1.0, -1.0))
    policy_quaternion = _multiply_wxyz(observed_wxyz, inverse_asset)
    qpos_root = np.concatenate((pose[:3], policy_quaternion))
    qvel_root = np.concatenate((velocity[:3], _rotate_world_to_body(velocity[3:], observed_wxyz)))
    return qpos_root, qvel_root
