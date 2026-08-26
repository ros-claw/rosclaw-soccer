"""Explicit MOSAIC G1 tensor semantics and canonical order conversion.

MOSAIC's conversion script writes ``robot.data.joint_pos`` and
``robot.data.body_pos_w`` directly.  Those tensors are in Isaac Lab's
articulation order, not in the registry order passed to ``find_joints``.
Treating the raw 29 columns as Unitree DDS order silently assigns shoulder
motion to ankles and waist joints.  This module makes that boundary explicit
and reusable by every MOSAIC consumer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json

MOSAIC_G1_ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MOSAIC_G1_ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

MOSAIC_G1_CANONICAL_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF = tuple(
    MOSAIC_G1_ISAACLAB_JOINT_NAMES.index(name) for name in G1_DDS_JOINT_NAMES
)
MOSAIC_G1_ISAACLAB_TO_CANONICAL_BODY = tuple(
    MOSAIC_G1_ISAACLAB_BODY_NAMES.index(name) for name in MOSAIC_G1_CANONICAL_BODY_NAMES
)

MOSAIC_G1_SEMANTIC_CONTRACT_HASH = str(
    hash_json(
        {
            "source_joint_names": MOSAIC_G1_ISAACLAB_JOINT_NAMES,
            "source_body_names": MOSAIC_G1_ISAACLAB_BODY_NAMES,
            "canonical_joint_names": G1_DDS_JOINT_NAMES,
            "canonical_body_names": MOSAIC_G1_CANONICAL_BODY_NAMES,
            "source_to_canonical_dof": MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
            "source_to_canonical_body": MOSAIC_G1_ISAACLAB_TO_CANONICAL_BODY,
            "source_root_quaternion_order": "wxyz",
            "schema_version": "rosclaw.growth.mosaic_g1_semantics.v1",
        }
    )
)


def canonicalize_mosaic_g1_joints(values: NDArray[Any]) -> NDArray[np.float64]:
    """Return finite MOSAIC joint tensors in canonical Unitree DDS order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 29 or not np.isfinite(array).all():
        raise ValueError("MOSAIC G1 joints must have 29 finite columns")
    return np.take(array, MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF, axis=-1)


def canonicalize_mosaic_g1_bodies(values: NDArray[Any]) -> NDArray[np.float64]:
    """Return finite MOSAIC body tensors in canonical MuJoCo body order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2] != 30 or not np.isfinite(array).all():
        raise ValueError("MOSAIC G1 bodies must contain 30 finite bodies")
    return np.take(array, MOSAIC_G1_ISAACLAB_TO_CANONICAL_BODY, axis=-2)


__all__ = [
    "MOSAIC_G1_CANONICAL_BODY_NAMES",
    "MOSAIC_G1_ISAACLAB_BODY_NAMES",
    "MOSAIC_G1_ISAACLAB_JOINT_NAMES",
    "MOSAIC_G1_ISAACLAB_TO_CANONICAL_BODY",
    "MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF",
    "MOSAIC_G1_SEMANTIC_CONTRACT_HASH",
    "canonicalize_mosaic_g1_bodies",
    "canonicalize_mosaic_g1_joints",
]
