# SPDX-License-Identifier: Apache-2.0
# Formulas adapted from G1-mjlab-soccer 45e3d1e and MJLab tracking rewards.
# See third_party/NOTICE.md. Experimental SIM-only course, not deployment.
"""Native anatomical motion tracking and unchanged source termination thresholds."""

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

TRACKING_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
TERMINATION_NAMES = ("fell_over", "anchor_pos_z", "anchor_ori", "ee_body_pos")
TRACKING_NAMES = (
    "anchor_orientation",
    "body_position",
    "body_orientation",
    "body_linear",
    "body_angular",
)


def rotation(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=np.float64))
    return matrix.reshape(3, 3)


def quaternion_error(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.zeros(3)
    mujoco.mju_subQuat(delta, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return float(np.linalg.norm(delta))


class NativeCourseTracking:
    def __init__(self, model: Any, layout: Any, references: list[dict[str, np.ndarray]]):
        self.model, self.layout, self.references = model, layout, references
        self.ids = [model.body("robot/" + name).id for name in TRACKING_BODIES]
        self.anchor = TRACKING_BODIES.index("torso_link")
        self.ee = [
            TRACKING_BODIES.index(n)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            )
        ]
        if len(references) != 8:
            raise ValueError("complete TRAIN8 body references required")
        for ref in references:
            length = len(ref["position"])
            for field, dim in (("position", 3), ("quaternion", 4), ("linear", 3), ("angular", 3)):
                if ref[field].shape != (length, 14, dim) or not np.isfinite(ref[field]).all():
                    raise ValueError("complete finite14 anatomical body references required")
            if np.max(abs(np.linalg.norm(ref["quaternion"], axis=-1) - 1)) > 1e-5:
                raise ValueError("unit source orientations required, no repair")

    def measure(
        self, data: Any, *, source: int, frame: int, origin: np.ndarray
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        ref = self.references[source]
        rp = ref["position"][frame].astype(np.float64) + origin
        rq = ref["quaternion"][frame].astype(np.float64)
        actual_p = data.xpos[self.ids]
        actual_q = data.xquat[self.ids]
        anchor_r = rotation(rq[self.anchor])
        current_r = data.xmat[self.ids[self.anchor]].reshape(3, 3)
        inverse = rq[self.anchor].copy()
        inverse[1:] *= -1
        relative = np.zeros(4)
        mujoco.mju_mulQuat(relative, actual_q[self.anchor], inverse)
        w, x, y, z = relative
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw_q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        yaw_r = rotation(yaw_q)
        delta = actual_p[self.anchor].copy()
        delta[2] = rp[self.anchor, 2]
        transformed_p = delta + (rp - rp[self.anchor]) @ yaw_r.T
        transformed_q = []
        for quat in rq:
            rotated = np.zeros(4)
            mujoco.mju_mulQuat(rotated, yaw_q, quat)
            transformed_q.append(rotated)
        linear, angular = [], []
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for body in self.ids:
            mujoco.mj_jacBody(self.model, data, jp, jr, body)
            linear.append(jp @ data.qvel)
            angular.append(jr @ data.qvel)
        errors = np.array(
            [
                quaternion_error(rq[self.anchor], actual_q[self.anchor]) ** 2,
                np.square(transformed_p - actual_p).sum(axis=-1).mean(),
                np.mean(
                    [
                        quaternion_error(a, b) ** 2
                        for a, b in zip(transformed_q, actual_q, strict=True)
                    ]
                ),
                np.square(ref["linear"][frame] - linear).sum(axis=-1).mean(),
                np.square(ref["angular"][frame] - angular).sum(axis=-1).mean(),
            ]
        )
        tracking = np.exp(-errors / np.square([0.4, 0.3, 0.4, 1.0, 3.14]))
        root_r = data.xmat[self.layout.pelvis].reshape(3, 3)
        guards = np.array(
            [
                np.arccos(np.clip(root_r[2, 2], -1, 1)) > np.deg2rad(70),
                abs(rp[self.anchor, 2] - actual_p[self.anchor, 2]) > 0.25,
                abs(anchor_r[2, 2] - current_r[2, 2]) > 0.8,
                np.any(abs(transformed_p[self.ee, 2] - actual_p[self.ee, 2]) > 0.25),
            ],
            dtype=bool,
        )
        if not np.isfinite(tracking).all():
            raise ValueError("nonfinite native motion tracking")
        return tracking, guards
