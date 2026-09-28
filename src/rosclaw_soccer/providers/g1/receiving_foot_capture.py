"""Bounded simulation-only task-space receiving teacher for local training.

This controller is a hypothesis generator, not a qualified runtime policy.
It never edits a ball state or bypasses the world's joint/torque guards.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class ReceivingFootCaptureTeacher:
    pre_gain: float
    post_gain: float
    offset_x_m: float = -0.18
    offset_y_m: float = 0.03
    offset_z_m: float = -0.08
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        values = (
            self.pre_gain,
            self.post_gain,
            self.offset_x_m,
            self.offset_y_m,
            self.offset_z_m,
        )
        if (
            any(type(value) is not float or not np.isfinite(value) for value in values)
            or not 0.0 <= self.pre_gain <= 1.0
            or not 0.0 <= self.post_gain <= 1.0
            or self.pre_gain == self.post_gain == 0.0
            or not -0.30 <= self.offset_x_m <= -0.08
            or not -0.10 <= self.offset_y_m <= 0.10
            or not -0.10 <= self.offset_z_m <= 0.0
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("bounded SIM_ONLY foot-capture teacher parameters required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "soccer.receiving_foot_capture_teacher.v1",
                    "pre_gain": self.pre_gain,
                    "post_gain": self.post_gain,
                    "offset_x_m": self.offset_x_m,
                    "offset_y_m": self.offset_y_m,
                    "offset_z_m": self.offset_z_m,
                    "activation_ceiling": self.activation_ceiling,
                }
            )
        )

    def motor_target(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        left_ankle_body: int,
        left_joint_dofs: NDArray[np.int64],
        left_q: NDArray[np.float64],
        foundation_target: NDArray[np.float64],
        current_target: NDArray[np.float64],
        left_joint_ranges: NDArray[np.float64],
        ball_position: NDArray[np.float64],
        has_foot_contact: bool,
        elapsed_sec: float,
    ) -> NDArray[np.float64]:
        if (
            type(left_ankle_body) is not int
            or not 0 <= left_ankle_body < model.nbody
            or left_joint_dofs.shape != (6,)
            or left_q.shape != (6,)
            or foundation_target.shape != (29,)
            or current_target.shape != (29,)
            or left_joint_ranges.shape != (6, 2)
            or ball_position.shape != (3,)
            or type(has_foot_contact) is not bool
            or not np.isfinite(elapsed_sec)
            or not all(
                np.isfinite(value).all()
                for value in (
                    left_q,
                    foundation_target,
                    current_target,
                    left_joint_ranges,
                    ball_position,
                )
            )
        ):
            raise ValueError("finite measured SIM_ONLY foot-capture state required")
        gain = self.post_gain if has_foot_contact else self.pre_gain
        if gain == 0.0 or has_foot_contact and not 0.0 <= elapsed_sec <= 0.50:
            return current_target
        desired_ankle = np.asarray(
            (
                ball_position[0] + self.offset_x_m,
                ball_position[1] + self.offset_y_m,
                np.clip(ball_position[2] + self.offset_z_m, 0.04, 0.08),
            ),
            dtype=np.float64,
        )
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacBody(model, data, jacp, jacr, left_ankle_body)
        local_jacobian = jacp[:, left_joint_dofs]
        error = np.clip(desired_ankle - data.xpos[left_ankle_body], -0.20, 0.20)
        step = local_jacobian.T @ np.linalg.solve(
            local_jacobian @ local_jacobian.T + 0.02 * np.eye(3), error
        )
        result = current_target.copy()
        result[:6] = np.clip(
            left_q + gain * step,
            np.maximum(foundation_target[:6] - 0.35, left_joint_ranges[:, 0]),
            np.minimum(foundation_target[:6] + 0.35, left_joint_ranges[:, 1]),
        )
        return result
