"""SIM_ONLY task-space receiving feedback over immutable measured foot state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_whole_body_foot_tap import ReceivingWholeBodyFootTap
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingTaskspaceFeedback(ReceivingWholeBodyFootTap):
    """Low-dimensional trainable gains; the slot/cursor owns action authority.

    The actor sees current body, ball, immutable foot pose/velocity/Jacobian,
    and completed own-foot contact only. No model, data, actuator, future path,
    or privileged target is passed into it.
    """

    pre_gain: float = 0.0
    post_gain: float = 0.0
    velocity_horizon_sec: float = 0.0
    hip_clearance_rad: float = 0.0
    ankle_compensation_rad: float = 0.0
    target_depth_m: float = 0.18
    target_lateral_m: float = 0.03
    _nonzero_frames: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        super().__post_init__()
        values = (
            self.pre_gain,
            self.post_gain,
            self.velocity_horizon_sec,
            self.hip_clearance_rad,
            self.ankle_compensation_rad,
            self.target_depth_m,
            self.target_lateral_m,
        )
        if (
            any(type(x) is not float or not math.isfinite(x) for x in values)
            or not 0 <= self.pre_gain <= 1
            or not 0 <= self.post_gain <= 1
            or not 0 <= self.velocity_horizon_sec <= 0.15
            or not -0.1 <= self.hip_clearance_rad <= 0.1
            or not -0.1 <= self.ankle_compensation_rad <= 0.1
            or not 0.08 <= self.target_depth_m <= 0.30
            or not -0.08 <= self.target_lateral_m <= 0.16
        ):
            raise ValueError("bounded task-space receiving gains required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_taskspace_feedback.v1",
                "parent_contract_hash": self.contract_hash,
                "pre_gain": self.pre_gain,
                "post_gain": self.post_gain,
                "velocity_horizon_sec": self.velocity_horizon_sec,
                "hip_clearance_rad": self.hip_clearance_rad,
                "ankle_compensation_rad": self.ankle_compensation_rad,
                "target_depth_m": self.target_depth_m,
                "target_lateral_m": self.target_lateral_m,
                "action_substrate": "A1_body29",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    @property
    def nonzero_frames(self) -> int:
        return self._nonzero_frames

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        super().propose(observation)  # complete typed/clocked foot-body-contact tape
        feet = observation.foot_kinematics
        assert feet is not None and feet.foot_linear_velocity_world_mps is not None
        snapshot = self.mailbox.snapshot
        first_time = snapshot.first_own_foot_time_sec
        if first_time is None:
            right = observation.qpos[37] < observation.qpos[1]
            gain = self.pre_gain
            active = True
        else:
            if snapshot.first_own_foot not in ("left_foot", "right_foot"):
                raise ValueError("measured receiving foot required")
            right = snapshot.first_own_foot == "right_foot"
            gain = self.post_gain
            active = observation.time_sec - first_time <= 1.2
        action = np.zeros(29, dtype=np.float64)
        if not active or gain == 0 and self.hip_clearance_rad == self.ankle_compensation_rad == 0:
            return tuple(float(x) for x in action)
        side = 1 if right else 0
        foot = np.asarray(feet.foot_position_world_m[side], dtype=np.float64)
        foot_vel = np.asarray(feet.foot_linear_velocity_world_mps[side], dtype=np.float64)
        ball = np.asarray(observation.qpos[36:39], dtype=np.float64)
        ball_vel = np.asarray(observation.qvel[35:38], dtype=np.float64)
        target = ball + np.asarray(
            (
                -self.target_depth_m,
                -self.target_lateral_m if right else self.target_lateral_m,
                -0.08,
            )
        )
        target[2] = np.clip(target[2], 0.04, 0.08)
        error = np.clip(target - foot, -0.15, 0.15)
        relative_velocity = np.clip(ball_vel - foot_vel, -1.5, 1.5)
        task_delta = error + self.velocity_horizon_sec * relative_velocity
        jacobian = np.asarray(feet.foot_linear_jacobian_world[side], dtype=np.float64)
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.02 * np.eye(3), task_delta)
        offset = 6 if right else 0
        action[offset : offset + 6] = gain * step
        action[offset] += self.hip_clearance_rad
        action[offset + 4] += self.ankle_compensation_rad
        np.clip(action, -0.1, 0.1, out=action)
        if np.any(action):
            self._nonzero_frames += 1
        return tuple(float(x) for x in action)
