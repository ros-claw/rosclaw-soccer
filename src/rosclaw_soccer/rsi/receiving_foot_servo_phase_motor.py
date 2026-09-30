"""SIM_ONLY measured foot-ball feedback after a genuine own-foot contact."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_adaptive_phase_motor import ReceivingAdaptivePhaseMotor
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingFootServoPhaseMotor(ReceivingAdaptivePhaseMotor):
    """Bounded task-space correction through measured foot Jacobian, not raw torque."""

    servo_position_gain: float = 0.0
    servo_velocity_horizon_sec: float = 0.0
    servo_contact_radius_m: float = 0.11
    servo_foot_index: int | None = field(init=False, default=None)
    servo_active_frames: int = field(init=False, default=0)
    servo_peak_residual_rad: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    self.servo_position_gain,
                    self.servo_velocity_horizon_sec,
                    self.servo_contact_radius_m,
                )
            )
            or not 0.0 <= self.servo_position_gain <= 0.5
            or not 0.0 <= self.servo_velocity_horizon_sec <= 0.08
            or not 0.08 <= self.servo_contact_radius_m <= 0.16
        ):
            raise ValueError("bounded finite post-contact foot servo required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_foot_servo_phase_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "position_gain": self.servo_position_gain,
                "velocity_horizon_sec": self.servo_velocity_horizon_sec,
                "contact_radius_m": self.servo_contact_radius_m,
                "trigger": "first_measured_own_foot_contact",
                "input": "same_frame_ball_foot_position_velocity_and_jacobian",
                "peak_joint_correction_rad": 0.10,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = super().propose(observation)
        if (
            self.protected_episode
            or self.selected_expert != "high"
            or self.servo_position_gain == 0.0
            and self.servo_velocity_horizon_sec == 0.0
            or self.first_contact_time_sec is None
        ):
            return base
        age = observation.time_sec - self.first_contact_time_sec
        if not 0 <= age < 0.32:
            return base
        if self.servo_foot_index is None:
            foot = observation.last_own_contact_foot
            if foot not in (1, 2):
                return base
            self.servo_foot_index = foot - 1
        feet = observation.foot_kinematics
        if feet is None or feet.foot_linear_velocity_world_mps is None:
            raise ValueError("same-frame measured foot position, speed and Jacobian required")
        index = self.servo_foot_index
        foot_position = np.asarray(feet.foot_position_world_m[index][:2], dtype=np.float64)
        foot_velocity = np.asarray(feet.foot_linear_velocity_world_mps[index][:2], dtype=np.float64)
        ball_position = np.asarray(observation.qpos[36:38], dtype=np.float64)
        ball_velocity = np.asarray(observation.qvel[35:37], dtype=np.float64)
        separation = ball_position - foot_position
        distance = float(np.linalg.norm(separation))
        position_error = (
            max(distance - self.servo_contact_radius_m, 0.0) * separation / distance
            if distance > 1e-9
            else np.zeros(2, dtype=np.float64)
        )
        desired = self.servo_position_gain * position_error + self.servo_velocity_horizon_sec * (
            ball_velocity - foot_velocity
        )
        desired = np.clip(desired, -0.08, 0.08)
        jacobian = np.asarray(feet.foot_linear_jacobian_world[index][:2], dtype=np.float64)
        correction = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.01 * np.eye(2), desired)
        correction = np.clip(correction, -0.10, 0.10)
        envelope = age / 0.04 if age < 0.04 else min(1.0, (0.32 - age) / 0.16)
        target = np.asarray(base, dtype=np.float64)
        joint_slice = slice(index * 6, index * 6 + 6)
        target[joint_slice] = np.clip(target[joint_slice] + envelope * correction, -0.35, 0.35)
        delta = float(np.max(np.abs(target[joint_slice] - np.asarray(base)[joint_slice])))
        if delta:
            self.servo_active_frames += 1
            self.servo_peak_residual_rad = max(self.servo_peak_residual_rad, delta)
        return tuple(float(value) for value in target)
