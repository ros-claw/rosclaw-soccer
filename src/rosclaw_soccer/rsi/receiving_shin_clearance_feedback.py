"""Bounded A1 shin-clearance correction coupled to measured foot control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_taskspace_feedback import ReceivingTaskspaceFeedback
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingShinClearanceFeedback(ReceivingTaskspaceFeedback):
    """Single feedback owner with a local shin-gap gradient, never direct torque."""

    clearance_gain: float = 0.0
    target_clearance_m: float = 0.06
    foot_preservation: float = 1.0
    maximum_foot_shift_m: float = 0.015
    prediction_horizon_sec: float = 0.0
    requires_shin_clearance: bool = field(init=False, default=True)
    _clearance_action_frames: int = field(init=False, default=0)
    _peak_predicted_foot_shift_m: float = field(init=False, default=0.0)
    _previous_clearance_m: tuple[float, float] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        values = (
            self.clearance_gain,
            self.target_clearance_m,
            self.foot_preservation,
            self.maximum_foot_shift_m,
            self.prediction_horizon_sec,
        )
        if (
            any(type(x) is not float or not math.isfinite(x) for x in values)
            or not 0 <= self.clearance_gain <= 1
            or not 0.02 <= self.target_clearance_m <= 0.12
            or not 0 <= self.foot_preservation <= 1
            or not 0.002 <= self.maximum_foot_shift_m <= 0.03
            or not 0 <= self.prediction_horizon_sec <= 0.15
        ):
            raise ValueError("bounded same-player shin-clearance feedback required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_shin_clearance_feedback.v1",
                "parent_contract_hash": self.contract_hash,
                "clearance_gain": self.clearance_gain,
                "target_clearance_m": self.target_clearance_m,
                "foot_preservation": self.foot_preservation,
                "maximum_foot_shift_m": self.maximum_foot_shift_m,
                "prediction_horizon_sec": self.prediction_horizon_sec,
                "requires_shin_clearance": True,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    @property
    def clearance_action_frames(self) -> int:
        return self._clearance_action_frames

    @property
    def peak_predicted_foot_shift_m(self) -> float:
        return self._peak_predicted_foot_shift_m

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        shin = observation.shin_clearance
        if shin is None or observation.foot_kinematics is None:
            raise ValueError("same-frame shin and foot measurements required")
        shin.__post_init__()
        if shin.agent_id != self.agent_id or shin.frame != observation.frame:
            raise ValueError("shin differential belongs to another player or frame")
        action = np.asarray(super().propose(observation), dtype=np.float64)
        previous_clearance = self._previous_clearance_m
        self._previous_clearance_m = shin.clearance_m
        if self.clearance_gain == 0:
            return tuple(float(x) for x in action)
        snapshot = self.mailbox.snapshot
        if snapshot.first_own_foot_time_sec is None:
            right = observation.qpos[37] < observation.qpos[1]
        else:
            right = snapshot.first_own_foot == "right_foot"
        side = 1 if right else 0
        gap = shin.clearance_m[side]
        closing_speed = (
            0.0 if previous_clearance is None else min((gap - previous_clearance[side]) / 0.02, 0.0)
        )
        predicted_gap = gap + self.prediction_horizon_sec * closing_speed
        if predicted_gap >= self.target_clearance_m:
            return tuple(float(x) for x in action)
        feet = observation.foot_kinematics
        ball = np.asarray(observation.qpos[36:39], dtype=np.float64)
        foot = np.asarray(feet.foot_position_world_m[side], dtype=np.float64)
        if np.linalg.norm(ball - foot) > 0.45 or observation.qpos[2] < 0.55:
            return tuple(float(x) for x in action)
        foot_jacobian = np.asarray(feet.foot_linear_jacobian_world[side], dtype=np.float64)
        gradient = np.asarray(shin.gradient_m_per_rad[side], dtype=np.float64)
        projected = gradient - self.foot_preservation * (
            foot_jacobian.T
            @ np.linalg.solve(
                foot_jacobian @ foot_jacobian.T + 0.002 * np.eye(3),
                foot_jacobian @ gradient,
            )
        )
        norm = float(np.linalg.norm(projected))
        if norm <= 1e-9:
            return tuple(float(x) for x in action)
        strength = self.clearance_gain * np.clip(
            (self.target_clearance_m - predicted_gap) / 0.04, 0, 1
        )
        correction = 0.1 * strength * projected / norm
        predicted_shift = float(np.linalg.norm(foot_jacobian @ correction))
        if predicted_shift > self.maximum_foot_shift_m:
            correction *= self.maximum_foot_shift_m / predicted_shift
            predicted_shift = self.maximum_foot_shift_m
        offset = 6 if right else 0
        action[offset : offset + 6] += correction
        np.clip(action, -0.1, 0.1, out=action)
        if np.any(correction):
            self._clearance_action_frames += 1
            self._peak_predicted_foot_shift_m = max(
                self._peak_predicted_foot_shift_m, predicted_shift
            )
        return tuple(float(x) for x in action)
