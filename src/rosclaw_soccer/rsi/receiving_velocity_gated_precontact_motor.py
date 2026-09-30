"""SIM_ONLY speed-limited measured-state gate for trained receiving contact."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_gated_precontact_motor import ReceivingGatedPrecontactMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingVelocityGatedPrecontactMotor(ReceivingGatedPrecontactMotor):
    """Only enter the near-side skill for measured, bounded incoming speed."""

    minimum_relative_vx_feature: float = -0.48
    activation_features: tuple[float, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.minimum_relative_vx_feature) is not float
            or not math.isfinite(self.minimum_relative_vx_feature)
            or not -0.55 <= self.minimum_relative_vx_feature <= -0.40
        ):
            raise ValueError("bounded measured incoming-speed activation required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_velocity_gated_precontact_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "minimum_relative_vx_feature": self.minimum_relative_vx_feature,
                "velocity_source": "same_frame_ball_minus_body_vx_divided_by_two",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not self.activation_decided and observation.frame >= 19:
            measured = ReceivingTemporalMotorExpert.features(observation)
            self.activation_features = measured
            self.activation_enabled = measured[3] >= self.minimum_relative_vx_feature and any(
                np.linalg.norm(np.asarray(measured) - np.asarray(row)) <= self.activation_radius
                for row in self.activation_states
            )
            self.activation_decided = True
            if self.activation_enabled:
                self.neural_weights = self.candidate_neural_weights
        return super().propose(observation)
