"""SIM_ONLY measured lateral-and-velocity selection for a receiving motor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.receiving_velocity_gated_precontact_motor import (
    ReceivingVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingLateralVelocityGatedPrecontactMotor(ReceivingVelocityGatedPrecontactMotor):
    """Select only from same-frame body/ball state, then freeze per episode."""

    maximum_relative_lateral_feature: float = 0.767

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.maximum_relative_lateral_feature) is not float
            or not math.isfinite(self.maximum_relative_lateral_feature)
            or not 0.760 <= self.maximum_relative_lateral_feature <= 0.775
        ):
            raise ValueError("bounded measured lateral activation required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_lateral_velocity_gated_precontact_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "maximum_relative_lateral_feature": self.maximum_relative_lateral_feature,
                "lateral_source": "same_frame_ball_minus_body_y_divided_by_point_two",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not self.activation_decided and observation.frame >= 19:
            measured = ReceivingTemporalMotorExpert.features(observation)
            self.activation_features = measured
            self.activation_enabled = (
                measured[1] <= self.maximum_relative_lateral_feature
                and measured[3] >= self.minimum_relative_vx_feature
                and any(
                    np.linalg.norm(np.asarray(measured) - np.asarray(row)) <= self.activation_radius
                    for row in self.activation_states
                )
            )
            self.activation_decided = True
            if self.activation_enabled:
                self.neural_weights = self.candidate_neural_weights
        return super().propose(observation)
