"""SIM_ONLY measured-state band routing for a retained receiving sub-skill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_lateral_velocity_gated_precontact_motor import (
    ReceivingLateralVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingBandRefinedPrecontactMotor(ReceivingLateralVelocityGatedPrecontactMotor):
    """Keep the proven actor except inside a bounded measured contact-state band."""

    refined_neural_weights: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    refined_lateral_minimum: float = 0.765
    refined_lateral_maximum: float = 0.769
    refined_relative_vx_minimum: float = -0.480
    refined_relative_vx_maximum: float = -0.465
    refined_episode: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        values = (
            self.refined_lateral_minimum,
            self.refined_lateral_maximum,
            self.refined_relative_vx_minimum,
            self.refined_relative_vx_maximum,
        )
        if (
            not isinstance(self.refined_neural_weights, KinematicMotorWeights)
            or any(type(value) is not float or not math.isfinite(value) for value in values)
            or not 0.750 <= values[0] < values[1] <= 0.775
            or not -0.55 <= values[2] < values[3] <= -0.40
            or values[1] > self.maximum_relative_lateral_feature
            or values[2] < self.minimum_relative_vx_feature
        ):
            raise ValueError("bounded finite measured refinement band required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_band_refined_precontact_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "refined_neural_weights_hash": self.refined_neural_weights.contract_hash,
                "refined_lateral_minimum": self.refined_lateral_minimum,
                "refined_lateral_maximum": self.refined_lateral_maximum,
                "refined_relative_vx_minimum": self.refined_relative_vx_minimum,
                "refined_relative_vx_maximum": self.refined_relative_vx_maximum,
                "source": "same_frame_body_ball_relative_state",
                "selection": "first_measured_state_whole_episode",
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
            self.refined_episode = bool(
                self.activation_enabled
                and self.refined_lateral_minimum <= measured[1] <= self.refined_lateral_maximum
                and self.refined_relative_vx_minimum
                <= measured[3]
                <= self.refined_relative_vx_maximum
            )
            self.activation_decided = True
            if self.activation_enabled:
                self.neural_weights = (
                    self.refined_neural_weights
                    if self.refined_episode
                    else self.candidate_neural_weights
                )
        return ReceivingFootServoPhaseMotor.propose(self, observation)
