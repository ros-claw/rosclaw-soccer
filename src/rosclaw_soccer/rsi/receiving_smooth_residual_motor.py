"""SIM_ONLY continuous measured-state gain for a bounded receiving residual."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

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
class ReceivingSmoothResidualMotor(ReceivingLateralVelocityGatedPrecontactMotor):
    """Interpolate a learned contact residual by first measured lateral state."""

    refined_neural_weights: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    gain_lateral_start: float = 0.765
    gain_lateral_full: float = 0.767
    residual_gain: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.refined_neural_weights, KinematicMotorWeights)
            or type(self.gain_lateral_start) is not float
            or type(self.gain_lateral_full) is not float
            or not math.isfinite(self.gain_lateral_start)
            or not math.isfinite(self.gain_lateral_full)
            or not 0.750
            <= self.gain_lateral_start
            < self.gain_lateral_full
            <= self.maximum_relative_lateral_feature
            or any(
                left != right
                for left, right in (
                    (
                        self.candidate_neural_weights.input_matrix,
                        self.refined_neural_weights.input_matrix,
                    ),
                    (
                        self.candidate_neural_weights.input_bias,
                        self.refined_neural_weights.input_bias,
                    ),
                    (
                        self.candidate_neural_weights.output_matrix,
                        self.refined_neural_weights.output_matrix,
                    ),
                )
            )
        ):
            raise ValueError("bounded compatible measured-state residual ramp required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_smooth_residual_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "refined_neural_weights_hash": self.refined_neural_weights.contract_hash,
                "gain_lateral_start": self.gain_lateral_start,
                "gain_lateral_full": self.gain_lateral_full,
                "gain_source": "same_frame_ball_minus_body_y_divided_by_point_two",
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
            self.activation_decided = True
            if self.activation_enabled:
                self.residual_gain = float(
                    np.clip(
                        (measured[1] - self.gain_lateral_start)
                        / (self.gain_lateral_full - self.gain_lateral_start),
                        0.0,
                        1.0,
                    )
                )
                if self.residual_gain == 0.0:
                    self.neural_weights = self.candidate_neural_weights
                elif self.residual_gain == 1.0:
                    self.neural_weights = self.refined_neural_weights
                else:
                    base = np.asarray(self.candidate_neural_weights.output_bias)
                    refined = np.asarray(self.refined_neural_weights.output_bias)
                    self.neural_weights = replace(
                        self.candidate_neural_weights,
                        output_bias=tuple(
                            float(value) for value in base + self.residual_gain * (refined - base)
                        ),
                    )
        return ReceivingFootServoPhaseMotor.propose(self, observation)
