"""SIM_ONLY measured-state gate for a trained precontact receiving skill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingGatedPrecontactMotor(ReceivingFootServoPhaseMotor):
    """Select one immutable learned policy from the first real body/ball state."""

    candidate_neural_weights: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    activation_states: tuple[tuple[float, ...], ...] = ()
    activation_radius: float = 0.017
    activation_decided: bool = field(init=False, default=False)
    activation_enabled: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.candidate_neural_weights, KinematicMotorWeights)
            or type(self.activation_states) is not tuple
            or len(self.activation_states) != 3
            or any(
                type(row) is not tuple
                or len(row) != 10
                or any(type(value) is not float or not math.isfinite(value) for value in row)
                for row in self.activation_states
            )
            or type(self.activation_radius) is not float
            or not 0.003 <= self.activation_radius <= 0.017
        ):
            raise ValueError("three finite measured precontact activation states required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_gated_precontact_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "candidate_neural_weights_hash": self.candidate_neural_weights.contract_hash,
                "activation_states": self.activation_states,
                "activation_radius": self.activation_radius,
                "selection": "first_measured_state_whole_episode",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not self.activation_decided and observation.frame >= 19:
            measured = np.asarray(ReceivingTemporalMotorExpert.features(observation))
            self.activation_enabled = any(
                np.linalg.norm(measured - np.asarray(row)) <= self.activation_radius
                for row in self.activation_states
            )
            self.activation_decided = True
            if self.activation_enabled:
                self.neural_weights = self.candidate_neural_weights
        return super().propose(observation)
