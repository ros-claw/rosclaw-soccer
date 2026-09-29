"""SIM_ONLY measured-state local gate for a contact-phase motor skill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_phase_contact_motor import ReceivingPhaseContactMotor
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingGatedPhaseContactMotor(ReceivingPhaseContactMotor):
    """Enable a learned post-foot-contact skill only near measured safe states."""

    activation_states: tuple[tuple[float, ...], ...] = ()
    activation_radius: float = 0.014
    activation_decided: bool = field(init=False, default=False)
    activation_enabled: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.activation_states) is not tuple
            or len(self.activation_states) != 2
            or any(
                type(row) is not tuple
                or len(row) != 10
                or any(
                    type(value) is not float or not math.isfinite(value) or abs(value) > 3
                    for value in row
                )
                for row in self.activation_states
            )
            or type(self.activation_radius) is not float
            or not 0.003 <= self.activation_radius <= 0.014
        ):
            raise ValueError("two finite measured phase activation states required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_gated_phase_contact_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "activation_states": self.activation_states,
                "activation_radius": self.activation_radius,
                "selection": "first_measured_state_whole_episode",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not self.activation_decided and observation.frame >= 19:
            measured = np.asarray(ReceivingKinematicTemporalExpert.features(observation)[:10])
            self.activation_enabled = any(
                np.linalg.norm(measured - np.asarray(row)) <= self.activation_radius
                for row in self.activation_states
            )
            self.activation_decided = True
            if not self.activation_enabled:
                self.post_contact_logits = (0.0,) * 12
        return super().propose(observation)
