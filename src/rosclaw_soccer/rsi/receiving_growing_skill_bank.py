"""SIM_ONLY measured skill-bank retention for a newly learned neural contact skill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingGrowingSkillBank(ReceivingProtectedComposedNeural):
    """Keep old successes; select a new actor only in its measured local state.

    This is episodic skill recall, not an unconstrained online actor. The parent
    guard still wins for its original six states, and the single A2 controller
    remains the only source of joint proposals.
    """

    learned_weights: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    learned_initial_features: tuple[float, ...] = ()
    learned_radius: float = 0.012
    learned_episode: bool | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.learned_weights, KinematicMotorWeights)
            or type(self.learned_initial_features) is not tuple
            or len(self.learned_initial_features) != 10
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 3
                for value in self.learned_initial_features
            )
            or type(self.learned_radius) is not float
            or not 0.003 <= self.learned_radius <= 0.02
            or any(
                np.linalg.norm(np.asarray(self.learned_initial_features) - np.asarray(row))
                <= self.learned_radius + self.protection_radius
                for row in self.protected_initial_features
            )
        ):
            raise ValueError("disjoint bounded measured skill-bank states required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_growing_skill_bank.v1",
                "parent_contract_hash": self.contract_hash,
                "learned_weights_hash": self.learned_weights.contract_hash,
                "learned_initial_features": self.learned_initial_features,
                "learned_radius": self.learned_radius,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if self.learned_episode is None and observation.frame >= 19:
            measured = np.asarray(ReceivingKinematicTemporalExpert.features(observation)[:10])
            self.learned_episode = bool(
                np.linalg.norm(measured - np.asarray(self.learned_initial_features))
                <= self.learned_radius
            )
            if self.learned_episode:
                self.neural_weights = self.learned_weights
        return super().propose(observation)
