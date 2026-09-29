"""SIM_ONLY append-only measured neural skill recall behind frozen parent guards."""

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


@dataclass(frozen=True)
class ReceivingMeasuredSkill:
    initial_features: tuple[float, ...]
    weights: KinematicMotorWeights
    radius: float = 0.006

    def __post_init__(self) -> None:
        if (
            type(self.initial_features) is not tuple
            or len(self.initial_features) != 10
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 3
                for value in self.initial_features
            )
            or not isinstance(self.weights, KinematicMotorWeights)
            or type(self.radius) is not float
            or not 0.003 <= self.radius <= 0.012
        ):
            raise ValueError("finite measured bounded neural skill required")


@dataclass
class ReceivingMultiSkillBank(ReceivingProtectedComposedNeural):
    """Recall local learned skills by measured state, never course ID or outcome."""

    learned_skills: tuple[ReceivingMeasuredSkill, ...] = ()
    selected_skill: int | None = field(init=False, default=None)
    selection_decided: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.learned_skills) is not tuple
            or not 2 <= len(self.learned_skills) <= 8
            or any(not isinstance(skill, ReceivingMeasuredSkill) for skill in self.learned_skills)
        ):
            raise ValueError("two to eight typed measured skills required")
        for index, skill in enumerate(self.learned_skills):
            feature = np.asarray(skill.initial_features)
            if any(
                np.linalg.norm(feature - np.asarray(row)) <= skill.radius + self.protection_radius
                for row in self.protected_initial_features
            ) or any(
                np.linalg.norm(feature - np.asarray(other.initial_features))
                <= skill.radius + other.radius
                for other in self.learned_skills[:index]
            ):
                raise ValueError("learned skills must be disjoint from old and each other")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_multi_skill_bank.v1",
                "parent_contract_hash": self.contract_hash,
                "learned_skills": [
                    {
                        "initial_features": skill.initial_features,
                        "weights_hash": skill.weights.contract_hash,
                        "radius": skill.radius,
                    }
                    for skill in self.learned_skills
                ],
                "selection": "first_measured_state_nearest_in_disjoint_radius",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not self.selection_decided and observation.frame >= 19:
            measured = np.asarray(ReceivingKinematicTemporalExpert.features(observation)[:10])
            distances = [
                float(np.linalg.norm(measured - np.asarray(skill.initial_features)))
                for skill in self.learned_skills
            ]
            matched = [
                index
                for index, (distance, skill) in enumerate(
                    zip(distances, self.learned_skills, strict=True)
                )
                if distance <= skill.radius
            ]
            if matched:
                self.selected_skill = min(matched, key=lambda index: distances[index])
                self.neural_weights = self.learned_skills[self.selected_skill].weights
            self.selection_decided = True
        return super().propose(observation)
