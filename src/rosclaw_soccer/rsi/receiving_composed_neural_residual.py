"""SIM_ONLY 48D online neural residual on a frozen composed receiving parent."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    FEATURE_COUNT,
    HIDDEN_COUNT,
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingComposedNeuralResidual(ReceivingComposedContactRouter):
    """Only a 12-joint bounded proposal; the ordinary A2 cursor remains owner."""

    neural_weights: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    exploration_std: float = 0.0
    exploration_seed: int = 0
    observed_frames: list[int] = field(init=False, default_factory=list)
    observed_features: list[tuple[float, ...]] = field(init=False, default_factory=list)
    sampled_logits: list[tuple[float, ...]] = field(init=False, default_factory=list)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.neural_weights, KinematicMotorWeights)
            or type(self.exploration_std) is not float
            or not math.isfinite(self.exploration_std)
            or not 0 <= self.exploration_std <= 0.2
            or type(self.exploration_seed) is not int
            or not 0 <= self.exploration_seed < 2**31
        ):
            raise ValueError("bounded SIM_ONLY neural residual exploration required")
        self._rng = np.random.default_rng(self.exploration_seed)
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_composed_neural_residual.v1",
                "parent_contract_hash": self.contract_hash,
                "neural_weights_hash": self.neural_weights.contract_hash,
                "exploration_std": self.exploration_std,
                "exploration_seed": self.exploration_seed,
                "residual_amplitude_rad": 0.15,
                "parent_fallback": "exact",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def _mean_logits(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        first = np.asarray(self.neural_weights.input_matrix, dtype=np.float64).reshape(
            HIDDEN_COUNT, FEATURE_COUNT
        )
        hidden = np.tanh(first @ features + np.asarray(self.neural_weights.input_bias))
        output: NDArray[np.float64] = np.asarray(
            self.neural_weights.output_matrix, dtype=np.float64
        ).reshape(12, HIDDEN_COUNT) @ hidden + np.asarray(self.neural_weights.output_bias)
        return output

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = super().propose(observation)
        if self.selected_expert == "parent" or not 19 <= observation.frame < 55:
            return base
        features = ReceivingKinematicTemporalExpert.features(observation)
        mean = self._mean_logits(np.asarray(features, dtype=np.float64))
        sampled = (
            mean + self._rng.normal(0.0, self.exploration_std, 12)
            if self.exploration_std > 0
            else mean
        )
        self.observed_frames.append(observation.frame)
        self.observed_features.append(features)
        self.sampled_logits.append(tuple(float(value) for value in sampled))
        # Zero policy is exactly parent. A2 still enforces its own amplitude,
        # rate, joint and torque limits after receiving this proposal.
        if not np.any(sampled):
            return base
        target = np.asarray(base, dtype=np.float64)
        target[:12] = np.clip(target[:12] + 0.15 * np.tanh(sampled), -0.35, 0.35)
        return tuple(float(value) for value in target)
