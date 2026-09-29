"""SIM_ONLY proprioceptive post-contact actor behind a frozen A2 precontact skill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    FEATURE_COUNT,
    HIDDEN_COUNT,
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_phase_contact_motor import ReceivingPhaseContactMotor
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingAdaptivePhaseMotor(ReceivingPhaseContactMotor):
    """48D measured feedback to bounded post-contact 12D logits each 20 ms."""

    post_policy: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    post_exploration_std: float = 0.0
    post_exploration_seed: int = 0
    post_observed_frames: list[int] = field(init=False, default_factory=list)
    post_observed_features: list[tuple[float, ...]] = field(init=False, default_factory=list)
    post_sampled_logits: list[tuple[float, ...]] = field(init=False, default_factory=list)
    _post_rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.post_policy, KinematicMotorWeights)
            or type(self.post_exploration_std) is not float
            or not math.isfinite(self.post_exploration_std)
            or not 0 <= self.post_exploration_std <= 0.2
            or type(self.post_exploration_seed) is not int
            or not 0 <= self.post_exploration_seed < 2**31
            or any(self.post_contact_logits)
        ):
            raise ValueError("bounded neural post-contact policy with zero static logits required")
        self._post_rng = np.random.default_rng(self.post_exploration_seed)
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_adaptive_phase_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "post_policy_hash": self.post_policy.contract_hash,
                "post_exploration_std": self.post_exploration_std,
                "post_exploration_seed": self.post_exploration_seed,
                "input": "same_frame_48D_proprioception_foot_velocity_shin_clearance",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def _post_logits(self, features: tuple[float, ...]) -> np.ndarray:
        first = np.asarray(self.post_policy.input_matrix).reshape(HIDDEN_COUNT, FEATURE_COUNT)
        hidden = np.tanh(first @ np.asarray(features) + np.asarray(self.post_policy.input_bias))
        output = np.asarray(self.post_policy.output_matrix).reshape(12, HIDDEN_COUNT) @ hidden
        return np.asarray(output + np.asarray(self.post_policy.output_bias), dtype=np.float64)

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = ReceivingProtectedComposedNeural.propose(self, observation)
        if self.protected_episode or self.selected_expert != "high":
            return base
        if self.first_contact_time_sec is None:
            self.first_contact_time_sec = observation.last_own_foot_contact_time_sec
        if self.first_contact_time_sec is None:
            return base
        age = observation.time_sec - self.first_contact_time_sec
        if not 0 <= age < 0.32:
            return base
        features = ReceivingKinematicTemporalExpert.features(observation)
        mean = self._post_logits(features)
        sampled = (
            mean + self._post_rng.normal(0.0, self.post_exploration_std, 12)
            if self.post_exploration_std > 0
            else mean
        )
        self.post_observed_frames.append(observation.frame)
        self.post_observed_features.append(features)
        self.post_sampled_logits.append(tuple(float(value) for value in sampled))
        if not np.any(sampled):
            return base
        if age < 0.04:
            envelope = age / 0.04
        elif age <= 0.16:
            envelope = 1.0
        else:
            envelope = (0.32 - age) / 0.16
        if envelope <= 0:
            return base
        target = np.asarray(base, dtype=np.float64)
        target[:12] = np.clip(target[:12] + 0.15 * envelope * np.tanh(sampled), -0.35, 0.35)
        delta = float(np.max(np.abs(target[:12] - np.asarray(base)[:12])))
        if delta:
            self.post_active_frames += 1
            self.post_peak_residual_rad = max(self.post_peak_residual_rad, delta)
        return tuple(float(value) for value in target)
