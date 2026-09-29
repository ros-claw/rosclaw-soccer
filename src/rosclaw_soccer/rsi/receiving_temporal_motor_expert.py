"""SIM_ONLY 50 Hz proprioceptive neural receiving policy under the A2 guard."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass(frozen=True)
class TemporalMotorWeights:
    input_matrix: tuple[float, ...] = (0.0,) * 320
    input_bias: tuple[float, ...] = (0.0,) * 32
    output_matrix: tuple[float, ...] = (0.0,) * 384
    output_bias: tuple[float, ...] = (0.0,) * 12

    def __post_init__(self) -> None:
        for values, size in (
            (self.input_matrix, 320),
            (self.input_bias, 32),
            (self.output_matrix, 384),
            (self.output_bias, 12),
        ):
            if (
                type(values) is not tuple
                or len(values) != size
                or any(
                    type(value) is not float or not math.isfinite(value) or abs(value) > 10
                    for value in values
                )
            ):
                raise ValueError("finite bounded temporal motor policy required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.rsi.temporal_motor_weights.v1",
                    "input_matrix": self.input_matrix,
                    "input_bias": self.input_bias,
                    "output_matrix": self.output_matrix,
                    "output_bias": self.output_bias,
                }
            )
        )


@dataclass
class ReceivingTemporalMotorExpert(ReceivingLateralPiecewiseExpert):
    policy: TemporalMotorWeights = field(default_factory=TemporalMotorWeights)
    exploration_std: float = 0.0
    exploration_seed: int = 0
    exploration_correlation: float = 0.0
    observed_frames: list[int] = field(init=False, default_factory=list)
    observed_features: list[tuple[float, ...]] = field(init=False, default_factory=list)
    sampled_logits: list[tuple[float, ...]] = field(init=False, default_factory=list)
    _rng: np.random.Generator = field(init=False)
    _previous_exploration: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.policy, TemporalMotorWeights)
            or type(self.exploration_std) is not float
            or not math.isfinite(self.exploration_std)
            or not 0 <= self.exploration_std <= 0.3
            or type(self.exploration_seed) is not int
            or not 0 <= self.exploration_seed < 2**31
            or type(self.exploration_correlation) is not float
            or not math.isfinite(self.exploration_correlation)
            or not 0 <= self.exploration_correlation < 0.99
        ):
            raise ValueError("bounded SIM_ONLY temporal exploration required")
        self._rng = np.random.default_rng(self.exploration_seed)
        contract = {
            "schema": "rosclaw_soccer.rsi.receiving_temporal_motor_expert.v1",
            "parent_contract_hash": self.contract_hash,
            "policy_hash": self.policy.contract_hash,
            "exploration_std": self.exploration_std,
            "exploration_seed": self.exploration_seed,
            "observation": "current_50hz_ball_body_contact_history",
            "activation_ceiling": "SIM_ONLY",
        }
        if self.exploration_correlation:
            contract["exploration_correlation"] = self.exploration_correlation
        self.contract_hash = hash_json(contract)

    @staticmethod
    def features(observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        qpos = np.asarray(observation.qpos, dtype=np.float64)
        qvel = np.asarray(observation.qvel, dtype=np.float64)
        ball_delta = qpos[36:39] - qpos[:3]
        ball_velocity = qvel[35:38] - qvel[:3]
        features = np.asarray(
            (
                ball_delta[0] / 1.0,
                ball_delta[1] / 0.2,
                ball_delta[2] / 0.5,
                ball_velocity[0] / 2.0,
                ball_velocity[1] / 2.0,
                qvel[0] / 2.0,
                qvel[1] / 2.0,
                (qpos[2] - 0.7) / 0.3,
                (observation.frame - 15) / 50.0,
                float(observation.last_own_foot_contact_time_sec is not None),
            ),
            dtype=np.float64,
        )
        if not np.isfinite(features).all():
            raise ValueError("finite measured 50 Hz body/ball/contact state required")
        return tuple(float(value) for value in np.clip(features, -2.0, 2.0))

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        frame = observation.frame
        if self._selected_side or not 15 <= frame < 65:
            return tuple(float(value) for value in base)
        features = self.features(observation)
        values = np.asarray(features, dtype=np.float64)
        hidden = np.tanh(
            np.asarray(self.policy.input_matrix).reshape(32, 10) @ values
            + np.asarray(self.policy.input_bias)
        )
        logits = np.asarray(self.policy.output_matrix).reshape(12, 32) @ hidden + np.asarray(
            self.policy.output_bias
        )
        if self.exploration_std:
            innovation = self._rng.normal(0.0, self.exploration_std, size=12)
            if self.exploration_correlation:
                if self._previous_exploration is None:
                    noise = innovation
                else:
                    noise = (
                        self.exploration_correlation * self._previous_exploration
                        + math.sqrt(1 - self.exploration_correlation**2) * innovation
                    )
                self._previous_exploration = noise
                logits += noise
            else:
                logits += innovation
        if frame < 19:
            fraction = (frame - 15) / 4.0
        elif frame <= 50:
            fraction = 1.0
        else:
            fraction = (65 - frame) / 15.0
        base[:12] += 0.25 * fraction * np.tanh(logits)
        np.clip(base, -0.35, 0.35, out=base)
        self.observed_frames.append(frame)
        self.observed_features.append(features)
        self.sampled_logits.append(tuple(float(value) for value in logits))
        return tuple(float(value) for value in base)
