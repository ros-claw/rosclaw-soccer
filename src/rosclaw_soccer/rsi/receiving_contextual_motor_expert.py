"""SIM_ONLY proprioceptive contextual neural residual with one guarded A2 owner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass(frozen=True)
class ContextualMotorWeights:
    input_matrix: tuple[float, ...] = (0.0,) * 32
    input_bias: tuple[float, ...] = (0.0,) * 16
    output_matrix: tuple[float, ...] = (0.0,) * 192
    output_bias: tuple[float, ...] = (0.0,) * 12

    def __post_init__(self) -> None:
        for values, size in (
            (self.input_matrix, 32),
            (self.input_bias, 16),
            (self.output_matrix, 192),
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
                raise ValueError("finite bounded contextual motor policy required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.rsi.contextual_motor_weights.v1",
                    "input_matrix": self.input_matrix,
                    "input_bias": self.input_bias,
                    "output_matrix": self.output_matrix,
                    "output_bias": self.output_bias,
                }
            )
        )


@dataclass
class ReceivingContextualMotorExpert(ReceivingLateralPiecewiseExpert):
    policy: ContextualMotorWeights = field(default_factory=ContextualMotorWeights)
    exploration_noise: tuple[float, ...] = (0.0,) * 12
    measured_context: tuple[float, float] | None = field(init=False, default=None)
    selected_raw_action: tuple[float, ...] | None = field(init=False, default=None)
    selected_action: tuple[float, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.policy, ContextualMotorWeights)
            or type(self.exploration_noise) is not tuple
            or len(self.exploration_noise) != 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1
                for value in self.exploration_noise
            )
        ):
            raise ValueError("bounded SIM_ONLY contextual exploration required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_contextual_motor_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "policy_hash": self.policy.contract_hash,
                "exploration_noise": self.exploration_noise,
                "observation": "initial_measured_relative_lateral_and_closing_speed",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        if self.measured_context is None:
            lateral = abs(float(observation.qpos[37] - observation.qpos[1]))
            closing_speed = abs(float(observation.qvel[35] - observation.qvel[0]))
            if not math.isfinite(lateral) or not math.isfinite(closing_speed):
                raise ValueError("finite current ball-body position and velocity required")
            context = np.clip(
                np.asarray(((lateral - 0.14) / 0.015, (closing_speed - 1.25) / 0.2)),
                -2.0,
                2.0,
            )
            hidden = np.tanh(
                np.asarray(self.policy.input_matrix).reshape(16, 2) @ context
                + np.asarray(self.policy.input_bias)
            )
            raw = (
                np.asarray(self.policy.output_matrix).reshape(12, 16) @ hidden
                + np.asarray(self.policy.output_bias)
                + np.asarray(self.exploration_noise)
            )
            self.measured_context = (float(context[0]), float(context[1]))
            self.selected_raw_action = tuple(float(value) for value in raw)
            self.selected_action = tuple(float(value) for value in np.tanh(raw))
        assert self.selected_action is not None
        if self._selected_side:
            return tuple(float(value) for value in base)
        frame = observation.frame
        if frame < 15:
            fraction = 0.0
        elif frame < 19:
            fraction = (frame - 15) / 4.0
        elif frame <= 30:
            fraction = 1.0
        elif frame < 40:
            fraction = (40 - frame) / 10.0
        else:
            fraction = 0.0
        base[:12] += 0.25 * fraction * np.asarray(self.selected_action)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
