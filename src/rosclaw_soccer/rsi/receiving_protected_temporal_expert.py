"""SIM_ONLY hard parent replay for measured, predeclared retention states."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingProtectedTemporalExpert(ReceivingTemporalMotorExpert):
    protected_initial_features: tuple[tuple[float, ...], ...] = ()
    protection_radius: float = 0.006
    protected_episode: bool | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.protected_initial_features) is not tuple
            or len(self.protected_initial_features) != 2
            or any(
                type(row) is not tuple
                or len(row) != 10
                or any(
                    type(value) is not float or not math.isfinite(value) or abs(value) > 2
                    for value in row
                )
                for row in self.protected_initial_features
            )
            or type(self.protection_radius) is not float
            or not 0.001 <= self.protection_radius <= 0.02
        ):
            raise ValueError("two finite bounded parent retention states required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_protected_temporal_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "protected_initial_features": self.protected_initial_features,
                "protection_radius": self.protection_radius,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if self.protected_episode is None and observation.frame >= 15:
            measured = np.asarray(self.features(observation))
            self.protected_episode = any(
                np.linalg.norm(measured - np.asarray(row)) <= self.protection_radius
                for row in self.protected_initial_features
            )
        if self.protected_episode:
            return ReceivingLateralPiecewiseExpert.propose(self, observation)
        return super().propose(observation)
