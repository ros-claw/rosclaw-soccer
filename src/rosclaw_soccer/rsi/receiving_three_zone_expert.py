"""SIM_ONLY measured-state selector of sealed receiving muscle-memory experts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.rsi.receiving_middle_basis_expert import ReceivingMiddleBasisExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingThreeZoneExpert(ReceivingMiddleBasisExpert):
    low_weights: tuple[float, ...] = (0.0,) * 12
    center_weights: tuple[float, ...] = (0.0,) * 12
    high_weights: tuple[float, ...] = (0.0,) * 12
    low_boundary_m: float = 0.137
    high_boundary_m: float = 0.142
    selected_zone: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            any(
                type(weights) is not tuple
                or len(weights) != 12
                or any(
                    type(value) is not float or not math.isfinite(value) or abs(value) > 1
                    for value in weights
                )
                for weights in (self.low_weights, self.center_weights, self.high_weights)
            )
            or type(self.low_boundary_m) is not float
            or type(self.high_boundary_m) is not float
            or not 0.13 <= self.low_boundary_m < self.high_boundary_m <= 0.15
        ):
            raise ValueError("bounded measured-state three-zone receiving experts required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_three_zone_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "low_weights": self.low_weights,
                "center_weights": self.center_weights,
                "high_weights": self.high_weights,
                "low_boundary_m": self.low_boundary_m,
                "high_boundary_m": self.high_boundary_m,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if self.selected_zone is None:
            lateral = float(observation.qpos[37] - observation.qpos[1])
            if not math.isfinite(lateral):
                raise ValueError("finite measured initial ball-body lateral required")
            absolute = abs(lateral)
            if absolute < self.low_boundary_m:
                self.selected_zone = "low"
                self.middle_weights = self.low_weights
            elif absolute > self.high_boundary_m:
                self.selected_zone = "high"
                self.middle_weights = self.high_weights
            else:
                self.selected_zone = "center"
                self.middle_weights = self.center_weights
        return super().propose(observation)
