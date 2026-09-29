"""SIM_ONLY compact-support measured-state receiving skill basis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingMiddleBasisExpert(ReceivingLateralPiecewiseExpert):
    middle_weights: tuple[float, ...] = (0.0,) * 12
    middle_center_m: float = 0.1395
    middle_support_radius_m: float = 0.0085
    middle_feature: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.middle_weights) is not tuple
            or len(self.middle_weights) != 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1
                for value in self.middle_weights
            )
            or type(self.middle_center_m) is not float
            or type(self.middle_support_radius_m) is not float
            or not 0.13 <= self.middle_center_m <= 0.15
            or not 0.005 <= self.middle_support_radius_m <= 0.009
        ):
            raise ValueError("bounded measured middle-course skill basis required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_middle_basis_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "middle_weights": self.middle_weights,
                "middle_center_m": self.middle_center_m,
                "middle_support_radius_m": self.middle_support_radius_m,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        if self.middle_feature is None:
            if self.measured_lateral_m is None:
                raise ValueError("measured initial ball-body lateral required")
            distance = abs(abs(self.measured_lateral_m) - self.middle_center_m)
            self.middle_feature = (
                0.0
                if distance >= self.middle_support_radius_m
                else 0.5 * (1.0 + math.cos(math.pi * distance / self.middle_support_radius_m))
            )
        if self._selected_side or self.middle_feature <= 0:
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
        base[:12] += (
            0.25
            * fraction
            * self.middle_feature
            * np.asarray(self.middle_weights, dtype=np.float64)
        )
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
