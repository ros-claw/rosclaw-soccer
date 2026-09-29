"""SIM_ONLY measured predictive shin-clearance reflex before ball contact."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingPredictiveShinExpert(ReceivingLateralPiecewiseExpert):
    shin_target_gap_m: float = 0.03
    shin_prediction_horizon_sec: float = 0.06
    shin_knee_gain_rad: float = 0.08
    requires_shin_clearance: bool = field(init=False, default=True)
    _previous_shin_gap_m: float | None = field(init=False, default=None)
    _previous_shin_frame: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    self.shin_target_gap_m,
                    self.shin_prediction_horizon_sec,
                    self.shin_knee_gain_rad,
                )
            )
            or not 0.01 <= self.shin_target_gap_m <= 0.08
            or not 0.02 <= self.shin_prediction_horizon_sec <= 0.10
            or not 0 <= self.shin_knee_gain_rad <= 0.12
        ):
            raise ValueError("bounded measured predictive shin reflex required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_predictive_shin_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "shin_target_gap_m": self.shin_target_gap_m,
                "shin_prediction_horizon_sec": self.shin_prediction_horizon_sec,
                "shin_knee_gain_rad": self.shin_knee_gain_rad,
                "requires_shin_clearance": True,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        measured = observation.shin_clearance
        if measured is None:
            raise ValueError("same-frame measured shin clearance required")
        gap = float(measured.clearance_m[0])
        if (
            self._previous_shin_frame is not None
            and observation.frame != self._previous_shin_frame + 1
        ):
            raise ValueError("consecutive measured shin states required")
        closing_mps = (
            0.0
            if self._previous_shin_gap_m is None
            else max(0.0, (self._previous_shin_gap_m - gap) / 0.02)
        )
        self._previous_shin_gap_m = gap
        self._previous_shin_frame = observation.frame
        if self._selected_side or not 18 <= observation.frame < 35:
            return tuple(float(value) for value in base)
        predicted_gap = gap - self.shin_prediction_horizon_sec * closing_mps
        risk = float(np.clip((self.shin_target_gap_m - predicted_gap) / 0.04, 0.0, 1.0))
        gradient = float(measured.gradient_m_per_rad[0][3])
        if abs(gradient) >= 1e-4:
            base[3] += math.copysign(self.shin_knee_gain_rad * risk, gradient)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
