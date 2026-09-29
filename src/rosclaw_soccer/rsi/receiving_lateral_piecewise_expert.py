"""SIM_ONLY bounded piecewise measured-lateral residual for receiving.

The first segment is frozen after training on near and middle launches. The
second segment may be trained without changing either previously passed point.
No simulator or motor authority is held by this actor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_side_conditioned_expert import ReceivingSideConditionedExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingLateralPiecewiseExpert(ReceivingSideConditionedExpert):
    old_lateral_slope: tuple[float, ...] = (0.0,) * 12
    far_lateral_slope: tuple[float, ...] = (0.0,) * 12
    near_lateral_m: float = 0.1307
    middle_lateral_m: float = 0.14861864755818321
    far_lateral_m: float = 0.16516459772735814
    lateral_deadband_m: float = 0.003
    old_slope_scale_m: float = 0.018
    measured_lateral_m: float | None = field(init=False, default=None)
    _old_feature: float | None = field(init=False, default=None)
    _far_feature: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            any(
                type(weights) is not tuple
                or len(weights) != 12
                or any(type(v) is not float or not math.isfinite(v) or abs(v) > 1 for v in weights)
                for weights in (self.old_lateral_slope, self.far_lateral_slope)
            )
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    self.near_lateral_m,
                    self.middle_lateral_m,
                    self.far_lateral_m,
                    self.lateral_deadband_m,
                    self.old_slope_scale_m,
                )
            )
            or not 0.10 <= self.near_lateral_m < self.middle_lateral_m < self.far_lateral_m <= 0.25
            or not 0 <= self.lateral_deadband_m < self.far_lateral_m - self.middle_lateral_m
            or not 0.01 <= self.old_slope_scale_m <= 0.05
        ):
            raise ValueError("bounded piecewise measured-lateral receiver required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_lateral_piecewise_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "old_lateral_slope": self.old_lateral_slope,
                "far_lateral_slope": self.far_lateral_slope,
                "near_lateral_m": self.near_lateral_m,
                "middle_lateral_m": self.middle_lateral_m,
                "far_lateral_m": self.far_lateral_m,
                "lateral_deadband_m": self.lateral_deadband_m,
                "old_slope_scale_m": self.old_slope_scale_m,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        if self._old_feature is None or self._far_feature is None:
            lateral = float(observation.qpos[37] - observation.qpos[1])
            if not math.isfinite(lateral):
                raise ValueError("finite measured ball-body lateral required")
            self.measured_lateral_m = lateral
            absolute = abs(lateral)
            middle_offset = self.middle_lateral_m - self.near_lateral_m
            old_limit = (middle_offset - self.lateral_deadband_m) / self.old_slope_scale_m
            self._old_feature = float(
                np.clip(
                    (absolute - self.near_lateral_m - self.lateral_deadband_m)
                    / self.old_slope_scale_m,
                    0.0,
                    old_limit,
                )
            )
            self._far_feature = float(
                np.clip(
                    (absolute - self.middle_lateral_m - self.lateral_deadband_m)
                    / (self.far_lateral_m - self.middle_lateral_m - self.lateral_deadband_m),
                    0.0,
                    1.0,
                )
            )
        assert self._selected_side is not None
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
        base[:12] += (
            0.25
            * fraction
            * (
                self._old_feature * np.asarray(self.old_lateral_slope, dtype=np.float64)
                + self._far_feature * np.asarray(self.far_lateral_slope, dtype=np.float64)
            )
        )
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
