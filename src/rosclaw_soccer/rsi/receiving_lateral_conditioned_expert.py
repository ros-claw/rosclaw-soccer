"""SIM_ONLY measured-lateral receiving residual on one guarded A2 authority."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_side_conditioned_expert import ReceivingSideConditionedExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingLateralConditionedExpert(ReceivingSideConditionedExpert):
    left_lateral_slope: tuple[float, ...] = (0.0,) * 12
    right_lateral_slope: tuple[float, ...] = (0.0,) * 12
    # The first feedback frame is after the receiver has already moved, so
    # this anchor refers to measured ball-to-pelvis lateral, not launch offset.
    anchor_lateral_m: float = 0.148
    scale_lateral_m: float = 0.018
    lateral_deadband_m: float = 0.003
    measured_lateral_m: float | None = field(init=False, default=None)
    _normalized_offset: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            any(
                type(weights) is not tuple
                or len(weights) != 12
                or any(type(v) is not float or not math.isfinite(v) or abs(v) > 1 for v in weights)
                for weights in (self.left_lateral_slope, self.right_lateral_slope)
            )
            or type(self.anchor_lateral_m) is not float
            or not 0.10 <= self.anchor_lateral_m <= 0.20
            or type(self.scale_lateral_m) is not float
            or not 0.01 <= self.scale_lateral_m <= 0.05
            or type(self.lateral_deadband_m) is not float
            or not 0 <= self.lateral_deadband_m <= 0.005
        ):
            raise ValueError("bounded measured-lateral A2 receiving slope required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_lateral_conditioned_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "left_lateral_slope": self.left_lateral_slope,
                "right_lateral_slope": self.right_lateral_slope,
                "anchor_lateral_m": self.anchor_lateral_m,
                "scale_lateral_m": self.scale_lateral_m,
                "lateral_deadband_m": self.lateral_deadband_m,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        if self._normalized_offset is None:
            lateral = float(observation.qpos[37] - observation.qpos[1])
            if not math.isfinite(lateral):
                raise ValueError("finite measured ball-body lateral required")
            self.measured_lateral_m = lateral
            offset = abs(lateral) - self.anchor_lateral_m
            offset = math.copysign(max(abs(offset) - self.lateral_deadband_m, 0.0), offset)
            self._normalized_offset = float(np.clip(offset / self.scale_lateral_m, -1.5, 1.5))
        assert self._selected_side is not None
        slope = self.right_lateral_slope if self._selected_side else self.left_lateral_slope
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
        base[:12] += 0.25 * fraction * self._normalized_offset * np.asarray(slope, dtype=np.float64)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
