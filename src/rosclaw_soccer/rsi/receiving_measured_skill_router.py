"""SIM_ONLY measured-state routing over sealed receiving specialist demonstrations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_middle_basis_expert import ReceivingMiddleBasisExpert
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass(frozen=True)
class ReceivingSkillKnot:
    lateral_m: float
    closing_speed_mps: float
    expert: str
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.lateral_m) is not float
            or not math.isfinite(self.lateral_m)
            or not 0.10 <= self.lateral_m <= 0.18
            or type(self.closing_speed_mps) is not float
            or not math.isfinite(self.closing_speed_mps)
            or not -2.0 <= self.closing_speed_mps <= -0.4
            or type(self.expert) is not str
            or re.fullmatch(r"[a-z_]{1,24}", self.expert) is None
            or type(self.weights) is not tuple
            or len(self.weights) != 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1
                for value in self.weights
            )
        ):
            raise ValueError("finite measured specialist knot required")


@dataclass
class ReceivingMeasuredSkillRouter(ReceivingMiddleBasisExpert):
    """Choose one skill once from the first observed ball/body state."""

    skill_knots: tuple[ReceivingSkillKnot, ...] = ()
    lateral_scale_m: float = 0.006
    speed_scale_mps: float = 0.12
    radius: float = 1.0
    selected_expert: str | None = field(init=False, default=None)
    selected_distance: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.skill_knots) is not tuple
            or not 1 <= len(self.skill_knots) <= 16
            or any(not isinstance(knot, ReceivingSkillKnot) for knot in self.skill_knots)
            or type(self.lateral_scale_m) is not float
            or not 0.002 <= self.lateral_scale_m <= 0.02
            or type(self.speed_scale_mps) is not float
            or not 0.04 <= self.speed_scale_mps <= 0.3
            or type(self.radius) is not float
            or not 0.5 <= self.radius <= 1.5
            or self.middle_weights != (0.0,) * 12
        ):
            raise ValueError("bounded immutable skill router and zero fallback required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_measured_skill_router.v1",
                "parent_contract_hash": self.contract_hash,
                "skill_knots": [vars(knot) for knot in self.skill_knots],
                "lateral_scale_m": self.lateral_scale_m,
                "speed_scale_mps": self.speed_scale_mps,
                "radius": self.radius,
                "fallback": "parent",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if self.selected_expert is None:
            features = ReceivingTemporalMotorExpert.features(observation)
            lateral = features[1] * 0.2
            closing = features[3] * 2.0
            distances = [
                math.hypot(
                    (lateral - knot.lateral_m) / self.lateral_scale_m,
                    (closing - knot.closing_speed_mps) / self.speed_scale_mps,
                )
                for knot in self.skill_knots
            ]
            index = int(np.argmin(distances))
            self.selected_distance = distances[index]
            if distances[index] <= self.radius:
                knot = self.skill_knots[index]
                self.middle_weights = knot.weights
                self.selected_expert = knot.expert
            else:
                self.selected_expert = "parent"
        return super().propose(observation)
