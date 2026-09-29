"""SIM_ONLY pre-impact orientation exploration over the guarded A1 feedback slot."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingImpactOrientationFeedback(ReceivingCoordinatedFeedback):
    """Change pre-touch hip roll/yaw while preserving the learned body synergies."""

    impact_roll: float = 0.0
    impact_yaw: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1
            for x in (self.impact_roll, self.impact_yaw)
        ):
            raise ValueError("bounded finite pre-impact orientation required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_impact_orientation_feedback.v1",
                "parent_contract_hash": self.contract_hash,
                "impact_roll": self.impact_roll,
                "impact_yaw": self.impact_yaw,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        action = np.asarray(super().propose(observation), dtype=np.float64)
        if self.mailbox.snapshot.first_own_foot_time_sec is not None:
            return tuple(float(x) for x in action)
        if abs(observation.qpos[36] - observation.qpos[0]) > 1.2:
            return tuple(float(x) for x in action)
        right = observation.qpos[37] < observation.qpos[1]
        offset = 6 if right else 0
        mirror = -1.0 if right else 1.0
        action[offset + 1] += 0.08 * mirror * self.impact_roll
        action[offset + 2] += 0.08 * mirror * self.impact_yaw
        action[offset + 5] -= 0.04 * mirror * self.impact_roll
        np.clip(action, -0.1, 0.1, out=action)
        return tuple(float(x) for x in action)
