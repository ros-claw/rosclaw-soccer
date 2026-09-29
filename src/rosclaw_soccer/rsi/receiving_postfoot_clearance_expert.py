"""SIM_ONLY bounded event-conditioned post-foot shin-clearance action."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingPostfootClearanceExpert(ReceivingLateralPiecewiseExpert):
    postfoot_left_hip_knee_ankle: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.postfoot_left_hip_knee_ankle) is not tuple
            or len(self.postfoot_left_hip_knee_ankle) != 3
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 0.12
                for value in self.postfoot_left_hip_knee_ankle
            )
        ):
            raise ValueError("bounded left post-foot clearance coefficients required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_postfoot_clearance_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "postfoot_left_hip_knee_ankle": self.postfoot_left_hip_knee_ankle,
                "activation_ceiling": "SIM_ONLY",
                "event_source": "completed_attributed_own_foot_contact",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        snapshot = self.mailbox.snapshot
        touched = snapshot.first_own_foot_time_sec
        if touched is None or snapshot.first_own_foot != "left_foot":
            return tuple(float(value) for value in base)
        elapsed = observation.time_sec - touched
        if not 0 <= elapsed <= 0.14:
            return tuple(float(value) for value in base)
        # Begin after a completed physics contact and fade before the next
        # control phase. The oracle, not this actor, owns rate and torque limits.
        gain = min(1.0, (elapsed + 0.02) / 0.04) * (1.0 - elapsed / 0.14)
        for joint, weight in zip((0, 3, 4), self.postfoot_left_hip_knee_ankle, strict=True):
            base[joint] += gain * weight
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
