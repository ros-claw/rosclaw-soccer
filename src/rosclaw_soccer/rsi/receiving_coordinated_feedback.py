"""SIM_ONLY bilateral receiving feedback with bounded whole-body synergies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.receiving_taskspace_feedback import ReceivingTaskspaceFeedback
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingCoordinatedFeedback(ReceivingTaskspaceFeedback):
    """Blend four learned, mirrored body synergies with measured foot feedback.

    Each coefficient is a normalized, dimensionless proposal. The A1 cursor
    remains the sole owner of the 0.1 rad amplitude and 0.02 rad/frame limits.
    """

    coordination: tuple[float, ...] = (0.0,) * 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.coordination) is not tuple
            or len(self.coordination) != 8
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1
                for x in self.coordination
            )
        ):
            raise ValueError("eight bounded finite whole-body synergy weights required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_coordinated_feedback.v1",
                "parent_contract_hash": self.contract_hash,
                "coordination": self.coordination,
                "action_substrate": "A1_body29",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        action = np.asarray(super().propose(observation), dtype=np.float64)
        snapshot = self.mailbox.snapshot
        if snapshot.first_own_foot_time_sec is None:
            right = observation.qpos[37] < observation.qpos[1]
            weights = self.coordination[:4]
            if abs(observation.qpos[36] - observation.qpos[0]) > 1.2:
                return tuple(float(x) for x in action)
        else:
            if snapshot.first_own_foot not in ("left_foot", "right_foot"):
                raise ValueError("measured receiving foot required")
            right = snapshot.first_own_foot == "right_foot"
            weights = self.coordination[4:]
            if observation.time_sec - snapshot.first_own_foot_time_sec > 1.2:
                return tuple(float(x) for x in action)
        kick = 6 if right else 0
        support = 0 if right else 6
        # Swing-foot retraction, ankle-preserving hip motion, planted-leg
        # counterstep, and upper-body counterbalance. No ball state is edited.
        basis = np.zeros((4, 29), dtype=np.float64)
        basis[0, kick + 3] = 1.0
        basis[0, kick + 4] = -0.5
        basis[1, kick] = 0.8
        basis[1, kick + 4] = -0.8
        basis[2, support] = -0.6
        basis[2, support + 3] = 0.8
        basis[2, support + 4] = -0.4
        basis[3, 12] = 0.5
        basis[3, 15 if right else 22] = -0.5
        basis[3, 22 if right else 15] = 0.5
        action += 0.08 * np.asarray(weights, dtype=np.float64) @ basis
        np.clip(action, -0.1, 0.1, out=action)
        return tuple(float(x) for x in action)
