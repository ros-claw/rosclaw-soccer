"""SIM_ONLY side-conditioned pre-contact receiving expert on one A2 authority.

The A2 oracle cursor, not this actor, owns the bounded and rate-limited joint
target residual. This actor never receives a writable simulator or motor handle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingPrecontactExpert(ReceivingCoordinatedFeedback):
    left_weights: tuple[float, ...] = (0.0,) * 12
    right_weights: tuple[float, ...] = (0.0,) * 12
    action_substrate: str = field(init=False, default="A2_body29_precontact")
    _selected_side: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(
            type(weights) is not tuple
            or len(weights) != 12
            or any(
                type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1 for v in weights
            )
            for weights in (self.left_weights, self.right_weights)
        ):
            raise ValueError("bounded bilateral pre-contact expert weights required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_precontact_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "left_weights": self.left_weights,
                "right_weights": self.right_weights,
                "action_substrate": self.action_substrate,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        if self._selected_side is None:
            self._selected_side = int(observation.qpos[37] < observation.qpos[1])
        weights = self.right_weights if self._selected_side else self.left_weights
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
        base[:12] += 0.25 * fraction * np.asarray(weights, dtype=np.float64)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(v) for v in base)
