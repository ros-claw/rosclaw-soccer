"""SIM_ONLY early admission for the bounded, side-conditioned A2 expert."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingEarlyPrecontactExpert(ReceivingPrecontactExpert):
    entry_frame: int = 10
    full_frame: int = 18
    _selected_side: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.entry_frame) is not int
            or type(self.full_frame) is not int
            or not 5 <= self.entry_frame < 15
            or not self.entry_frame + 3 <= self.full_frame <= 20
        ):
            raise ValueError("bounded early A2 entry and full-target frames required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_early_precontact_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "entry_frame": self.entry_frame,
                "full_frame": self.full_frame,
                "base_authority_start_frame": 15,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(ReceivingCoordinatedFeedback.propose(self, observation), dtype=np.float64)
        if self._selected_side is None:
            self._selected_side = int(observation.qpos[37] < observation.qpos[1])
        if observation.frame < 15:
            base[:] = 0.0
        weights = self.right_weights if self._selected_side else self.left_weights
        frame = observation.frame
        if frame < self.entry_frame:
            fraction = 0.0
        elif frame < self.full_frame:
            fraction = (frame - self.entry_frame) / (self.full_frame - self.entry_frame)
        elif frame <= 30:
            fraction = 1.0
        elif frame < 40:
            fraction = (40 - frame) / 10.0
        else:
            fraction = 0.0
        base[:12] += 0.25 * fraction * np.asarray(weights, dtype=np.float64)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(v) for v in base)
