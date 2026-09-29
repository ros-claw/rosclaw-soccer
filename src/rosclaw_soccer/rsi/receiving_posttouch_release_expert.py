"""SIM_ONLY measured-foot transition from A2 pre-contact stance to release."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingPosttouchReleaseExpert(ReceivingPrecontactExpert):
    release_frames: int = 2
    _selected_side: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.release_frames) is not int or not 0 <= self.release_frames <= 8:
            raise ValueError("bounded measured-contact release window required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_posttouch_release_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "release_frames": self.release_frames,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(ReceivingCoordinatedFeedback.propose(self, observation), dtype=np.float64)
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
        first_time = self.mailbox.snapshot.first_own_foot_time_sec
        if first_time is not None:
            age_frames = max(0.0, (observation.time_sec - first_time) / 0.02)
            if self.release_frames == 0:
                fraction = 0.0
            else:
                fraction = min(fraction, max(0.0, 1.0 - age_frames / self.release_frames))
        base[:12] += 0.25 * fraction * np.asarray(weights, dtype=np.float64)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(v) for v in base)
