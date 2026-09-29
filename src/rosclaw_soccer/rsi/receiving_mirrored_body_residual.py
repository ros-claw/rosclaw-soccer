"""Mirror the bounded A1 contact-phase receiver using current ball/foot facts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.receiving_whole_body_residual import ReceivingWholeBodyResidual
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation

_MIRROR_ORDER = np.asarray(
    (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14)
    + tuple(range(22, 29))
    + tuple(range(15, 22)),
    dtype=np.int64,
)
_MIRROR_SIGN = np.asarray(
    (1.0, -1.0, -1.0, 1.0, 1.0, -1.0) * 2
    + (-1.0, -1.0, 1.0)
    + (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0) * 2,
    dtype=np.float64,
)


@dataclass
class ReceivingMirroredBodyResidual(ReceivingWholeBodyResidual):
    """One learned coefficient vector shared by either actual receiving foot.

    Pre-contact side comes from the *current* ball/pelvis lateral position.
    Post-contact side comes from completed native foot-contact attribution.
    Neither branch can access a future ball path or simulator state handle.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_mirrored_body_residual.v1",
                "parent_contract_hash": self.contract_hash,
                "mirror_order": _MIRROR_ORDER.tolist(),
                "mirror_sign": _MIRROR_SIGN.tolist(),
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        raw = super().propose(observation)
        snapshot = self.mailbox.snapshot
        if snapshot.first_own_foot is None:
            right = observation.qpos[37] < observation.qpos[1]
        elif snapshot.first_own_foot == "right_foot":
            right = True
        elif snapshot.first_own_foot == "left_foot":
            right = False
        else:
            raise ValueError("only a measured left/right foot may select a mirrored action")
        if not right:
            return raw
        mirrored = np.asarray(raw, dtype=np.float64)[_MIRROR_ORDER] * _MIRROR_SIGN
        return tuple(float(x) for x in mirrored)
