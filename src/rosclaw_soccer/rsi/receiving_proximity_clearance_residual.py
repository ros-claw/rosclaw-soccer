"""Current-ball-distance-gated A1 shin-clearance research actor, SIM_ONLY."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.rsi.receiving_shin_clearance_residual import ReceivingShinClearanceResidual
from rosclaw_soccer.rsi.receiving_whole_body_residual import ReceivingWholeBodyResidual
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingProximityClearanceResidual(ReceivingShinClearanceResidual):
    """Latches on current ball-pelvis XY distance, then pre-poses either leg."""

    entry_distance_m: float = 0.65
    _entered: bool = field(init=False, default=False)
    _entry_frame: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.entry_distance_m) is not float
            or not 0.3 <= self.entry_distance_m <= 1.0
            or not math.isfinite(self.entry_distance_m)
        ):
            raise ValueError("bounded measured-ball entry distance required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_proximity_clearance_residual.v1",
                "parent_contract_hash": self.contract_hash,
                "entry_distance_m": self.entry_distance_m,
                "observation": "current_measured_ball_and_pelvis_xy_only",
            }
        )

    @property
    def entry_frame(self) -> int | None:
        return self._entry_frame

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        # Bypass only the parent's action construction, never its validation.
        ReceivingWholeBodyResidual.propose(self, observation)
        snapshot = self.mailbox.snapshot
        ball_distance = math.hypot(
            observation.qpos[36] - observation.qpos[0],
            observation.qpos[37] - observation.qpos[1],
        )
        if not self._entered and ball_distance <= self.entry_distance_m:
            self._entered = True
            self._entry_frame = observation.frame
        touched = snapshot.first_own_foot_time_sec
        if touched is None:
            active = self._entered
            right = observation.qpos[37] < observation.qpos[1]
        else:
            active = self._entered and observation.time_sec - touched <= self.post_contact_hold_sec
            if snapshot.first_own_foot not in ("left_foot", "right_foot"):
                raise ValueError("completed own-foot side required")
            right = snapshot.first_own_foot == "right_foot"
        desired = [0.0] * 29
        if active:
            hip, knee, ankle = (6, 9, 10) if right else (0, 3, 4)
            desired[hip] = self.hip_pitch_rad
            desired[knee] = self.knee_rad
            desired[ankle] = self.ankle_pitch_rad
            if any(desired):
                self._nonzero_frames += 1
        return tuple(desired)
