"""Early, geometry-informed bounded A1 clearance experiment for either foot."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rosclaw_soccer.rsi.receiving_whole_body_residual import ReceivingWholeBodyResidual
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingShinClearanceResidual(ReceivingWholeBodyResidual):
    """Searchable hip/knee/ankle pre-pose, not a deployed motion policy.

    Input uses only current ball-side and completed own-foot contact. The
    independent A1 cursor still limits amplitude and per-frame target changes.
    """

    hip_pitch_rad: float = 0.0
    knee_rad: float = 0.0
    ankle_pitch_rad: float = 0.0
    post_contact_hold_sec: float = 0.12

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.coefficients != (0.0,) * 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 0.1
                for value in (self.hip_pitch_rad, self.knee_rad, self.ankle_pitch_rad)
            )
            or self.post_contact_hold_sec not in (0.08, 0.12, 0.2)
        ):
            raise ValueError("bounded clearance-only body residual required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_shin_clearance_residual.v1",
                "parent_contract_hash": self.contract_hash,
                "hip_pitch_rad": self.hip_pitch_rad,
                "knee_rad": self.knee_rad,
                "ankle_pitch_rad": self.ankle_pitch_rad,
                "post_contact_hold_sec": self.post_contact_hold_sec,
                "source_geometry_report_hash": (
                    "sha256:4b0a3d7275370f64dd0ffe5482c2fc032c18095dc47bb17bb84898af1fd642f6"
                ),
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        # Parent validates same-agent, complete contact, time and frame; its
        # zero coefficients do not exert a physical action.
        super().propose(observation)
        snapshot = self.mailbox.snapshot
        touched = snapshot.first_own_foot_time_sec
        if touched is None:
            active = True
            right = observation.qpos[37] < observation.qpos[1]
        else:
            active = observation.time_sec - touched <= self.post_contact_hold_sec
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
