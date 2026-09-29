"""SIM_ONLY state-routed composition of measured local receiving motor skills."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseRouter
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingComposedContactRouter(ReceivingFootPhaseRouter):
    """High: teacher-relative foot feedback; low: learned post-touch support.

    Center uses its unchanged previously trained middle-basis skill. The
    selected expert depends only on the first measured ball/body state.
    """

    low_post_weights: tuple[float, ...] = (0.0,) * 12
    low_first_contact_time_sec: float | None = field(init=False, default=None)
    low_active_frames: int = field(init=False, default=0)
    low_peak_residual_rad: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.corrected_experts != ("high",)
            or type(self.low_post_weights) is not tuple
            or len(self.low_post_weights) != 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1
                for value in self.low_post_weights
            )
        ):
            raise ValueError("one bounded low specialist and high-only foot correction required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_composed_contact_router.v1",
                "parent_contract_hash": self.contract_hash,
                "low_post_weights": self.low_post_weights,
                "low_trigger": "measured_first_own_foot_contact",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = super().propose(observation)
        if self.selected_expert != "low" or not any(self.low_post_weights):
            return base
        if self.low_first_contact_time_sec is None:
            self.low_first_contact_time_sec = observation.last_own_foot_contact_time_sec
        if self.low_first_contact_time_sec is None:
            return base
        elapsed = observation.time_sec - self.low_first_contact_time_sec
        if elapsed < 0 or elapsed >= 0.4:
            return base
        if elapsed < 0.04:
            fraction = elapsed / 0.04
        elif elapsed <= 0.18:
            fraction = 1.0
        else:
            fraction = (0.4 - elapsed) / 0.22
        if fraction <= 0:
            return base
        target = np.asarray(base, dtype=np.float64)
        target[:12] = np.clip(
            target[:12] + 0.25 * fraction * np.asarray(self.low_post_weights), -0.35, 0.35
        )
        peak = float(np.max(np.abs(target[:12] - np.asarray(base)[:12])))
        if peak > 0:
            self.low_active_frames += 1
            self.low_peak_residual_rad = max(self.low_peak_residual_rad, peak)
        return tuple(float(value) for value in target)
