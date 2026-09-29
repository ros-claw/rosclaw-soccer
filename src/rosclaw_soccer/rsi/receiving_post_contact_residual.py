"""SIM_ONLY trainable post-contact support residual under the existing A2 guard."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingMeasuredSkillRouter
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingPostContactResidual(ReceivingMeasuredSkillRouter):
    """One learned 12-joint support action, phased by measured first own-foot touch.

    The old parent fallback is immutable. The A2 cursor, not this actor, owns
    all motor bounds, torque limits and execution authority.
    """

    post_weights: tuple[float, ...] = (0.0,) * 12
    first_contact_time_sec: float | None = field(init=False, default=None)
    active_frames: int = field(init=False, default=0)
    peak_residual_rad: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.post_weights) is not tuple
            or len(self.post_weights) != 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1
                for value in self.post_weights
            )
        ):
            raise ValueError("finite bounded 12-joint post-contact support weights required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_post_contact_residual.v1",
                "parent_contract_hash": self.contract_hash,
                "post_weights": self.post_weights,
                "trigger": "measured_first_own_foot_contact",
                "parent_fallback": "unchanged",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = super().propose(observation)
        if self.selected_expert == "parent" or not any(self.post_weights):
            return base
        if self.first_contact_time_sec is None:
            self.first_contact_time_sec = observation.last_own_foot_contact_time_sec
        if self.first_contact_time_sec is None:
            return base
        elapsed = observation.time_sec - self.first_contact_time_sec
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
            target[:12] + 0.25 * fraction * np.asarray(self.post_weights), -0.35, 0.35
        )
        peak = float(np.max(np.abs(target[:12] - np.asarray(base)[:12])))
        if peak > 0:
            self.active_frames += 1
            self.peak_residual_rad = max(self.peak_residual_rad, peak)
        return tuple(float(value) for value in target)
