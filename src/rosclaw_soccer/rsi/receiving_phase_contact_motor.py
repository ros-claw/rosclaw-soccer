"""SIM_ONLY measured-foot-triggered post-contact neural motor phase."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingPhaseContactMotor(ReceivingProtectedComposedNeural):
    """Add bounded smooth support action only after measured own-foot contact."""

    post_contact_logits: tuple[float, ...] = (0.0,) * 12
    first_contact_time_sec: float | None = field(init=False, default=None)
    post_active_frames: int = field(init=False, default=0)
    post_peak_residual_rad: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.post_contact_logits) is not tuple
            or len(self.post_contact_logits) != 12
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1.2
                for value in self.post_contact_logits
            )
        ):
            raise ValueError("bounded finite post-contact motor logits required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_phase_contact_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "post_contact_logits": self.post_contact_logits,
                "trigger": "first_measured_own_foot_contact",
                "fade_in_sec": 0.04,
                "fade_out_start_sec": 0.16,
                "fade_out_end_sec": 0.32,
                "peak_amplitude_rad": 0.15,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = super().propose(observation)
        if (
            self.protected_episode
            or self.selected_expert != "high"
            or not any(self.post_contact_logits)
        ):
            return base
        if self.first_contact_time_sec is None:
            self.first_contact_time_sec = observation.last_own_foot_contact_time_sec
        if self.first_contact_time_sec is None:
            return base
        age = observation.time_sec - self.first_contact_time_sec
        if age < 0 or age >= 0.32:
            return base
        if age < 0.04:
            envelope = age / 0.04
        elif age <= 0.16:
            envelope = 1.0
        else:
            envelope = (0.32 - age) / 0.16
        if envelope <= 0:
            return base
        target = np.asarray(base, dtype=np.float64)
        target[:12] = np.clip(
            target[:12] + 0.15 * envelope * np.tanh(self.post_contact_logits), -0.35, 0.35
        )
        delta = float(np.max(np.abs(target[:12] - np.asarray(base)[:12])))
        if delta:
            self.post_active_frames += 1
            self.post_peak_residual_rad = max(self.post_peak_residual_rad, delta)
        return tuple(float(value) for value in target)
