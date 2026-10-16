"""Measured-state admission between SIM approach and whole-body contact heads.

No inference, simulator, interpolation, fabricated history or motion authority.
The caller must feed both heads actual measured history and record which target
and PD contract was executed. Admission is a research hypothesis, not safety.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True, slots=True)
class ContactHandoffConfig:
    entry_gap_x_m: float = 0.9
    maximum_absolute_gap_y_m: float = 0.25
    minimum_pelvis_z_m: float = 0.65
    maximum_upright_tilt_rad: float = 0.35
    earliest_frame: int = 30
    latest_frame: int = 200
    horizon_frames: int = 300
    reference_frames: int = 260
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        fixed = (
            (self.entry_gap_x_m, 0.9),
            (self.maximum_absolute_gap_y_m, 0.25),
            (self.minimum_pelvis_z_m, 0.65),
            (self.maximum_upright_tilt_rad, 0.35),
        )
        if (
            any(type(value) is not float or value != expected for value, expected in fixed)
            or type(self.earliest_frame) is not int
            or self.earliest_frame != 30
            or type(self.latest_frame) is not int
            or self.latest_frame != 200
            or type(self.horizon_frames) is not int
            or self.horizon_frames != 300
            or type(self.reference_frames) is not int
            or not 1 <= self.reference_frames <= 3000
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError(
                "fixed SIM handoff pilot and explicit bounded reference length required"
            )

    def contract(self) -> dict[str, Any]:
        value = dict(
            schema="soccer.g1.contact_handoff_config.v1",
            parameters=asdict(self),
            gap_coordinates="current world ball position minus pelvis position",
            contact_clock="hold reference0 before admission; advance from0 once latched",
            body_contact_before_admission="permanently exclude handoff for this episode",
            recurrent_history="actual measurements only; never reset at handoff or invent samples",
            old_policy_action_input="previous executed target in contact-head action coordinates",
            admission_is_physical_safety_proof=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))


@dataclass(frozen=True, slots=True)
class ContactHandoffDecision:
    frame: int
    controller: str
    reference_frame: int
    transitioned: bool
    handoff_frame: int | None
    excluded_by_previous_body_contact: bool
    activation_ceiling: str = "SIM_ONLY"


class WholeBodyContactHandoff:
    """Episode-local monotone latch, not an executor or robot capability."""

    def __init__(self, config: ContactHandoffConfig) -> None:
        if type(config) is not ContactHandoffConfig:
            raise ValueError("exact SIM handoff config required")
        config.__post_init__()
        self._config = config
        self._next_frame = 0
        self._handoff_frame: int | None = None
        self._contact_excluded = False
        self._faulted = False

    def advance(
        self,
        *,
        frame: int,
        gap_x_m: float,
        gap_y_m: float,
        pelvis_z_m: float,
        upright_tilt_rad: float,
        previous_body_ball_contact: bool,
    ) -> ContactHandoffDecision:
        if self._faulted:
            raise ValueError("faulted SIM handoff sequence; new episode object required")
        try:
            if (
                type(frame) is not int
                or frame != self._next_frame
                or not 0 <= frame < self._config.horizon_frames
                or any(
                    type(value) not in (float, int) or not math.isfinite(value)
                    for value in (gap_x_m, gap_y_m, pelvis_z_m, upright_tilt_rad)
                )
                or upright_tilt_rad < 0
                or upright_tilt_rad > math.pi
                or type(previous_body_ball_contact) is not bool
                or (frame == 0 and previous_body_ball_contact)
            ):
                raise ValueError(
                    "ordered finite measured state and previous-interval contact required"
                )
            transitioned = False
            if self._handoff_frame is None:
                self._contact_excluded |= previous_body_ball_contact
                admitted = (
                    not self._contact_excluded
                    and self._config.earliest_frame <= frame <= self._config.latest_frame
                    and 0 < gap_x_m <= self._config.entry_gap_x_m
                    and abs(gap_y_m) <= self._config.maximum_absolute_gap_y_m
                    and pelvis_z_m >= self._config.minimum_pelvis_z_m
                    and upright_tilt_rad <= self._config.maximum_upright_tilt_rad
                )
                if admitted:
                    self._handoff_frame = frame
                    transitioned = True
            contact = self._handoff_frame is not None
            reference_frame = (
                min(frame - self._handoff_frame, self._config.reference_frames - 1)
                if self._handoff_frame is not None
                else 0
            )
            decision = ContactHandoffDecision(
                frame=frame,
                controller="contact" if contact else "approach",
                reference_frame=reference_frame,
                transitioned=transitioned,
                handoff_frame=self._handoff_frame,
                excluded_by_previous_body_contact=self._contact_excluded,
            )
        except Exception:
            self._faulted = True
            raise
        self._next_frame += 1
        return decision
