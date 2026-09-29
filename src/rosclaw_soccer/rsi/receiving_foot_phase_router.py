"""SIM_ONLY foot-relative teacher tracking inside the existing guarded A2 action."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import ReceivingKinematicTemporalExpert
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingMeasuredSkillRouter
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass(frozen=True)
class ReceivingFootPhaseReference:
    expert: str
    initial_lateral_m: float
    initial_closing_speed_mps: float
    first_foot_frame: int
    features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=np.float64)
        if (
            self.expert not in ("low", "high", "parent")
            or not math.isfinite(self.initial_lateral_m)
            or not 0.10 <= self.initial_lateral_m <= 0.18
            or not math.isfinite(self.initial_closing_speed_mps)
            or not -2.0 <= self.initial_closing_speed_mps <= -0.4
            or type(self.first_foot_frame) is not int
            or not 20 <= self.first_foot_frame <= 50
            or values.shape != (50, 48)
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 3.001
        ):
            raise ValueError("verified bounded 50-frame foot teacher reference required")


@dataclass
class ReceivingFootPhaseRouter(ReceivingMeasuredSkillRouter):
    """Track a genuine same-expert foot-relative trajectory, not a course label.

    All corrections are computed from immutable same-frame proprioception and
    returned to the ordinary A2 cursor, which retains the joint safety limits.
    """

    references: tuple[ReceivingFootPhaseReference, ...] = ()
    foot_gain: float = 0.0
    post_multiplier: float = 1.0
    shin_guard_m: float = 0.0
    velocity_horizon_sec: float = 0.02
    peak_correction_rad: float = field(init=False, default=0.0)
    nonzero_frames: int = field(init=False, default=0)
    selected_reference: int | None = field(init=False, default=None)
    first_contact_frame: int | None = field(init=False, default=None)
    requires_shin_clearance: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.references) is not tuple
            or not 1 <= len(self.references) <= 12
            or any(not isinstance(item, ReceivingFootPhaseReference) for item in self.references)
            or type(self.foot_gain) is not float
            or not math.isfinite(self.foot_gain)
            or not 0 <= self.foot_gain <= 1
            or type(self.post_multiplier) is not float
            or not math.isfinite(self.post_multiplier)
            or not 0 <= self.post_multiplier <= 2
            or type(self.shin_guard_m) is not float
            or not math.isfinite(self.shin_guard_m)
            or not 0 <= self.shin_guard_m <= 0.08
            or type(self.velocity_horizon_sec) is not float
            or not math.isfinite(self.velocity_horizon_sec)
            or not 0 <= self.velocity_horizon_sec <= 0.06
        ):
            raise ValueError("bounded measured foot-phase router required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_foot_phase_router.v1",
                "parent_contract_hash": self.contract_hash,
                "references": [
                    {
                        "expert": item.expert,
                        "initial_lateral_m": item.initial_lateral_m,
                        "initial_closing_speed_mps": item.initial_closing_speed_mps,
                        "first_foot_frame": item.first_foot_frame,
                        "features_hash": hash_json(item.features),
                    }
                    for item in self.references
                ],
                "foot_gain": self.foot_gain,
                "post_multiplier": self.post_multiplier,
                "shin_guard_m": self.shin_guard_m,
                "velocity_horizon_sec": self.velocity_horizon_sec,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = super().propose(observation)
        # Parent fallback and old anchor courses remain byte-for-byte unchanged.
        if self.selected_expert == "parent" or observation.frame >= 45:
            return base
        feet = observation.foot_kinematics
        if feet is None or feet.foot_linear_velocity_world_mps is None:
            raise ValueError("same-frame measured foot velocity and Jacobian required")
        measured = np.asarray(ReceivingKinematicTemporalExpert.features(observation))
        if self.selected_reference is None:
            choices = [
                (index, item)
                for index, item in enumerate(self.references)
                if item.expert == self.selected_expert
            ]
            if not choices:
                raise ValueError("routed specialist has no sealed foot reference")
            self.selected_reference = min(
                choices,
                key=lambda pair: math.hypot(
                    (measured[1] * 0.2 - pair[1].initial_lateral_m) / 0.006,
                    (measured[3] * 2.0 - pair[1].initial_closing_speed_mps) / 0.12,
                ),
            )[0]
        if self.foot_gain == 0 or observation.frame < 19:
            return base
        shin = observation.shin_clearance
        if shin is None:
            raise ValueError("same-frame shin clearance required for bounded feedback")
        reference = self.references[self.selected_reference]
        recorded = np.asarray(reference.features)
        if observation.last_own_foot_contact_time_sec is not None:
            if self.first_contact_frame is None:
                self.first_contact_frame = observation.frame
            index = int(
                np.clip(
                    reference.first_foot_frame + observation.frame - self.first_contact_frame - 15,
                    0,
                    49,
                )
            )
        else:
            # Match the observable approach phase, within three 50-Hz frames.
            lower = max(0, observation.frame - 15 - 3)
            upper = min(50, observation.frame - 15 + 4)
            candidate = np.arange(lower, upper)
            index = int(candidate[np.argmin(np.abs(recorded[candidate, 0] - measured[0]))])
        if (
            self.shin_guard_m > 0
            and self.first_contact_frame is not None
            and min(shin.clearance_m) < self.shin_guard_m
        ):
            return base
        side = int(self._selected_side or 0)
        position_error = np.clip(
            recorded[index, 10 + side * 3 : 13 + side * 3] * 0.5
            - measured[10 + side * 3 : 13 + side * 3] * 0.5,
            -0.035,
            0.035,
        )
        velocity_error = np.clip(
            (
                recorded[index, 16 + side * 3 : 19 + side * 3]
                - measured[16 + side * 3 : 19 + side * 3]
            )
            * 2.0,
            -0.5,
            0.5,
        )
        task_delta = position_error + self.velocity_horizon_sec * velocity_error
        jacobian = np.asarray(feet.foot_linear_jacobian_world[side], dtype=np.float64)
        correction = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.02 * np.eye(3), task_delta
        )
        phase_gain = self.foot_gain * (
            self.post_multiplier if self.first_contact_frame is not None else 1.0
        )
        correction = np.clip(phase_gain * correction, -0.04, 0.04)
        target = np.asarray(base, dtype=np.float64)
        offset = side * 6
        target[offset : offset + 6] = np.clip(target[offset : offset + 6] + correction, -0.35, 0.35)
        actual = target[offset : offset + 6] - np.asarray(base)[offset : offset + 6]
        peak = float(np.max(np.abs(actual)))
        if peak > 0:
            self.nonzero_frames += 1
            self.peak_correction_rad = max(self.peak_correction_rad, peak)
        return tuple(float(value) for value in target)
