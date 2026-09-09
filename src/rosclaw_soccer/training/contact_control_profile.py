"""Bounded SIM-only control ablations shared by collection and examination."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


@dataclass(frozen=True)
class ContactControlProfile:
    separation_m: float = 0.85
    stiffness_scale: float = 0.8
    guard_margin_rad: float = 0.08
    strike_residual_enabled: bool = False
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(v)
                for v in (self.separation_m, self.stiffness_scale, self.guard_margin_rad)
            )
            or not 0.55 <= self.separation_m <= 1.2
            or not 0.4 <= self.stiffness_scale <= 1
            or not 0.04 <= self.guard_margin_rad <= 0.1
            or type(self.strike_residual_enabled) is not bool
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("contact control profile exceeds the simulation envelope")

    def apply(
        self, world: IndependentTeamWorldConfig, teacher: G1LocomotionContactTeacherConfig
    ) -> tuple[IndependentTeamWorldConfig, G1LocomotionContactTeacherConfig]:
        return (
            replace(
                world,
                minimum_player_separation_m=self.separation_m,
                joint_guard_margin_rad=self.guard_margin_rad,
                strike_residual_enabled=self.strike_residual_enabled,
            ),
            replace(teacher, contact_leg_stiffness_scale=self.stiffness_scale),
        )
