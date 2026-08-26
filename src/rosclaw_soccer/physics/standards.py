"""Metric football dimensions bound to the current IFAB Laws of the Game."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

IFAB_FIELD_SOURCE = "https://www.theifab.com/laws/latest/the-field-of-play/"
IFAB_BALL_SOURCE = "https://www.theifab.com/laws/latest/the-ball/"


@dataclass(frozen=True)
class IFABRegulationSpec:
    """One explicit adult international-match geometry and ball selection.

    IFAB specifies ranges for field and ball measurements. The selected values
    are inside those ranges; they are not learned parameters. Net depth and
    contact coefficients are versioned simulator engineering parameters
    because the Laws do not prescribe them.
    """

    field_length_m: float = 105.0
    field_width_m: float = 68.0
    line_width_m: float = 0.10
    goal_inside_width_m: float = 7.32
    goal_inside_height_m: float = 2.44
    goal_frame_diameter_m: float = 0.10
    goal_area_depth_m: float = 5.50
    penalty_area_depth_m: float = 16.50
    penalty_mark_distance_m: float = 11.0
    corner_arc_radius_m: float = 1.0
    ball_circumference_m: float = 0.69
    ball_mass_kg: float = 0.43
    ball_pressure_atm: float = 0.80
    net_depth_m: float = 2.0
    ball_sliding_friction: float = 0.10
    ball_torsional_friction: float = 0.005
    ball_rolling_friction: float = 0.00010
    ball_linear_damping_n_s_m: float = 0.004
    net_stiffness_n_m: float = 180.0
    net_damping_n_s_m: float = 10.0
    schema_version: str = "rosclaw_soccer.ifab_regulation.v1"

    def __post_init__(self) -> None:
        numeric = tuple(
            value
            for key, value in asdict(self).items()
            if key != "schema_version" and isinstance(value, float)
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("regulation measurements must be finite")
        if not 100.0 <= self.field_length_m <= 110.0:
            raise ValueError("international field length must be in [100, 110] m")
        if not 64.0 <= self.field_width_m <= 75.0:
            raise ValueError("international field width must be in [64, 75] m")
        if not 0.0 < self.line_width_m <= 0.12:
            raise ValueError("field line width must be in (0, 0.12] m")
        if self.goal_inside_width_m != 7.32 or self.goal_inside_height_m != 2.44:
            raise ValueError("adult regulation goal must be exactly 7.32 x 2.44 m inside")
        if not 0.0 < self.goal_frame_diameter_m <= 0.12:
            raise ValueError("goal frame width/depth must not exceed 0.12 m")
        if not 0.68 <= self.ball_circumference_m <= 0.70:
            raise ValueError("ball circumference must be in [0.68, 0.70] m")
        if not 0.410 <= self.ball_mass_kg <= 0.450:
            raise ValueError("ball mass must be in [0.410, 0.450] kg")
        if not 0.6 <= self.ball_pressure_atm <= 1.1:
            raise ValueError("ball pressure must be in [0.6, 1.1] atm")
        if not 1.0 <= self.net_depth_m <= 3.0:
            raise ValueError("simulator net depth must be in [1, 3] m")
        if not 0.03 <= self.ball_sliding_friction <= 0.80:
            raise ValueError("ball sliding friction must be in [0.03, 0.80]")
        if not 0.0 <= self.ball_rolling_friction <= 0.001:
            raise ValueError("ball rolling friction must be in [0, 0.001]")

    @property
    def ball_radius_m(self) -> float:
        return self.ball_circumference_m / (2.0 * math.pi)

    @property
    def ball_solid_sphere_inertia_kg_m2(self) -> float:
        return 0.4 * self.ball_mass_kg * self.ball_radius_m**2

    @property
    def goal_frame_radius_m(self) -> float:
        return self.goal_frame_diameter_m / 2.0

    @property
    def goal_area_inside_width_m(self) -> float:
        return self.goal_inside_width_m + 2.0 * self.goal_area_depth_m

    @property
    def penalty_area_inside_width_m(self) -> float:
        return self.goal_inside_width_m + 2.0 * self.penalty_area_depth_m

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "ball_radius_m": self.ball_radius_m,
                "ball_solid_sphere_inertia_kg_m2": (self.ball_solid_sphere_inertia_kg_m2),
                "goal_area_inside_width_m": self.goal_area_inside_width_m,
                "penalty_area_inside_width_m": self.penalty_area_inside_width_m,
                "normative_sources": [IFAB_FIELD_SOURCE, IFAB_BALL_SOURCE],
                "net_depth_is_normative": False,
                "contact_coefficients_are_normative": False,
            }
        )
        return value


__all__ = ["IFAB_BALL_SOURCE", "IFAB_FIELD_SOURCE", "IFABRegulationSpec"]
