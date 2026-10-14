"""Bounded SIM teacher-search interface, not trained control or authority."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

BOUNDS = {
    "forward_cap_m_s": (0.8, 1.48),
    "closing_gain_s_inv": (0.3, 3.0),
    "stance_gap_m": (0.15, 0.65),
    "lateral_gain_s_inv": (0.0, 2.0),
    "lateral_finish_gap_m": (0.66, 1.4),
    "lateral_offset_m": (-0.08, 0.08),
    "foot_forward_cap_m": (0.02, 0.1),
    "foot_lateral_cap_m": (0.04, 0.15),
    "foot_vertical_offset_m": (0.0, 0.06),
}


@dataclass(frozen=True, slots=True)
class BoundedFirstTouchControl:
    forward_cap_m_s: float = 1.4
    closing_gain_s_inv: float = 1.2
    stance_gap_m: float = 0.35
    lateral_gain_s_inv: float = 1.2
    lateral_finish_gap_m: float = 0.95
    lateral_offset_m: float = 0.0
    foot_forward_cap_m: float = 0.08
    foot_lateral_cap_m: float = 0.15
    foot_vertical_offset_m: float = 0.04

    def __post_init__(self) -> None:
        for name, (lower, upper) in BOUNDS.items():
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("finite bounded float teacher parameters required")

    def navigation(
        self, *, gap_x: float, gap_y: float, ball_vx: float, before_contact: bool
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        self.__post_init__()
        if type(before_contact) is not bool or any(
            type(v) is not float or not math.isfinite(v) or abs(v) > 100
            for v in (gap_x, gap_y, ball_vx)
        ):
            raise ValueError("bounded current measured SIM state required")
        if before_contact:
            raw_forward = ball_vx + self.closing_gain_s_inv * (gap_x - self.stance_gap_m)
            acquisition_weight = float(
                np.clip(
                    (gap_x - self.stance_gap_m) / (self.lateral_finish_gap_m - self.stance_gap_m),
                    0.0,
                    1.0,
                )
            )
            raw_lateral = (
                self.lateral_gain_s_inv * (gap_y - self.lateral_offset_m) * acquisition_weight
            )
        else:
            raw_forward, raw_lateral = self.forward_cap_m_s, 0.0
        raw = (raw_forward, raw_lateral, 0.0)
        projected = (
            float(np.clip(raw_forward, 0.0, self.forward_cap_m_s)),
            float(np.clip(raw_lateral, -0.2, 0.2)),
            0.0,
        )
        return raw, projected

    def contract(self) -> dict[str, Any]:
        self.__post_init__()
        value = dict(
            schema="soccer.rsi.bounded_first_touch_control.v1",
            parameters=asdict(self),
            bounds={name: list(bounds) for name, bounds in BOUNDS.items()},
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            coordinate_frame="world",
            teacher_privileged_current_state=True,
            future_state_read=False,
            maximum_navigation_speed_m_s=1.5,
            motor_joint_and_torque_limits_unchanged=True,
            trained_policy=False,
            activation_ceiling="SIM_ONLY",
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))
