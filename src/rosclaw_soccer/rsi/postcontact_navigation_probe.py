"""Explicit SIM intervention for follow-through reachability, not a policy.

The reference tracks measured ball motion while seeking a longitudinal gap.
Only forward navigation changes; original lateral/yaw and motor limits remain.
Reference slew bounds are not guarantees about physical root acceleration.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True, slots=True)
class PostcontactNavigationProbe:
    target_gap_m: float = 0.55
    separation_gain_s_inv: float = 1.0
    reference_slew_m_s2: float = 1.5

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("target_gap_m", 0.35, 0.8),
            ("separation_gain_s_inv", 0.3, 2.0),
            ("reference_slew_m_s2", 0.5, 3.0),
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("finite bounded postcontact SIM parameters required")

    def navigation(
        self, *, previous_forward: float, gap_x: float, ball_vx: float, forward_cap: float
    ) -> tuple[float, float]:
        self.__post_init__()
        if (
            any(
                type(v) is not float or not math.isfinite(v) or abs(v) > 100.0
                for v in (previous_forward, gap_x, ball_vx, forward_cap)
            )
            or not 0.8 <= forward_cap <= 1.48
            or not 0.0 <= previous_forward <= forward_cap
        ):
            raise ValueError("measured SIM state and previous bounded forward reference required")
        raw = ball_vx + self.separation_gain_s_inv * (gap_x - self.target_gap_m)
        desired = min(forward_cap, max(0.0, raw))
        step = self.reference_slew_m_s2 * 0.02
        actual = min(previous_forward + step, max(previous_forward - step, desired))
        return raw, actual

    def contract(self) -> dict[str, Any]:
        self.__post_init__()
        value = dict(
            schema="soccer.rsi.postcontact_navigation_probe.v1",
            parameters=asdict(self),
            control_dt_s=0.02,
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            forward_only=True,
            current_measured_state_only=True,
            future_state_read=False,
            reference_slew_is_not_physical_acceleration=True,
            motor_joint_and_torque_limits_unchanged=True,
            activation_ceiling="SIM_ONLY",
            trained_policy=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))
