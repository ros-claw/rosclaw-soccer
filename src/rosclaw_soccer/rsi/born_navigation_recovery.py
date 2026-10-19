"""Typed post-contact navigation lease born at a completed SIM boundary.

Not a torque intervention, policy promotion, learned actor or live interface.
Only lower the existing forward request; lateral/yaw, caps and timing remain
unchanged. A separate full-task physical comparison must qualify each option.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

FORWARD_CAPS = {"nav_brake_0_7": 0.7, "nav_brake_0": 0.0}


@dataclass(frozen=True, slots=True)
class BornNavigationRecovery:
    start_frame: int
    action_name: str

    def __post_init__(self) -> None:
        if (
            type(self.start_frame) is not int
            or not 31 <= self.start_frame < 300
            or type(self.action_name) is not str
            or self.action_name not in FORWARD_CAPS
        ):
            raise ValueError("typed current post-contact navigation option required")

    def apply(
        self,
        requested: tuple[float, float, float],
        projected: tuple[float, float, float],
        *,
        frame: int,
        after_measured_contact: bool,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        self.__post_init__()
        if (
            type(frame) is not int
            or not self.start_frame <= frame < 300
            or after_measured_contact is not True
            or any(type(a) is not tuple or len(a) != 3 for a in (requested, projected))
            or any(
                type(v) is not float or not np.isfinite(v)
                for a in (requested, projected)
                for v in a
            )
            or not 0.0 <= projected[0] <= 1.4
            or abs(projected[1]) > 0.2
            or requested[2] != 0.0
            or projected[2] != 0.0
        ):
            raise ValueError("finite original post-contact command and active H300 lease required")
        cap = FORWARD_CAPS[self.action_name]
        return (min(requested[0], cap), *requested[1:]), (min(projected[0], cap), *projected[1:])

    def contract(self) -> dict[str, Any]:
        self.__post_init__()
        value = dict(
            schema="soccer.rsi.born_navigation_recovery.v1",
            action_name=self.action_name,
            start_frame=self.start_frame,
            end_exclusive=300,
            forward_cap_m_s=FORWARD_CAPS[self.action_name],
            changes_lateral_yaw=False,
            changes_motor_or_torque_caps=False,
            future_state_used=False,
            activation_ceiling="SIM_ONLY",
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))
