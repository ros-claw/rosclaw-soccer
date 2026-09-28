"""Bounded SIM_ONLY interception policy family for physical search training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamAdaptiveInterceptNavigation:
    agent_id: str
    foundation_hash: str
    foundation_config_hash: str
    forward_gain: float
    lateral_gain: float
    target_gap_m: float
    activation_max_gap_m: float
    speed_cap_mps: float
    contract_hash: str = field(init=False)
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    history: list[tuple[int, float, float, float, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        values = (
            self.forward_gain,
            self.lateral_gain,
            self.target_gap_m,
            self.activation_max_gap_m,
            self.speed_cap_mps,
        )
        if (
            any(type(value) is not float or not math.isfinite(value) for value in values)
            or not -0.4 <= self.forward_gain <= 0.8
            or not -1.6 <= self.lateral_gain <= 1.6
            or not 0.3 <= self.target_gap_m <= 0.8
            or not 1.2 <= self.activation_max_gap_m <= 2.0
            or not 0.18 <= self.speed_cap_mps <= 0.25
        ):
            raise ValueError("finite bounded SIM_ONLY interception policy required")
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rsi_team_adaptive_intercept_navigation_v28",
                    "agent_id": self.agent_id,
                    "foundation_hash": self.foundation_hash,
                    "foundation_config_hash": self.foundation_config_hash,
                    "forward_gain": self.forward_gain,
                    "lateral_gain": self.lateral_gain,
                    "target_gap_m": self.target_gap_m,
                    "activation_max_gap_m": self.activation_max_gap_m,
                    "speed_cap_mps": self.speed_cap_mps,
                    "activation_ceiling": self.activation_ceiling,
                }
            )
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta:
        if observation.agent_id != self.agent_id:
            raise ValueError("foreign navigation observation")
        feet = dict((name, (x, y, z)) for name, x, y, z in observation.effector_positions)
        if set(feet) != {"left_foot", "right_foot"}:
            raise ValueError("both measured feet required")
        left, right = feet["left_foot"], feet["right_foot"]
        if left[2] - right[2] >= 0.02:
            foot = left
        elif right[2] - left[2] >= 0.02:
            foot = right
        else:
            foot = min(
                (left, right), key=lambda value: abs(observation.ball_position[1] - value[1])
            )
        x_gap = observation.ball_position[0] - foot[0]
        y_error = observation.ball_position[1] - foot[1]
        dx = dy = 0.0
        if observation.frame >= 30 and 0.15 <= x_gap <= self.activation_max_gap_m:
            dx = max(
                -self.speed_cap_mps,
                min(self.speed_cap_mps, self.forward_gain * (x_gap - self.target_gap_m)),
            )
            dy = max(-self.speed_cap_mps, min(self.speed_cap_mps, self.lateral_gain * y_error))
            norm = math.hypot(dx, dy)
            if norm > 0.25:
                dx, dy = dx * 0.25 / norm, dy * 0.25 / norm
        self.history.append((observation.frame, x_gap, y_error, dx, dy))
        return NavigationDelta(
            observation.agent_id, observation.frame, observation.time_sec, (dx, dy, 0.0)
        )
