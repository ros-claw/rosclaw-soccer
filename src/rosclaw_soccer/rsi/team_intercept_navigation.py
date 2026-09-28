"""Bounded SIM_ONLY navigation adapter for learning a G1 receiving stance.

The frozen locomotion foundation still owns whole-body movement.  This policy
only proposes a small world-frame velocity delta through the existing guarded
shared-world navigation slot; it has no simulator or actuator handle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamInterceptNavigation:
    agent_id: str
    foundation_hash: str
    foundation_config_hash: str
    forward_gain: float
    lateral_gain: float
    contract_hash: str = field(init=False)
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    history: list[tuple[int, float, float, float, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if (
            type(self.forward_gain) is not float
            or type(self.lateral_gain) is not float
            or self.forward_gain not in (-0.4, 0.0, 0.4)
            or self.lateral_gain not in (-0.8, 0.0, 0.4, 0.8, 1.2)
        ):
            raise ValueError("frozen bounded intercept-navigation parameter family required")
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rsi_team_intercept_navigation_v15",
                    "agent_id": self.agent_id,
                    "foundation_hash": self.foundation_hash,
                    "foundation_config_hash": self.foundation_config_hash,
                    "forward_gain": self.forward_gain,
                    "lateral_gain": self.lateral_gain,
                    "activation_ceiling": self.activation_ceiling,
                }
            )
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta:
        if observation.agent_id != self.agent_id:
            raise ValueError("foreign navigation observation")
        feet = dict((name, (x, y, z)) for name, x, y, z in observation.effector_positions)
        if "left_foot" not in feet:
            raise ValueError("measured left foot required")
        foot = feet["left_foot"]
        x_gap = observation.ball_position[0] - foot[0]
        y_error = observation.ball_position[1] - foot[1]
        dx = dy = 0.0
        if observation.frame >= 30 and 0.2 <= x_gap <= 1.4:
            dx = max(-0.18, min(0.18, self.forward_gain * (x_gap - 0.55)))
            dy = max(-0.18, min(0.18, self.lateral_gain * y_error))
            norm = math.hypot(dx, dy)
            if norm > 0.25:
                dx, dy = dx * 0.25 / norm, dy * 0.25 / norm
        self.history.append((observation.frame, x_gap, y_error, dx, dy))
        return NavigationDelta(
            observation.agent_id,
            observation.frame,
            observation.time_sec,
            (dx, dy, 0.0),
        )
