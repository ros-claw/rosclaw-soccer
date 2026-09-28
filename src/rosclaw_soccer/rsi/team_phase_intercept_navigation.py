"""Phase-aware receiving stance on top of a frozen G1 locomotion foundation."""

from __future__ import annotations

import math

from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


class TeamPhaseInterceptNavigation(TeamInterceptNavigation):
    """Use the currently reachable swing-side foot, not a fixed left ankle."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rsi_team_phase_intercept_navigation_v16",
                    "agent_id": self.agent_id,
                    "foundation_hash": self.foundation_hash,
                    "foundation_config_hash": self.foundation_config_hash,
                    "forward_gain": self.forward_gain,
                    "lateral_gain": self.lateral_gain,
                    "foot_selection": "raised_by_0.02m_else_nearest_lateral",
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
            foot = min((left, right), key=lambda f: abs(observation.ball_position[1] - f[1]))
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
