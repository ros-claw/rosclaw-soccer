"""State-conditioned, bounded G1 receiving stance learned from consumed courses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamContextPhaseNavigation:
    agent_id: str
    foundation_hash: str
    foundation_config_hash: str
    contract_hash: str = field(init=False)
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    entry_foot_gap_m: float | None = field(init=False, default=None)
    selected_lateral_gain: float | None = field(init=False, default=None)
    history: list[tuple[int, float, float, float, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rsi_team_context_phase_navigation_v17",
                    "agent_id": self.agent_id,
                    "foundation_hash": self.foundation_hash,
                    "foundation_config_hash": self.foundation_config_hash,
                    "entry_frame": 30,
                    "entry_foot_gap_threshold_m": 1.4,
                    "close_lateral_gain": 0.8,
                    "far_lateral_gain": 0.0,
                    "forward_gain": 0.4,
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
        if observation.frame == 30:
            self.entry_foot_gap_m = x_gap
            self.selected_lateral_gain = 0.8 if x_gap <= 1.4 else 0.0
        dx = dy = 0.0
        if observation.frame >= 30 and 0.2 <= x_gap <= 1.4:
            if self.selected_lateral_gain is None:
                raise ValueError("missing same-episode measured entry context")
            dx = max(-0.18, min(0.18, 0.4 * (x_gap - 0.55)))
            dy = max(-0.18, min(0.18, self.selected_lateral_gain * y_error))
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
