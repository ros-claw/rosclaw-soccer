"""Bounded, body-aware feedback navigation for a measured receive-to-shot task.

This SIM_ONLY candidate has no motor, ball-state, or simulator authority. Its
five coefficients are eligible for physical curriculum search; performance
must be checked on fresh multi-agent physics before any stronger claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamStrikeFeedbackNavigation:
    agent_id: str
    foundation_hash: str
    foundation_config_hash: str
    position_gain: float
    prediction_horizon_sec: float
    stance_depth_m: float
    stance_lateral_m: float
    body_velocity_damping: float
    contract_hash: str = field(init=False)
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    history: list[tuple[int, float, float, float, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        values = (
            self.position_gain,
            self.prediction_horizon_sec,
            self.stance_depth_m,
            self.stance_lateral_m,
            self.body_velocity_damping,
        )
        if (
            any(type(value) is not float or not math.isfinite(value) for value in values)
            or not 0.0 <= self.position_gain <= 1.5
            or not 0.0 <= self.prediction_horizon_sec <= 1.0
            or not 0.30 <= self.stance_depth_m <= 0.80
            or not -0.35 <= self.stance_lateral_m <= 0.35
            or not 0.0 <= self.body_velocity_damping <= 0.6
        ):
            raise ValueError("finite bounded strike-navigation coefficients required")
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.rsi.team_strike_feedback_navigation.v1",
                    "agent_id": self.agent_id,
                    "foundation_hash": self.foundation_hash,
                    "foundation_config_hash": self.foundation_config_hash,
                    "position_gain": self.position_gain,
                    "prediction_horizon_sec": self.prediction_horizon_sec,
                    "stance_depth_m": self.stance_depth_m,
                    "stance_lateral_m": self.stance_lateral_m,
                    "body_velocity_damping": self.body_velocity_damping,
                    "activation_ceiling": self.activation_ceiling,
                }
            )
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        if observation.agent_id != self.agent_id:
            raise ValueError("foreign navigation observation")
        if observation.intent != "shoot":
            return None
        body = observation.body_pose
        ball = observation.ball_position
        if math.hypot(ball[0] - body[0], ball[1] - body[1]) > 2.5:
            return None
        predicted_x = ball[0] + self.prediction_horizon_sec * observation.ball_velocity[0]
        predicted_y = ball[1] + self.prediction_horizon_sec * observation.ball_velocity[1]
        direction_x = observation.task_target[0] - predicted_x
        direction_y = observation.task_target[1] - predicted_y
        distance = math.hypot(direction_x, direction_y)
        if distance < 0.1:
            return None
        direction_x /= distance
        direction_y /= distance
        target_x = (
            predicted_x - self.stance_depth_m * direction_x - self.stance_lateral_m * direction_y
        )
        target_y = (
            predicted_y - self.stance_depth_m * direction_y + self.stance_lateral_m * direction_x
        )
        dx = self.position_gain * (target_x - body[0])
        dy = self.position_gain * (target_y - body[1])
        dx -= self.body_velocity_damping * observation.body_velocity[0]
        dy -= self.body_velocity_damping * observation.body_velocity[1]
        norm = math.hypot(dx, dy)
        if norm > 0.25:
            dx, dy = dx * 0.25 / norm, dy * 0.25 / norm
        self.history.append((observation.frame, target_x, target_y, dx, dy))
        return NavigationDelta(
            observation.agent_id,
            observation.frame,
            observation.time_sec,
            (dx, dy, 0.0),
        )
