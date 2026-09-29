"""SIM_ONLY measured-state mixture of retained and adaptive receive experts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.rsi.team_receive_body_tap import TeamReceiveBodyTap
from rosclaw_soccer.rsi.team_receive_phase_actor import TeamReceivePhaseActor
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamReceiveGatedMixtureActor(TeamReceivePhaseActor):
    alternative_weights: tuple[float, float, float, float, float] = (0.0,) * 5
    gate_weights: tuple[float, float, float, float] = (0.0,) * 4
    alternative_active_frames: int = field(init=False, default=0)
    maximum_gate: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.alternative_weights) is not tuple
            or len(self.alternative_weights) != 5
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 2.0
                for value in self.alternative_weights
            )
            or type(self.gate_weights) is not tuple
            or len(self.gate_weights) != 4
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 12.0
                for value in self.gate_weights
            )
        ):
            raise ValueError("bounded same-body receive mixture required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.team_receive_gated_mixture_actor.v1",
                "retained_actor_contract_hash": self.contract_hash,
                "alternative_weights": self.alternative_weights,
                "gate_weights": self.gate_weights,
                "action_cap_mps": 0.25,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        TeamReceiveBodyTap.propose(self, observation)
        sign = 1.0 if self.agent_id.startswith("red.") else -1.0
        x_gap = sign * (observation.body_pose[0] - observation.ball_position[0])
        if (
            not observation.committed_receiver
            or observation.intent not in {"receive", "intercept", "run_in_behind"}
            or sign * observation.ball_velocity[0] <= 0.20
            or not 0.15 <= x_gap <= 1.2
        ):
            self._smoothed = (0.0, 0.0)
            return None
        feet = observation.effector_positions
        side = min(
            range(2),
            key=lambda index: math.hypot(
                feet[index][1] - observation.ball_position[0],
                feet[index][2] - observation.ball_position[1],
            ),
        )
        foot = feet[side]
        velocity = observation.effector_velocities[side]
        knee_velocity = observation.joint_velocities_radps[side * 6 + 3]
        rel_y = observation.ball_position[1] - foot[2]
        rel_vy = observation.ball_velocity[1] - velocity[2]
        body_vy = observation.body_velocity[1]
        gate_features = (
            1.0,
            10.0 * (0.30 - abs(observation.ball_velocity[1])),
            rel_y,
            body_vy,
        )
        logit = sum(
            weight * feature
            for weight, feature in zip(self.gate_weights, gate_features, strict=True)
        )
        gate = 0.0 if logit <= -8.0 else 1.0 if logit >= 8.0 else 1.0 / (1.0 + math.exp(-logit))
        self.maximum_gate = max(self.maximum_gate, gate)
        if gate > 0.01:
            self.alternative_active_frames += 1
        w = tuple(
            retained + gate * (alternative - retained)
            for retained, alternative in zip(self.weights, self.alternative_weights, strict=True)
        )
        raw_x = sign * 0.25 * math.tanh(w[0] * knee_velocity / 3.0 + w[1] * (x_gap - 0.15))
        raw_y = 0.25 * math.tanh(w[2] * rel_y + w[3] * rel_vy + w[4] * body_vy)
        dx = 0.75 * self._smoothed[0] + 0.25 * raw_x
        dy = 0.75 * self._smoothed[1] + 0.25 * raw_y
        speed = math.hypot(dx, dy)
        if speed > 0.25:
            dx *= 0.25 / speed
            dy *= 0.25 / speed
        self._smoothed = (dx, dy)
        if abs(dx) + abs(dy) <= 1e-8:
            return None
        self.active_frames += 1
        self.peak_delta_mps = max(self.peak_delta_mps, math.hypot(dx, dy))
        return NavigationDelta(
            observation.agent_id, observation.frame, observation.time_sec, (dx, dy, 0.0)
        )
