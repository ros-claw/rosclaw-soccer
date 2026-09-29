"""Causal two-stage SIM_ONLY receive navigation from attributed real foot contact."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.rsi.team_receive_phase_actor import TeamReceivePhaseActor
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamReceiveContactPhaseActor(TeamReceivePhaseActor):
    post_weights: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    mailbox: ReceiveContactMailbox | None = None
    post_active_frames: int = field(init=False, default=0)
    _post_smoothed: tuple[float, float] = field(init=False, default=(0.0, 0.0))

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.mailbox, ReceiveContactMailbox)
            or self.mailbox.agent_id != self.agent_id
            or type(self.post_weights) is not tuple
            or len(self.post_weights) != 5
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 2.0
                for value in self.post_weights
            )
        ):
            raise ValueError("finite bounded post-contact receive weights required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.team_receive_contact_phase_actor.v1",
                "precontact_actor_contract_hash": self.contract_hash,
                "mailbox_contract_hash": self.mailbox.contract_hash,
                "post_weights": self.post_weights,
                "contact_source": "attributed_500hz_own_foot",
                "post_contact_horizon_sec": 0.6,
                "action_cap_mps": 0.25,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        mailbox = self.mailbox
        if mailbox is None:
            raise ValueError("bound contact mailbox required")
        snapshot = mailbox.snapshot
        if abs(observation.time_sec - snapshot.physics_time_sec) > 0.002001:
            raise ValueError("navigation requires completed previous physics")
        if snapshot.first_own_foot_time_sec is None or not any(self.post_weights):
            return super().propose(observation)
        baseline = super().propose(observation)
        elapsed = observation.time_sec - snapshot.first_own_foot_time_sec
        if not 0.0 <= elapsed <= 0.6:
            self._post_smoothed = (0.0, 0.0)
            return baseline
        sign = 1.0 if self.agent_id.startswith("red.") else -1.0
        ball = observation.ball_position
        foot_side = 0 if snapshot.first_own_foot == "left_foot" else 1
        foot = observation.effector_positions[foot_side]
        if math.hypot(ball[0] - foot[1], ball[1] - foot[2]) > 0.80:
            return baseline
        w = self.post_weights
        relative_x = sign * (ball[0] - foot[1])
        relative_y = ball[1] - foot[2]
        raw_x = (
            sign * 0.25 * math.tanh(w[0] * relative_x + w[2] * sign * observation.ball_velocity[0])
        )
        raw_y = 0.25 * math.tanh(
            w[1] * relative_y
            + w[3] * observation.ball_velocity[1]
            + w[4] * observation.body_velocity[1]
        )
        post_x = 0.75 * self._post_smoothed[0] + 0.25 * raw_x
        post_y = 0.75 * self._post_smoothed[1] + 0.25 * raw_y
        self._post_smoothed = (post_x, post_y)
        if abs(post_x) + abs(post_y) <= 1e-8:
            return baseline
        base_x, base_y = (0.0, 0.0) if baseline is None else baseline.velocity_delta[:2]
        dx, dy = base_x + post_x, base_y + post_y
        speed = math.hypot(dx, dy)
        if speed > 0.25:
            dx *= 0.25 / speed
            dy *= 0.25 / speed
        self.post_active_frames += 1
        self.peak_delta_mps = max(self.peak_delta_mps, math.hypot(dx, dy))
        return NavigationDelta(
            observation.agent_id,
            observation.frame,
            observation.time_sec,
            (dx, dy, 0.0),
        )
