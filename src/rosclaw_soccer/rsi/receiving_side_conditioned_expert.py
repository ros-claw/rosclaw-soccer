"""One SIM_ONLY A2 authority selecting a left or right receiving expert.

Each child receives the same immutable observation and completed contact
mailbox. Only one child is ever active during a rollout, and the world retains
all motor, joint-range, and hard-torque guards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingSideConditionedExpert:
    agent_id: str
    schedule_hash: str
    mailbox: ReceiveContactMailbox
    coordination: tuple[float, ...]
    left_weights: tuple[float, ...]
    right_weights: tuple[float, ...]
    left_post_gain: float = 0.0
    right_post_gain: float = 0.4
    left_velocity_horizon_sec: float = 0.0
    right_velocity_horizon_sec: float = 0.0
    left_target_depth_m: float = 0.25
    left_target_lateral_m: float = 0.12
    right_target_depth_m: float = 0.25
    right_target_lateral_m: float = 0.12
    action_substrate: str = field(init=False, default="A2_body29_precontact")
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    requires_contact_history: bool = field(init=False, default=True)
    requires_foot_kinematics: bool = field(init=False, default=True)
    contract_hash: str = field(init=False)
    _children: tuple[ReceivingPrecontactExpert, ReceivingPrecontactExpert] = field(init=False)
    _selected_side: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        settings = (
            self.left_post_gain,
            self.right_post_gain,
            self.left_velocity_horizon_sec,
            self.right_velocity_horizon_sec,
            self.left_target_depth_m,
            self.left_target_lateral_m,
            self.right_target_depth_m,
            self.right_target_lateral_m,
        )
        if (
            not isinstance(self.mailbox, ReceiveContactMailbox)
            or self.mailbox.agent_id != self.agent_id
            or any(type(v) is not float or not math.isfinite(v) for v in settings)
            or not 0 <= self.left_post_gain <= 1
            or not 0 <= self.right_post_gain <= 1
            or not 0 <= self.left_velocity_horizon_sec <= 0.15
            or not 0 <= self.right_velocity_horizon_sec <= 0.15
        ):
            raise ValueError("finite same-player SIM_ONLY side-conditioned expert required")
        children = []
        for gain, horizon, depth, lateral in (
            (
                self.left_post_gain,
                self.left_velocity_horizon_sec,
                self.left_target_depth_m,
                self.left_target_lateral_m,
            ),
            (
                self.right_post_gain,
                self.right_velocity_horizon_sec,
                self.right_target_depth_m,
                self.right_target_lateral_m,
            ),
        ):
            children.append(
                ReceivingPrecontactExpert(
                    self.agent_id,
                    self.schedule_hash,
                    self.mailbox,
                    0.35,
                    gain,
                    horizon,
                    target_depth_m=depth,
                    target_lateral_m=lateral,
                    coordination=self.coordination,
                    left_weights=self.left_weights,
                    right_weights=self.right_weights,
                )
            )
        self._children = (children[0], children[1])
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_side_conditioned_expert.v1",
                "agent_id": self.agent_id,
                "schedule_hash": self.schedule_hash,
                "mailbox_hash": self.mailbox.contract_hash,
                "children": [child.contract_hash for child in self._children],
                "selection": "initial_measured_ball_lateral_relative_to_pelvis",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        observation.__post_init__()
        if (
            observation.agent_id != self.agent_id
            or observation.action_substrate != self.action_substrate
        ):
            raise ValueError("same-player A2 receiving observation required")
        if self._selected_side is None:
            self._selected_side = int(observation.qpos[37] < observation.qpos[1])
        return self._children[self._selected_side].propose(observation)
