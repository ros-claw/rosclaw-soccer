"""One SIM_ONLY navigation authority selecting a side from measured ball pose."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.rsi.team_receive_contact_phase_actor import TeamReceiveContactPhaseActor
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation


@dataclass
class TeamReceiveSideNavigation:
    agent_id: str
    foundation_hash: str
    foundation_config_hash: str
    mailbox: ReceiveContactMailbox
    left_post_weights: tuple[float, float, float, float, float]
    right_post_weights: tuple[float, float, float, float, float]
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    needs_effector_velocities: bool = field(init=False, default=True)
    needs_full_proprioception: bool = field(init=False, default=True)
    needs_continuous_body_context: bool = field(init=False, default=True)
    contract_hash: str = field(init=False)
    _children: tuple[TeamReceiveContactPhaseActor, TeamReceiveContactPhaseActor] = field(init=False)
    _selected_side: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mailbox, ReceiveContactMailbox)
            or self.mailbox.agent_id != self.agent_id
            or any(
                type(weights) is not tuple
                or len(weights) != 5
                or any(type(v) is not float or not math.isfinite(v) or abs(v) > 2 for v in weights)
                for weights in (self.left_post_weights, self.right_post_weights)
            )
        ):
            raise ValueError("bounded same-player side-conditioned navigation required")
        children = tuple(
            TeamReceiveContactPhaseActor(
                self.agent_id,
                self.foundation_hash,
                self.foundation_config_hash,
                weights=(0.0,) * 5,
                post_weights=weights,
                mailbox=self.mailbox,
            )
            for weights in (self.left_post_weights, self.right_post_weights)
        )
        self._children = (children[0], children[1])
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.team_receive_side_navigation.v1",
                "agent_id": self.agent_id,
                "mailbox_contract_hash": self.mailbox.contract_hash,
                "children": [child.contract_hash for child in self._children],
                "selection": "frame_zero_measured_ball_lateral_relative_to_pelvis",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        observation.__post_init__()
        if observation.agent_id != self.agent_id:
            raise ValueError("same-player navigation observation required")
        if self._selected_side is None:
            if observation.frame != 0:
                raise ValueError("side-conditioned navigation requires frame-zero selection")
            self._selected_side = int(observation.ball_position[1] < observation.body_pose[1])
        return self._children[self._selected_side].propose(observation)

    @property
    def selected_side(self) -> int | None:
        return self._selected_side

    @property
    def post_active_frames(self) -> int:
        if self._selected_side is None:
            return 0
        return self._children[self._selected_side].post_active_frames

    @property
    def peak_delta_mps(self) -> float:
        if self._selected_side is None:
            return 0.0
        return self._children[self._selected_side].peak_delta_mps
