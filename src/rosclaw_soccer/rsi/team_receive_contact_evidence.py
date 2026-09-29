"""One-way immutable contact facts from a private evidence consumer to navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


@dataclass(frozen=True)
class ReceiveContactSnapshot:
    physics_time_sec: float
    first_own_foot_time_sec: float | None
    first_own_foot: str | None
    first_contact_relative_y_mps: float | None
    prefoot_nonfoot_count: int


class ReceiveContactMailbox:
    """Facts only; no simulator, motor, permit or navigation proposal handle."""

    def __init__(self, agent_id: str) -> None:
        if not agent_id.startswith(("red.", "blue.")):
            raise ValueError("team-bound contact mailbox required")
        self.agent_id = agent_id
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receive_contact_mailbox.v1",
                "agent_id": agent_id,
                "activation_ceiling": "SIM_ONLY",
                "write_authority": "attributed_completed_physics_only",
            }
        )
        self._snapshot = ReceiveContactSnapshot(0.0, None, None, None, 0)

    @property
    def snapshot(self) -> ReceiveContactSnapshot:
        return self._snapshot

    def consume(self, observation: TeamMotorPhysicsObservation) -> None:
        previous = self._snapshot
        if (
            not isinstance(observation, TeamMotorPhysicsObservation)
            or observation.observer_agent_id != self.agent_id
            or not observation.contacts_complete
            or abs(observation.time_sec - previous.physics_time_sec - 0.002) > 1e-6
        ):
            raise ValueError("ordered attributed 500Hz contact facts required")
        observation.__post_init__()
        first_time = previous.first_own_foot_time_sec
        first_foot = previous.first_own_foot
        relative_y = previous.first_contact_relative_y_mps
        nonfoot_count = previous.prefoot_nonfoot_count
        if first_time is None:
            own = [
                contact
                for contact in observation.ball_contacts
                if contact.agent_id == self.agent_id
                and contact.is_foot
                and contact.normal_force_n > 1.0
            ]
            if own:
                contact = max(own, key=lambda item: item.normal_force_n)
                first_time, first_foot = observation.time_sec, contact.effector
                if contact.counterpart_minus_ball_velocity_world_mps is not None:
                    relative_y = contact.counterpart_minus_ball_velocity_world_mps[1]
            else:
                nonfoot_count += sum(
                    contact.agent_id == self.agent_id and not contact.is_foot
                    for contact in observation.ball_contacts
                )
        if relative_y is not None and not math.isfinite(relative_y):
            raise ValueError("finite measured contact velocity required")
        self._snapshot = ReceiveContactSnapshot(
            observation.time_sec, first_time, first_foot, relative_y, nonfoot_count
        )


class ReceiveContactEvidence:
    """Distinct registered evidence provider; its output has no action authority."""

    needs_contact_velocity = True

    def __init__(self, mailbox: ReceiveContactMailbox) -> None:
        if not isinstance(mailbox, ReceiveContactMailbox):
            raise ValueError("typed contact mailbox required")
        self.agent_id = mailbox.agent_id
        self.mailbox = mailbox
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receive_contact_evidence.v1",
                "mailbox_hash": mailbox.contract_hash,
                "agent_id": self.agent_id,
                "activation_ceiling": "SIM_ONLY",
                "needs_contact_velocity": True,
                "motor_authority": False,
            }
        )

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        self.mailbox.consume(observation)
