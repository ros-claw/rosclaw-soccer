"""Causal, disjoint evidence-to-navigation interface for SIM_ONLY receiving."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_contact_phase_actor import TeamReceiveContactPhaseActor
from rosclaw_soccer.skills.team.motor_option import TeamBallContact, TeamMotorPhysicsObservation
from rosclaw_soccer.skills.team.physics_evidence import PhysicsEvidenceSlot


def _physics(time_sec: float, *contacts: TeamBallContact) -> TeamMotorPhysicsObservation:
    return TeamMotorPhysicsObservation(
        time_sec,
        (0.0,) * 43,
        (0.0,) * 41,
        True,
        max((c.normal_force_n for c in contacts if c.is_foot), default=0.0),
        max((c.normal_force_n for c in contacts if not c.is_foot), default=0.0),
        observer_agent_id="red.finisher",
        contacts_complete=True,
        ball_contacts=contacts,
    )


def test_contact_facts_are_attributed_immutable_and_causal() -> None:
    mailbox = ReceiveContactMailbox("red.finisher")
    evidence = ReceiveContactEvidence(mailbox)
    slot = PhysicsEvidenceSlot(evidence)
    assert not hasattr(evidence, "propose")
    nonfoot = TeamBallContact(24, "red.finisher", "body", 3.0)
    foot = TeamBallContact(25, "red.finisher", "right_foot", 5.0)
    slot.observe_physics(_physics(0.002, nonfoot))
    before = mailbox.snapshot
    slot.observe_physics(_physics(0.004, foot))
    assert before.prefoot_nonfoot_count == 1 and before.first_own_foot_time_sec is None
    assert mailbox.snapshot.first_own_foot_time_sec == 0.004
    assert mailbox.snapshot.first_own_foot == "right_foot"
    assert not slot.faulted
    with pytest.raises(AttributeError):
        mailbox.snapshot.physics_time_sec = 10.0
    slot.observe_physics(replace(_physics(0.006), observer_agent_id="blue.finisher"))
    assert slot.faulted and mailbox.snapshot.physics_time_sec == 0.004


def test_navigation_actor_cannot_be_registered_as_physics_consumer() -> None:
    mailbox = ReceiveContactMailbox("red.finisher")
    actor = TeamReceiveContactPhaseActor(
        "red.finisher",
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
        mailbox=mailbox,
    )
    assert not hasattr(actor, "observe_physics")
    assert not hasattr(actor, "needs_contact_velocity")
    with pytest.raises(ValueError):
        TeamReceiveContactPhaseActor(
            "blue.finisher",
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            mailbox=mailbox,
        )
    with pytest.raises(ValueError, match="completed previous physics"):
        from tests.test_navigation_option import observation

        actor.propose(replace(observation(2), agent_id="red.finisher"))
