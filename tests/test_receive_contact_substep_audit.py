import pytest

from rosclaw_soccer.rsi.receive_contact_substep_audit import ReceiveContactSubstepAudit
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.skills.team.motor_option import TeamBallContact, TeamMotorPhysicsObservation
from rosclaw_soccer.skills.team.physics_evidence import PhysicsEvidenceSlot


def observation(time_sec, contacts=()):
    return TeamMotorPhysicsObservation(
        time_sec=time_sec,
        qpos=tuple(float(i) for i in range(43)),
        qvel=tuple(float(i) for i in range(41)),
        world_bodies_safe=True,
        foot_normal_force_n=max((c.normal_force_n for c in contacts if c.is_foot), default=0.0),
        other_non_ground_normal_force_n=max(
            (c.normal_force_n for c in contacts if not c.is_foot), default=0.0
        ),
        observer_agent_id="red.finisher",
        contacts_complete=True,
        ball_contacts=contacts,
    )


def contact(relative=(0.1, 0.2, 0.3)):
    return TeamBallContact(
        5, "red.finisher", "left_foot", 3.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), relative
    )


def test_substep_observer_keeps_physics_only_and_mailbox_order():
    mailbox = ReceiveContactMailbox("red.finisher")
    audit = ReceiveContactSubstepAudit(ReceiveContactEvidence(mailbox))
    assert not hasattr(audit, "propose")
    audit.observe_physics(observation(0.002))
    audit.observe_physics(observation(0.004, (contact(),)))
    assert mailbox.snapshot.first_own_foot_time_sec == 0.004
    assert len(audit.samples) == 1
    assert audit.samples[0]["ball_position_m"] == (36.0, 37.0, 38.0)
    assert audit.samples[0]["ball_velocity_mps"] == (35.0, 36.0, 37.0)
    assert audit.samples[0]["contacts"][0]["normal_force_n"] == 3.0
    assert audit.activation_ceiling == "SIM_ONLY"


def test_missing_velocity_and_overflow_are_fail_closed():
    audit = ReceiveContactSubstepAudit(
        ReceiveContactEvidence(ReceiveContactMailbox("red.finisher")), max_samples=1
    )
    slot = PhysicsEvidenceSlot(audit)
    slot.observe_physics(observation(0.002, (contact(),)))
    slot.observe_physics(observation(0.004, (contact(),)))
    assert slot.faulted and len(audit.samples) == 1
    second = PhysicsEvidenceSlot(
        ReceiveContactSubstepAudit(ReceiveContactEvidence(ReceiveContactMailbox("red.finisher")))
    )
    second.observe_physics(observation(0.002, (contact(None),)))
    assert second.faulted
    with pytest.raises(ValueError):
        ReceiveContactSubstepAudit(
            ReceiveContactEvidence(ReceiveContactMailbox("red.finisher")), max_samples=0
        )
