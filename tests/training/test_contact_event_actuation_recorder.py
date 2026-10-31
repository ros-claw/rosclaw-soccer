from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.motor_option import (
    TeamActuatorEvidence,
    TeamBallContact,
    TeamMotorPhysicsObservation,
)
from rosclaw_soccer.skills.team.physics_evidence import PhysicsEvidenceSlot
from rosclaw_soccer.training.contact_event_actuation_recorder import ContactEventActuationRecorder


def observation(step: int, *, own_force: float = 0.0) -> TeamMotorPhysicsObservation:
    contacts = ()
    if own_force:
        contacts = (
            TeamBallContact(
                1,
                "red.finisher",
                "left_foot",
                own_force,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
    return TeamMotorPhysicsObservation(
        time_sec=step * 0.002,
        qpos=(0.0,) * 43,
        qvel=(0.0,) * 41,
        world_bodies_safe=True,
        foot_normal_force_n=own_force,
        other_non_ground_normal_force_n=0.0,
        observer_agent_id="red.finisher",
        contacts_complete=True,
        ball_contacts=contacts,
        actuator_evidence=TeamActuatorEvidence((0.0,) * 29, (0.0,) * 29, (0.0,) * 29),
    )


def test_overlapping_windows_deduplicate_and_keep_both_sides():
    recorder = ContactEventActuationRecorder("red.finisher", context_steps=2, maximum_substeps=20)
    slot = PhysicsEvidenceSlot(recorder)
    for step in range(1, 11):
        slot.observe_physics(observation(step, own_force=5.0 if step in (3, 5, 9) else 0.0))
    assert not slot.faulted
    recorder.verified_observations(control_frame_count=1, terminal_time_sec=0.02)
    data = recorder.to_dict()
    assert data["observed_steps"] == 10
    assert data["event_substeps"] == 3
    assert [r["step"] for r in data["rows"]] == list(range(1, 11))
    assert data["hardware_authorized"] is False


def test_no_contact_has_empty_windows_and_rejects_invalid_horizon():
    recorder = ContactEventActuationRecorder("red.finisher", context_steps=2, maximum_substeps=20)
    for step in range(1, 11):
        recorder.observe_physics(observation(step))
    assert recorder.to_dict()["rows"] == []
    for terminal in (float("nan"), 0.04):
        with pytest.raises(ValueError, match="authenticated"):
            recorder.verified_observations(control_frame_count=1, terminal_time_sec=terminal)


def test_requires_actuator_velocity_and_exact_clock():
    recorder = ContactEventActuationRecorder("red.finisher", maximum_substeps=20)
    with pytest.raises(ValueError, match="ordered"):
        recorder.observe_physics(replace(observation(1), actuator_evidence=None))
    with pytest.raises(ValueError, match="ordered"):
        recorder.observe_physics(observation(2))
    contact = observation(1, own_force=3.0)
    missing = replace(contact.ball_contacts[0], counterpart_minus_ball_velocity_world_mps=None)
    with pytest.raises(ValueError, match="ordered"):
        recorder.observe_physics(replace(contact, ball_contacts=(missing,)))
