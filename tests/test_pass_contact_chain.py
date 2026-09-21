from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.motor_option import TeamBallContact, TeamMotorPhysicsObservation
from rosclaw_soccer.training.pass_contact_chain import inspect_pass_contact_chain


def sample(t, contacts=()):
    return TeamMotorPhysicsObservation(
        t,
        (0.0,) * 43,
        (0.0,) * 41,
        True,
        max(
            (c.normal_force_n for c in contacts if c.agent_id == "blue.a" and c.is_foot), default=0
        ),
        max(
            (c.normal_force_n for c in contacts if not (c.agent_id == "blue.a" and c.is_foot)),
            default=0,
        ),
        "blue.a",
        True,
        contacts,
    )


def foot(agent):
    return TeamBallContact(1, agent, "left_foot", 1.0)


def chain():
    return (
        sample(0.002),
        sample(0.004, (foot("blue.a"),)),
        sample(0.006),
        sample(0.008, (foot("blue.b"),)),
    )


def inspect(rows):
    return inspect_pass_contact_chain(
        rows, sender_id="blue.a", receiver_id="blue.b", request_time_sec=0.002
    )


def test_clean_transfer_is_not_ready_or_training_authority():
    result = inspect(chain())
    assert result.clean_transfer_observed
    assert result.source_contact_sec == 0.004 and result.receiver_contact_sec == 0.008
    assert not result.successor_ready_verified and not result.training_authorized


@pytest.mark.parametrize(
    "contact",
    [
        foot("red.c"),
        TeamBallContact(2, "blue.a", "body", 1.0),
        TeamBallContact(3, None, "environment", 1.0),
    ],
)
def test_intervention_cannot_be_relabelled_as_sender_pass(contact):
    rows = chain()
    result = inspect((*rows[:2], sample(0.006, (contact,)), rows[3]))
    assert not result.clean_transfer_observed and result.interrupted_at_sec == 0.006


def test_no_departure_or_simultaneous_contact_is_not_pass():
    for contacts in ((foot("blue.b"),), (foot("blue.a"), foot("blue.b"))):
        result = inspect((*chain()[:2], sample(0.006, contacts), sample(0.008)))
        assert not result.clean_transfer_observed


def test_receiver_alone_and_zero_force_sender_do_not_prove_pass():
    rows = chain()
    for contact in ((), (replace(foot("blue.a"), normal_force_n=0.0),)):
        result = inspect((rows[0], sample(0.004, contact), *rows[2:]))
        assert "no_sender_foot_contact" in result.reasons


@pytest.mark.parametrize("index,time", [(2, 0.004), (2, 0.007), (0, 0.003)])
def test_incomplete_clock_rejected(index, time):
    rows = list(chain())
    rows[index] = replace(rows[index], time_sec=time)
    with pytest.raises(ValueError):
        inspect(tuple(rows))


def test_faulty_tail_and_unsafe_tail_cannot_hide_after_arrival():
    with pytest.raises(ValueError):
        inspect((*chain(), sample(0.012)))
    result = inspect((*chain(), replace(sample(0.010), world_bodies_safe=False)))
    assert not result.clean_transfer_observed and "unsafe_episode" in result.reasons
