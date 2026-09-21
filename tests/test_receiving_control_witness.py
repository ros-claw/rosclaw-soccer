from dataclasses import replace

import pytest

from rosclaw_soccer.training.receiving_control_witness import ReceivingControlWitness
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from tests.training.test_receiving_feedback import observation

FEET = ((0.0, 0.0, 0.03), (0.2, 0.0, 0.03))


def sample(frame, *, own=0.0, interruption=None):
    obs = observation(frame)
    q = list(obs.qpos)
    q[2] = 0.8
    q[36:39] = (0.2, 0.0, 0.115)
    return replace(
        obs,
        qpos=tuple(q),
        last_own_foot_contact_time_sec=own,
        last_own_contact_foot=None if own is None else 1,
        contact_history=ReceivingContactHistory(obs.time_sec, interruption),
    )


def observe(witness, obs, **kwargs):
    return witness.observe(obs, foot_positions=FEET, minimum_joint_margin_rad=0.1, **kwargs)


def test_half_second_streak_only_admits_probe_not_ready_or_motion():
    witness = ReceivingControlWitness("blue.finisher", start_frame=0)
    for i in range(25):
        assert not observe(witness, sample(i)).handoff_probe_admissible
    result = observe(witness, sample(25))
    assert result.handoff_probe_admissible and result.control_streak_sec == pytest.approx(0.5)
    assert result.successor_ready_verified is False


def test_new_interruption_resets_even_if_own_foot_recontacts_before_next_sample():
    witness = ReceivingControlWitness("blue.finisher", start_frame=0)
    for i in range(26):
        observe(witness, sample(i))
    result = observe(witness, sample(26, own=0.519, interruption=0.518))
    assert result.clean_own_contact
    assert result.control_streak_sec == 0 and not result.handoff_probe_admissible


def test_same_substep_tie_and_no_contact_do_not_establish_control():
    for own, interruption in ((None, None), (0.02, 0.02), (0.0, 0.02)):
        witness = ReceivingControlWitness("blue.finisher", start_frame=1)
        result = observe(witness, sample(1, own=own, interruption=interruption))
        assert not result.clean_own_contact and not result.handoff_probe_admissible


@pytest.mark.parametrize(
    "field,value", [("agent_id", "red.finisher"), ("next_frame", 2), ("minimum_control_sec", 0)]
)
def test_public_stream_binding_and_interval_are_read_only(field, value):
    witness = ReceivingControlWitness("blue.finisher", start_frame=0)
    with pytest.raises(AttributeError):
        setattr(witness, field, value)


@pytest.mark.parametrize("new_interruption", [None, 0.01])
def test_interruption_history_cannot_be_forgotten_or_regress(new_interruption):
    witness = ReceivingControlWitness("blue.finisher", start_frame=1)
    observe(witness, sample(1, own=0.02, interruption=0.02))
    with pytest.raises(ValueError, match="regress"):
        observe(witness, sample(2, own=0.04, interruption=new_interruption))
    assert witness.faulted


@pytest.mark.parametrize("new_contact", [None, 0.01])
def test_own_contact_history_cannot_be_forgotten_or_regress(new_contact):
    witness = ReceivingControlWitness("blue.finisher", start_frame=1)
    observe(witness, sample(1, own=0.02))
    with pytest.raises(ValueError, match="own-contact history cannot regress"):
        observe(witness, sample(2, own=new_contact))
    assert witness.faulted


@pytest.mark.parametrize("failure", ["speed", "distance", "height", "joint", "body"])
def test_measured_control_loss_resets_streak(failure):
    witness = ReceivingControlWitness("blue.finisher", start_frame=0)
    for i in range(26):
        observe(witness, sample(i))
    obs = sample(26)
    q = list(obs.qpos)
    v = list(obs.qvel)
    if failure == "speed":
        v[35] = 0.36
    if failure == "distance":
        q[36] = 1.0
    if failure == "height":
        q[38] = 0.21
    if failure == "body":
        q[2] = 0.54
    result = witness.observe(
        replace(obs, qpos=tuple(q), qvel=tuple(v)),
        foot_positions=FEET,
        minimum_joint_margin_rad=-0.01 if failure == "joint" else 0.1,
    )
    assert result.control_streak_sec == 0 and not result.handoff_probe_admissible
    assert not observe(witness, sample(27)).handoff_probe_admissible


@pytest.mark.parametrize("bad", ["gap", "foreign", "missing", "nonfinite"])
def test_invalid_stream_latches_instead_of_preserving_control(bad):
    witness = ReceivingControlWitness("blue.finisher", start_frame=0)
    obs = sample(1 if bad == "gap" else 0)
    if bad == "foreign":
        obs = replace(obs, agent_id="red.finisher")
    if bad == "missing":
        obs = replace(obs, contact_history=None)
    with pytest.raises(ValueError):
        witness.observe(
            obs,
            foot_positions=FEET,
            minimum_joint_margin_rad=float("nan") if bad == "nonfinite" else 0.1,
        )
    assert witness.faulted
    with pytest.raises(ValueError, match="latched"):
        observe(witness, sample(0))
