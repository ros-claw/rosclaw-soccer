"""Admission protocol tests do not certify contact or stability."""

from dataclasses import replace

import pytest

from rosclaw_soccer.providers.g1.contact_handoff import (
    ContactHandoffConfig,
    WholeBodyContactHandoff,
)


def advance(latch, index, **changes):
    values = dict(
        frame=index,
        gap_x_m=0.8,
        gap_y_m=0.0,
        pelvis_z_m=0.75,
        upright_tilt_rad=0.1,
        previous_body_ball_contact=False,
    )
    values.update(changes)
    return latch.advance(**values)


def prefix(latch, end):
    for frame in range(end):
        result = advance(latch, frame, gap_x_m=2.0)
        assert result.controller == "approach"
        assert result.reference_frame == 0


def test_single_latch_and_reference_clock_without_reset():
    latch = WholeBodyContactHandoff(ContactHandoffConfig())
    prefix(latch, 30)
    decision = advance(latch, 30)
    assert decision.transitioned and decision.reference_frame == 0
    for frame in range(31, 300):
        decision = advance(latch, frame, gap_x_m=-2.0, previous_body_ball_contact=True)
        assert decision.controller == "contact"
        assert not decision.transitioned
        assert decision.handoff_frame == 30
        assert decision.reference_frame == min(frame - 30, 259)
    with pytest.raises(ValueError):
        advance(latch, 300)


@pytest.mark.parametrize(
    "changes",
    [
        dict(gap_x_m=0.0),
        dict(gap_x_m=-0.1),
        dict(gap_x_m=0.900001),
        dict(gap_y_m=0.250001),
        dict(gap_y_m=-0.250001),
        dict(pelvis_z_m=0.649999),
        dict(upright_tilt_rad=0.350001),
    ],
)
def test_current_state_blocks_but_does_not_invent_a_failure(changes):
    latch = WholeBodyContactHandoff(ContactHandoffConfig())
    prefix(latch, 30)
    assert advance(latch, 30, **changes).controller == "approach"
    assert advance(latch, 31).controller == "contact"


def test_previous_contact_excludes_even_if_contact_signal_later_disappears():
    latch = WholeBodyContactHandoff(ContactHandoffConfig())
    prefix(latch, 29)
    assert advance(latch, 29, previous_body_ball_contact=True).excluded_by_previous_body_contact
    for frame in range(30, 300):
        assert advance(latch, frame).controller == "approach"


@pytest.mark.parametrize("entry,allowed", [(29, False), (30, True), (200, True), (201, False)])
def test_fixed_admission_time_boundaries(entry, allowed):
    latch = WholeBodyContactHandoff(ContactHandoffConfig())
    prefix(latch, entry)
    assert (advance(latch, entry).controller == "contact") is allowed


@pytest.mark.parametrize(
    "changes",
    [
        dict(frame=True),
        dict(frame=1),
        dict(gap_x_m=float("nan")),
        dict(gap_y_m=float("inf")),
        dict(pelvis_z_m=True),
        dict(upright_tilt_rad=-0.1),
        dict(upright_tilt_rad=3.2),
        dict(previous_body_ball_contact=1),
        dict(previous_body_ball_contact=True),
    ],
)
def test_bad_input_faults_sequence_without_latching(changes):
    latch = WholeBodyContactHandoff(ContactHandoffConfig())
    with pytest.raises(ValueError):
        advance(latch, 0, **changes)
    with pytest.raises(ValueError, match="faulted"):
        advance(latch, 0)


@pytest.mark.parametrize(
    "changes",
    [
        dict(activation_ceiling="REAL"),
        dict(entry_gap_x_m=1.0),
        dict(entry_gap_x_m=True),
        dict(earliest_frame=0),
        dict(latest_frame=239),
        dict(horizon_frames=360),
        dict(reference_frames=0),
        dict(reference_frames=True),
        dict(reference_frames=3001),
    ],
)
def test_unregistered_protocol_rejected(changes):
    with pytest.raises(ValueError):
        replace(ContactHandoffConfig(), **changes)


def test_contract_denies_physical_authority_and_fake_history():
    contract = ContactHandoffConfig().contract()
    assert not contract["hardware_authorized"]
    assert not contract["promotion_authorized"]
    assert not contract["admission_is_physical_safety_proof"]
    assert "actual measurements" in contract["recurrent_history"]
    assert "executed target" in contract["old_policy_action_input"]
