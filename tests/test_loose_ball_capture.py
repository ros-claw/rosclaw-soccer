from dataclasses import replace

import pytest

from rosclaw_soccer.growth.loose_ball_capture import admit_loose_ball_capture
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def admission(**changes):
    values = dict(
        intent=TacticalIntent.RECEIVE,
        effector_code=1,
        contact_force_n=2.0,
        fresh_contact=True,
        receive_lease_present=False,
        motor_busy=False,
        capture_busy=False,
    )
    values.update(changes)
    return admit_loose_ball_capture(**values)


@pytest.mark.parametrize("foot", [1, 2])
def test_actual_fresh_receiving_foot_admitted(foot):
    assert admission(effector_code=foot)


@pytest.mark.parametrize(
    "change",
    [
        {"effector_code": 0},
        {"effector_code": 3},
        {"effector_code": 4},
        {"contact_force_n": 1.0},
        {"fresh_contact": False},
        {"receive_lease_present": True},
        {"motor_busy": True},
        {"capture_busy": True},
        {"intent": TacticalIntent.PASS},
        {"intent": TacticalIntent.SHOOT},
        {"intent": TacticalIntent.RECOVER},
    ],
)
def test_no_unmeasured_retrigger_or_live_option_preemption(change):
    assert not admission(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"contact_force_n": float("nan")},
        {"contact_force_n": float("inf")},
        {"contact_force_n": -1},
        {"effector_code": True},
        {"motor_busy": 0},
    ],
)
def test_malformed_inputs_rejected(change):
    with pytest.raises(ValueError):
        admission(**change)


def test_opt_in_requires_strict_contact_and_preserves_default_hash():
    base = IndependentTeamWorldConfig()
    assert (
        base.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    with pytest.raises(ValueError):
        replace(base, loose_ball_capture_hold=True)
    with pytest.raises(ValueError):
        replace(base, strict_receive_handoff=True, loose_ball_capture_hold=1)
    enabled = replace(base, strict_receive_handoff=True, loose_ball_capture_hold=True)
    assert enabled.config_hash != replace(base, strict_receive_handoff=True).config_hash


@pytest.mark.parametrize(
    "changes",
    [
        {"loose_ball_capture_control": True},
        {"loose_ball_capture_hold_sec": float("nan")},
        {"loose_ball_capture_hold_sec": 0.19},
        {"loose_ball_capture_hold_sec": 1.01},
    ],
)
def test_scoped_control_requires_capture_and_bounded_duration(changes):
    with pytest.raises(ValueError):
        replace(IndependentTeamWorldConfig(), **changes)


def test_free_capture_does_not_reconfigure_committed_capture():
    base = IndependentTeamWorldConfig(strict_receive_handoff=True)
    scoped = replace(
        base,
        loose_ball_capture_hold=True,
        loose_ball_capture_control=True,
        loose_ball_capture_hold_sec=0.6,
    )
    assert scoped.post_receive_hold_sec == base.post_receive_hold_sec == 0.2
    assert scoped.post_receive_contact_control is base.post_receive_contact_control is False
    assert scoped.config_hash != base.config_hash


def test_live_balance_requires_explicit_scoped_active_control():
    base = IndependentTeamWorldConfig(strict_receive_handoff=True, loose_ball_capture_hold=True)
    with pytest.raises(ValueError):
        replace(base, loose_ball_capture_live_foundation=True)
    with pytest.raises(ValueError):
        replace(base, loose_ball_capture_control=True, loose_ball_capture_live_foundation=1)
    active = replace(base, loose_ball_capture_control=True)
    live = replace(active, loose_ball_capture_live_foundation=True)
    assert active.config_hash != live.config_hash
    assert live.post_receive_contact_control is False
