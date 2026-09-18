from dataclasses import replace

import pytest

from rosclaw_soccer.growth.loose_ball_capture import hold_capture_navigation
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


@pytest.mark.parametrize("intent", list(TacticalIntent))
def test_only_acquisition_tasks_can_follow_during_measured_capture(intent):
    assert hold_capture_navigation(
        capture_active=True, follow_enabled=True, live_foundation=True, intent=intent
    ) is (intent not in {TacticalIntent.RECEIVE, TacticalIntent.PRESS, TacticalIntent.INTERCEPT})


@pytest.mark.parametrize("follow,live", [(False, False), (False, True), (True, False)])
def test_legacy_or_frozen_capture_keeps_navigation_hold(follow, live):
    assert hold_capture_navigation(
        capture_active=True,
        follow_enabled=follow,
        live_foundation=live,
        intent=TacticalIntent.RECEIVE,
    )


def test_no_capture_is_not_fabricated():
    assert not hold_capture_navigation(
        capture_active=False,
        follow_enabled=True,
        live_foundation=True,
        intent=TacticalIntent.RECEIVE,
    )


@pytest.mark.parametrize("field", ["capture_active", "follow_enabled", "live_foundation"])
def test_flags_must_be_explicit(field):
    values = dict(
        capture_active=True,
        follow_enabled=True,
        live_foundation=True,
        intent=TacticalIntent.RECEIVE,
    )
    values[field] = 1
    with pytest.raises(ValueError):
        hold_capture_navigation(**values)


def test_configuration_requires_active_balance_and_binds_hash():
    base = IndependentTeamWorldConfig(
        strict_receive_handoff=True, loose_ball_capture_hold=True, loose_ball_capture_control=True
    )
    with pytest.raises(ValueError):
        replace(base, loose_ball_capture_follow_navigation=True)
    live = replace(base, loose_ball_capture_live_foundation=True)
    assert replace(live, loose_ball_capture_follow_navigation=False).config_hash == live.config_hash
    assert replace(live, loose_ball_capture_follow_navigation=True).config_hash != live.config_hash
    with pytest.raises(ValueError):
        replace(live, loose_ball_capture_follow_navigation=1)
