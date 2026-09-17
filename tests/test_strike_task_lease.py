from dataclasses import replace

import pytest

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.growth.strike_task_lease import should_release_strike_task
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


@pytest.mark.parametrize("intent", list(TacticalIntent))
def test_only_lost_ball_reacquisition_or_recovery_releases(intent):
    assert should_release_strike_task(
        intent=intent, owns_ball=False, motor_executing=False, phase_active=False
    ) == (
        intent
        in {
            TacticalIntent.RECEIVE,
            TacticalIntent.PRESS,
            TacticalIntent.INTERCEPT,
            TacticalIntent.RECOVER,
        }
    )


@pytest.mark.parametrize("state", ["owns_ball", "motor_executing", "phase_active"])
def test_never_interrupt_executing_or_owned_strike(state):
    kwargs = dict(owns_ball=False, motor_executing=False, phase_active=False)
    kwargs[state] = True
    assert not should_release_strike_task(intent=TacticalIntent.RECEIVE, **kwargs)


def test_explicit_config_changes_identity():
    c = IndependentTeamWorldConfig()
    assert not c.revalidate_strike_task_lease
    assert (
        c.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    assert c.config_hash != replace(c, revalidate_strike_task_lease=True).config_hash
    with pytest.raises(ValueError):
        replace(c, revalidate_strike_task_lease=1)


def test_ambiguous_state_rejected():
    with pytest.raises(ValueError):
        should_release_strike_task(
            intent="receive", owns_ball=False, motor_executing=False, phase_active=False
        )
