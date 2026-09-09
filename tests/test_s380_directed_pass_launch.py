from dataclasses import replace

import pytest

from rosclaw_soccer.growth.pass_handoff import PassHandoff
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def handshake():
    return PassHandoff("blue.playmaker", "blue.finisher", 10.0, launch_target_xy=(2.0, 0.0))


def touch(state, **changes):
    values = dict(
        agent_id="blue.playmaker",
        foot=True,
        force_n=2.0,
        time_sec=12.0,
        ball_xy=(4.0, 0.0),
        ball_velocity_xy=(-0.6, 0.0),
    )
    values.update(changes)
    return state.observe_directed_launch(**values)


def test_preparation_tap_does_not_start_flight():
    state = handshake().observe_contact(
        agent_id="blue.playmaker", foot=True, force_n=5.0, time_sec=11.0
    )
    assert state.source_foot_contact_sec is None
    state = touch(state, ball_velocity_xy=(0.2, 0.1))
    assert state.source_foot_contact_sec is None
    state = touch(state, time_sec=12.5)
    assert state.source_foot_contact_sec == 12.5
    assert not state.expired(15.0)
    assert state.expired(15.5)


@pytest.mark.parametrize(
    "changes",
    [
        dict(force_n=0.0),
        dict(force_n=1.0),
        dict(foot=False),
        dict(ball_velocity_xy=(0.6, 0.0)),
        dict(ball_velocity_xy=(-0.39, 0.2)),
        dict(ball_velocity_xy=(0.0, 1.0)),
        dict(ball_xy=(2.0, 0.0)),
        dict(agent_id="blue.defender"),
    ],
)
def test_no_launch_from_wrong_or_weak_evidence(changes):
    assert touch(handshake(), **changes).source_foot_contact_sec is None


def test_no_infinite_renewal_or_expired_preparation_revival():
    state = touch(handshake())
    assert touch(state, time_sec=14.0).source_foot_contact_sec == 12.0
    assert touch(handshake(), time_sec=13.0).source_foot_contact_sec is None
    assert touch(state, time_sec=15.0).expired(15.0)


def test_opponent_contact_interrupts_even_when_ball_toward_receiver():
    state = touch(handshake(), agent_id="red.finisher")
    assert state.interrupted and state.source_foot_contact_sec is None


@pytest.mark.parametrize("target", [[2.0, 0.0], (float("nan"), 0.0), (True, 0.0), (2000.0, 0.0)])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        replace(handshake(), launch_target_xy=target)


def test_legacy_contact_path_and_new_configuration_identity():
    old = PassHandoff("blue.playmaker", "blue.finisher", 10.0)
    assert (
        old.observe_contact(
            agent_id=old.source, foot=True, force_n=0.1, time_sec=11.0
        ).source_foot_contact_sec
        == 11.0
    )
    with pytest.raises(ValueError, match="bound target"):
        touch(old)
    with pytest.raises(ValueError, match="strict physical"):
        IndependentTeamWorldConfig(directed_pass_launch=True)
    baseline = IndependentTeamWorldConfig(strict_receive_handoff=True)
    assert replace(baseline, directed_pass_launch=True).config_hash != baseline.config_hash


@pytest.mark.parametrize("velocity", [(float("nan"), 0.0), (-1e308, 0.0), (True, 0.0), [-0.6, 0.0]])
def test_invalid_velocity_never_creates_a_launch(velocity):
    with pytest.raises(ValueError, match="finite measured"):
        touch(handshake(), ball_velocity_xy=velocity)
