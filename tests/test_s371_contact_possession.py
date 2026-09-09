import math

import pytest

from rosclaw_soccer.growth.contact_possession import ContactPossession
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def observe(state, frame, **kwargs):
    values = dict(
        time_sec=frame * 0.002,
        foot_agents=frozenset(),
        interrupted=False,
        ball_speed_mps=0.2,
        nearest_foot_m=0.25,
        body_safe=True,
    )
    values.update(kwargs)
    state.observe(**values)


def confirmed():
    state = ContactPossession()
    observe(state, 0, foot_agents=frozenset({"blue.playmaker"}))
    for i in range(1, 151):
        observe(state, i)
    assert state.owner_at(0.3) == "blue.playmaker"
    return state


def test_nearness_never_creates_ownership():
    state = ContactPossession()
    for i in range(300):
        observe(state, i)
    assert state.owner_at(0.598) is None


@pytest.mark.parametrize(
    "change",
    [
        dict(ball_speed_mps=0.51),
        dict(nearest_foot_m=0.36),
        dict(body_safe=False),
        dict(interrupted=True),
        dict(foot_agents=frozenset({"blue.playmaker", "red.finisher"})),
    ],
)
def test_control_loss_requires_new_contact(change):
    state = confirmed()
    observe(state, 151, **change)
    for i in range(152, 400):
        observe(state, i)
    assert state.owner_at(0.798) is None


def test_stale_contact_expires_even_when_near():
    state = confirmed()
    for i in range(151, 1502):
        observe(state, i)
    assert state.owner_at(3.002) is None


def test_gap_and_fresh_contact():
    state = confirmed()
    observe(state, 160)
    assert state.tracking_agent_id is None
    observe(state, 161, foot_agents=frozenset({"red.finisher"}))
    for i in range(162, 312):
        observe(state, i)
    assert state.owner_at(0.622) == "red.finisher"
    assert state.owner_at(0.7) is None


@pytest.mark.parametrize("time", [math.nan, math.inf, -1.0, 0.3, True])
def test_bad_clock_latches_closed(time):
    state = confirmed()
    observe(state, 151, time_sec=time)
    assert state.faulted
    observe(state, 152, foot_agents=frozenset({"red.finisher"}))
    assert state.owner_at(0.304) is None


def test_retention_requires_strict_handoff():
    with pytest.raises(ValueError, match="strict physical"):
        IndependentTeamWorldConfig(controlled_possession_retention=True)
    baseline = IndependentTeamWorldConfig(strict_receive_handoff=True)
    candidate = IndependentTeamWorldConfig(
        strict_receive_handoff=True, controlled_possession_retention=True
    )
    assert baseline.config_hash != candidate.config_hash
