from dataclasses import asdict, replace

import pytest
import test_s220_basic_ball_play as basic_ball_play
from test_s220_basic_ball_play import observation

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def course(monkeypatch):
    return basic_ball_play.course.__wrapped__(monkeypatch)


def situation(course, *, team="red", enabled=True, owned=False):
    other = "blue" if team == "red" else "red"
    cell = next(c for c in course.cells if c.agent_id == team + ".finisher")
    cell = replace(
        cell,
        tactical_profile=replace(
            cell.tactical_profile,
            active_competition=True,
            anticipatory_contact=True,
            blocked_shot_layoff=enabled,
        ),
    )
    value = observation(course, cell.agent_id)

    def point(x, y, z=0.78):
        return (x, y, z) if team == "red" else (6 - x, -y, z)

    value = replace(
        value,
        self_state=replace(value.self_state, position_m=point(3.4, 0)),
        ball_position_m=point(4, 0, 0.115),
        own_goal_m=point(-1.5, 0, 0),
        opponent_goal_m=point(7.5, 0, 0),
        possession_agent_id=cell.agent_id if owned else None,
        teammate_states=tuple(
            replace(
                s, position_m=point(2, -1.5) if s.agent_id.endswith("playmaker") else point(0, -2)
            )
            for s in value.teammate_states
        ),
        opponent_states=tuple(
            replace(
                s,
                position_m=point(5.5, 0.14)
                if s.agent_id == other + ".defender"
                else point(7, 0)
                if s.agent_id.endswith("goalkeeper")
                else point(6, 2.5),
            )
            for s in value.opponent_states
        ),
    )
    return cell, value


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("owned", [False, True])
def test_blocked_finisher_can_lay_off_to_backward_playmaker(course, team, owned):
    cell, value = situation(course, team=team, owned=owned)
    decision = cell.decide(value)
    assert decision.intent is TacticalIntent.PASS
    assert decision.target_agent_id == team + ".playmaker"
    assert value.possession_agent_id == (cell.agent_id if owned else None)


def test_default_and_hash_remain_legacy(course):
    cell, value = situation(course, enabled=False)
    assert cell.decide(value).intent is TacticalIntent.SHOOT
    legacy = asdict(cell.tactical_profile)
    legacy.pop("blocked_shot_layoff")
    legacy.pop("moving_ball_finish_intent")
    assert cell.tactical_profile.profile_hash == hash_json(legacy)
    assert cell.to_dict()["tactical_profile"] == legacy
    assert (
        replace(cell.tactical_profile, blocked_shot_layoff=True).profile_hash
        != cell.tactical_profile.profile_hash
    )


@pytest.mark.parametrize("bad", [1, "true", None])
def test_flag_requires_boolean(course, bad):
    cell, _ = situation(course)
    with pytest.raises(ValueError):
        replace(cell.tactical_profile, blocked_shot_layoff=bad)


def test_keeper_alone_does_not_cancel_shot(course):
    cell, value = situation(course)
    value = replace(
        value,
        opponent_states=tuple(
            replace(s, position_m=(5.5, 2.5, 0.78)) if s.agent_id.endswith("defender") else s
            for s in value.opponent_states
        ),
    )
    assert cell.decide(value).intent is TacticalIntent.SHOOT


@pytest.mark.parametrize("position", [(3.0, 0.0, 0.78), (8.0, 0.0, 0.78)])
def test_blocker_must_be_between_ball_and_goal(course, position):
    cell, value = situation(course)
    value = replace(
        value,
        opponent_states=tuple(
            replace(s, position_m=position) if s.agent_id.endswith("defender") else s
            for s in value.opponent_states
        ),
    )
    assert cell.decide(value).intent is TacticalIntent.SHOOT


def test_unsafe_or_unstable_outlet_is_not_used(course):
    cell, value = situation(course)
    unstable = replace(
        value, teammate_states=tuple(replace(s, stable=False) for s in value.teammate_states)
    )
    assert cell.decide(unstable).intent is TacticalIntent.SHOOT
    blocked = replace(
        value,
        opponent_states=tuple(
            replace(s, position_m=(2.8, -0.75, 0.78)) if s.agent_id.endswith("playmaker") else s
            for s in value.opponent_states
        ),
    )
    assert cell.decide(blocked).intent is TacticalIntent.SHOOT


def test_incoming_pass_and_recovery_take_priority(course):
    cell, value = situation(course)
    incoming = replace(value, active_receive_source_agent_id="red.playmaker")
    assert cell.decide(incoming).intent is TacticalIntent.RECEIVE
    unstable = replace(value, self_state=replace(value.self_state, stable=False))
    assert cell.decide(unstable).intent is TacticalIntent.RECOVER


def test_does_not_override_opponent_possession(course):
    cell, value = situation(course)
    value = replace(value, possession_agent_id="blue.defender")
    assert cell.decide(value).intent is not TacticalIntent.PASS


def test_layoff_flag_does_not_grant_unbound_passing_skill(course):
    cell, value = situation(course)
    cell = replace(cell, self_model=replace(cell.self_model, basic_ball_play=False))
    assert cell.decide(value).intent is TacticalIntent.SHOOT
