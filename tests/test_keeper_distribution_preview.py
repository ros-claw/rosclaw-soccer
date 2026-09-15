from dataclasses import asdict, replace

import pytest
import test_s220_basic_ball_play as basic

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def course(monkeypatch):
    return basic.course.__wrapped__(monkeypatch)


def situation(course, team="red", enabled=True):
    cell = next(c for c in course.cells if c.agent_id == team + ".goalkeeper")
    cell = replace(
        cell,
        tactical_profile=replace(
            cell.tactical_profile,
            anticipatory_contact=True,
            keeper_distribution_preview=enabled,
        ),
    )
    value = basic.observation(course, cell.agent_id)

    def point(x, y, z=0.78):
        return (x, y, z) if team == "red" else (6 - x, -y, z)

    value = replace(
        value,
        self_state=replace(value.self_state, position_m=point(-0.95, 0)),
        own_goal_m=point(-1.5, 0, 0),
        opponent_goal_m=point(7.5, 0, 0),
        ball_position_m=point(-0.4, 0, 0.115),
        ball_velocity_mps=(0.0, 0.0, 0.0),
        ball_chaser_agent_id=cell.agent_id,
        possession_agent_id=None,
        teammate_states=tuple(
            replace(s, position_m=point(1.3, -0.8)) for s in value.teammate_states
        ),
        opponent_states=tuple(replace(s, position_m=point(4, 2)) for s in value.opponent_states),
    )
    return cell, value


@pytest.mark.parametrize("team", ["red", "blue"])
def test_keeper_can_negotiate_before_contact_without_claiming_possession(course, team):
    cell, value = situation(course, team)
    decision = cell.decide(value)
    assert decision.intent is TacticalIntent.PASS
    assert decision.target_agent_id in cell.self_model.teammate_ids
    assert value.possession_agent_id is None


def test_default_preserves_receive_and_legacy_hash(course):
    cell, value = situation(course, enabled=False)
    assert cell.decide(value).intent is TacticalIntent.RECEIVE
    legacy = asdict(cell.tactical_profile)
    for field in (
        "blocked_shot_layoff",
        "moving_ball_finish_intent",
        "keeper_distribution_preview",
    ):
        legacy.pop(field)
    assert cell.tactical_profile.profile_hash == hash_json(legacy)


@pytest.mark.parametrize(
    "change",
    ["incoming", "airborne", "fast", "other_chaser", "opponent", "unstable", "blocked", "unbound"],
)
def test_preview_cannot_override_incoming_ball_or_safety_or_authority(course, change):
    cell, value = situation(course)
    if change == "incoming":
        value = replace(value, active_receive_source_agent_id=cell.self_model.teammate_ids[0])
    elif change == "airborne":
        value = replace(value, ball_position_m=(-0.4, 0, 0.9))
    elif change == "fast":
        value = replace(value, ball_velocity_mps=(-2.0, 0.0, 0.0))
    elif change == "other_chaser":
        value = replace(value, ball_chaser_agent_id=cell.self_model.teammate_ids[0])
    elif change == "opponent":
        value = replace(value, possession_agent_id=cell.self_model.opponent_ids[0])
    elif change == "unstable":
        value = replace(value, self_state=replace(value.self_state, stable=False))
    elif change == "blocked":
        value = replace(
            value,
            opponent_states=tuple(
                replace(s, position_m=(0.3, -0.3, 0.78)) for s in value.opponent_states
            ),
        )
    elif change == "unbound":
        cell = replace(cell, self_model=replace(cell.self_model, basic_ball_play=False))
    assert cell.decide(value).intent is not TacticalIntent.PASS


@pytest.mark.parametrize("bad", [1, None, "true"])
def test_preview_flag_is_explicit(course, bad):
    cell, _ = situation(course)
    with pytest.raises(ValueError):
        replace(cell.tactical_profile, keeper_distribution_preview=bad)
