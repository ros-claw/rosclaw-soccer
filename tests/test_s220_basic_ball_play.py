from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellObservation,
    AgentPhysicalState,
    build_independent_agent_cell,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, SoccerSkill, TacticalIntent
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.world.field import G1TrainingGoalSpec


@pytest.fixture
def course(monkeypatch):
    import rosclaw_soccer.training.four_vs_four_match as module

    digest = "sha256:" + "a" * 64
    foundation = build_independent_agent_cell(
        agent_id="red.goalkeeper",
        team_id="red",
        primary_role=MatchRole.GOALKEEPER,
        teammate_ids=("red.playmaker",),
        opponent_ids=("blue.playmaker",),
        body_hash=digest,
        foundation_policy_hash=digest,
        home_position_m=(0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        module,
        "build_independent_three_vs_three_fixture",
        lambda _: SimpleNamespace(
            cells=(foundation,),
            foundation_policy_hash=digest,
            goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0),
        ),
    )
    return build_four_vs_four_fixture(Path("unused"), basic_ball_play=True)


def observation(course, agent_id, owner=None):
    states = {
        p.agent_id: AgentPhysicalState(
            p.agent_id, (*p.origin_m[:2], 0.78), (0.0, 0.0, 0.0), 0.78, 0.02, True
        )
        for p in course.players
    }
    cell = next(c for c in course.cells if c.agent_id == agent_id)
    own = states[agent_id]
    return AgentCellObservation(
        observer_agent_id=agent_id,
        time_sec=0.0,
        ball_position_m=(own.position_m[0] + 0.55, own.position_m[1] - 0.12, 0.115),
        ball_velocity_mps=(0.0, 0.0, 0.0),
        own_goal_m=(-1.5, 0.0, 0.0),
        opponent_goal_m=(7.5, 0.0, 0.0),
        possession_agent_id=owner,
        ball_chaser_agent_id=agent_id,
        self_state=own,
        teammate_states=tuple(states[i] for i in cell.self_model.teammate_ids),
        opponent_states=tuple(states[i] for i in cell.self_model.opponent_ids),
    )


def test_defender_acquires_then_passes_instead_of_role_dead_end(course):
    defender = next(c for c in course.cells if c.agent_id == "red.defender")
    loose = observation(course, defender.agent_id)
    assert defender.decide(loose).intent is TacticalIntent.RECEIVE
    held = replace(loose, possession_agent_id=defender.agent_id)
    decision = defender.decide(held)
    assert decision.intent is TacticalIntent.PASS
    assert decision.target_agent_id in defender.self_model.teammate_ids
    assert loose.possession_agent_id is None


def test_keeper_can_collect_nearby_ball_and_choose_outlet(course):
    keeper = next(c for c in course.cells if c.agent_id == "red.goalkeeper")
    assert keeper.decide(observation(course, keeper.agent_id)).intent is TacticalIntent.RECEIVE
    decision = keeper.decide(observation(course, keeper.agent_id, keeper.agent_id))
    assert decision.intent is TacticalIntent.PASS and decision.target_agent_id == "red.defender"
    defender = next(c for c in course.cells if c.agent_id == "red.defender")
    support = defender.decide(observation(course, defender.agent_id, keeper.agent_id))
    assert support.intent is TacticalIntent.SUPPORT


def test_new_permissions_are_explicit_untrained_and_do_not_remove_role_boundaries(course):
    legacy = build_four_vs_four_fixture(Path("unused"))
    defender = next(c for c in legacy.cells if c.agent_id == "red.defender")
    assert not defender.self_model.authorizes(TacticalIntent.PASS, SoccerSkill.LEAD_PASS)
    assert "basic_ball_play" not in defender.self_model.to_dict()
    with pytest.raises(ValueError):
        replace(defender.self_model, basic_ball_play=True)
    enabled = next(c for c in course.cells if c.agent_id == "red.defender")
    assert enabled.self_model.skill(SoccerSkill.LEAD_PASS).proficiency == 0
    assert not enabled.self_model.authorizes(TacticalIntent.SHOOT, SoccerSkill.FINISHING)
    assert (
        legacy.fixture_hash
        == build_four_vs_four_fixture(Path("unused"), basic_ball_play=False).fixture_hash
    )
