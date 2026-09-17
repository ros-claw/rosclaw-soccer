from dataclasses import replace

import pytest
from test_s199_independent_agent_cells import _cells, _observations

from rosclaw_soccer.growth.ball_role_tasks import ball_role_task
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent


def setup():
    cells = tuple(
        replace(c, tactical_profile=replace(c.tactical_profile, active_competition=True))
        for c in _cells()
    )
    return cells, _observations(cells, possession_agent_id=None)


def test_one_challenger_per_team_and_distinct_support():
    cells, obs = setup()
    result = [ball_role_task(c, o, c.decide(o)) for c, o in zip(cells, obs, strict=True)]
    for team in ("red", "blue"):
        assert (
            sum(
                d.agent_id.startswith(team + ".") and reason == "challenge_observed_ball"
                for d, reason in result
            )
            == 1
        )
    assert any(reason.startswith("offer_") for _, reason in result)
    assert all(
        d.observation_hash == o.observation_hash for (d, _), o in zip(result, obs, strict=True)
    )


def test_preserves_recovery_keeper_and_ball_action():
    cells, obs = setup()
    for c, o in zip(cells, obs, strict=True):
        unstable = replace(o, self_state=replace(o.self_state, stable=False))
        d = c.decide(unstable)
        assert ball_role_task(c, unstable, d) == (d, "recover_before_ball_task")
    c, o = cells[0], obs[0]
    d = c.decide(o)
    assert ball_role_task(c, o, d)[0] is d
    owned = _observations(cells, possession_agent_id="red.playmaker")[1]
    d = cells[1].decide(owned)
    assert d.intent is TacticalIntent.PASS
    assert ball_role_task(cells[1], owned, d)[0] is d


def test_reception_and_unknown_observation_cannot_be_overridden():
    cells, obs = setup()
    c, o = cells[2], obs[2]
    incoming = replace(o, active_receive_source_agent_id="red.playmaker")
    d = c.decide(incoming)
    assert ball_role_task(c, incoming, d) == (d, "honor_live_reception")
    with pytest.raises(ValueError):
        ball_role_task(c, o, d)
    with pytest.raises(ValueError):
        ball_role_task(cells[1], o, c.decide(o))


def test_support_targets_stay_inside_and_ball_relative():
    cells, obs = setup()
    c = cells[2]
    targets = []
    for y in (-1.0, 1.0):
        o = replace(obs[2], ball_position_m=(2.15, y, 0.115), possession_agent_id="red.playmaker")
        d, reason = ball_role_task(c, o, c.decide(o))
        assert -1.25 <= d.target_position_m[0] <= 7.25
        assert abs(d.target_position_m[1]) <= 2.8
        targets.append(d.target_position_m)
    assert targets[0] != targets[1]


def test_four_roles_include_defender_in_team_challenge(monkeypatch):
    import test_s199_independent_agent_cells as fixtures

    monkeypatch.setattr(
        fixtures,
        "_LAYOUT",
        (
            *fixtures._LAYOUT,
            ("red.defender", "red", MatchRole.DEFENDER, (2.1, -0.3, 0.0)),
            ("blue.defender", "blue", MatchRole.DEFENDER, (2.3, -0.4, 0.0)),
        ),
    )
    initial = fixtures._cells()
    bindings = {b.skill: b for c in initial for b in c.self_model.skills}
    cells = tuple(
        replace(
            c,
            self_model=replace(c.self_model, basic_ball_play=True, skills=tuple(bindings.values())),
            tactical_profile=replace(c.tactical_profile, active_competition=True),
        )
        for c in initial
    )
    obs = fixtures._observations(cells, possession_agent_id=None)
    outcomes = [ball_role_task(c, o, c.decide(o)) for c, o in zip(cells, obs, strict=True)]
    challenged = {d.agent_id for d, reason in outcomes if reason == "challenge_observed_ball"}
    assert challenged == {"red.defender", "blue.defender"}
    assert sum(reason == "protect_goal_or_distribute" for _, reason in outcomes) == 2


def test_roster_permutation_does_not_change_task_target():
    cells, obs = setup()
    for c, o in zip(cells, obs, strict=True):
        reordered = replace(
            o,
            teammate_states=tuple(reversed(o.teammate_states)),
            opponent_states=tuple(reversed(o.opponent_states)),
        )
        d, reason = ball_role_task(c, o, c.decide(o))
        r, other_reason = ball_role_task(c, reordered, c.decide(reordered))
        assert (d.intent, d.target_position_m, reason) == (
            r.intent,
            r.target_position_m,
            other_reason,
        )
