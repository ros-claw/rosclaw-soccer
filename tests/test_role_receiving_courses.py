import math
from collections import Counter
from dataclasses import replace

import pytest

from rosclaw_soccer.training.role_receiving_courses import (
    ROSTER,
    ReceivingCourse,
    receiving_ball_launch,
    receiving_courses,
    receiving_practice_task,
)


def test_balanced_real_players_and_distinct_seeds():
    courses = receiving_courses(repetitions=2, first_seed=100)
    assert len(courses) == 32
    assert len({c.seed for c in courses}) == 32
    assert Counter(c.agent_id for c in courses) == dict.fromkeys(ROSTER, 4)
    assert len({(c.agent_id, c.speed_mps, c.lateral_m) for c in courses}) == 32


@pytest.mark.parametrize("repetitions,seed", [(True, 1), (0, 1), (129, 1), (1, -1), (1, 2**32)])
def test_reject_bad_plan(repetitions, seed):
    with pytest.raises(ValueError):
        receiving_courses(repetitions=repetitions, first_seed=seed)


def test_launch_rotates_with_team_not_player_identity():
    blue = receiving_ball_launch(
        ReceivingCourse("blue.goalkeeper", 1, 1.25, 0.08), origin=(6.0, 0.0, 0.0), radius_m=0.11
    )
    red = receiving_ball_launch(
        ReceivingCourse("red.goalkeeper", 2, 1.25, 0.08), origin=(0.0, 0.0, 0.0), radius_m=0.11
    )
    assert blue[0] == pytest.approx((6 - red[0][0], -red[0][1], red[0][2]))
    assert blue[1][0] == -red[1][0]


@pytest.mark.parametrize("speed", [math.nan, math.inf, 0.0, 4.0])
def test_reject_invalid_launch(speed):
    with pytest.raises(ValueError):
        receiving_ball_launch(
            ReceivingCourse("red.defender", 1, speed, 0.0), origin=(0.0, 0.0, 0.0), radius_m=0.11
        )


def test_coached_receiving_preserves_identity_recovery_and_other_players(monkeypatch):
    import test_s199_independent_agent_cells as fixtures

    from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent

    monkeypatch.setattr(
        fixtures,
        "_LAYOUT",
        (
            *fixtures._LAYOUT,
            ("red.defender", "red", MatchRole.DEFENDER, (0.5, 0.6, 0.0)),
            ("blue.defender", "blue", MatchRole.DEFENDER, (5.5, -0.6, 0.0)),
        ),
    )
    initial = fixtures._cells()
    bindings = {b.skill: b for c in initial for b in c.self_model.skills}
    cells = tuple(
        replace(
            c,
            self_model=replace(c.self_model, basic_ball_play=True, skills=tuple(bindings.values())),
        )
        for c in initial
    )
    observations = fixtures._observations(cells, possession_agent_id=None)
    assert {c.agent_id for c in cells} == set(ROSTER)
    for cell, obs in zip(cells, observations, strict=True):
        decision = cell.decide(obs)
        assigned = receiving_practice_task(cell, obs, decision, focal_agent_id=cell.agent_id)
        assert assigned.intent is TacticalIntent.RECEIVE
        assert assigned.observation_hash == obs.observation_hash
        assert obs.possession_agent_id is None
        other = next(c.agent_id for c in cells if c.agent_id != cell.agent_id)
        assert receiving_practice_task(cell, obs, decision, focal_agent_id=other) is decision
        unstable = replace(obs, self_state=replace(obs.self_state, stable=False))
        recovery = cell.decide(unstable)
        assert (
            receiving_practice_task(cell, unstable, recovery, focal_agent_id=cell.agent_id)
            is recovery
        )
        with pytest.raises(ValueError):
            receiving_practice_task(cell, unstable, decision, focal_agent_id=cell.agent_id)


def test_keeper_without_first_touch_binding_cannot_be_granted_authority():
    from test_s199_independent_agent_cells import _cells, _observations

    cells = _cells()
    obs = _observations(cells, possession_agent_id=None)[0]
    with pytest.raises(ValueError, match="capability"):
        receiving_practice_task(
            cells[0], obs, cells[0].decide(obs), focal_agent_id=cells[0].agent_id
        )
