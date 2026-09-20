from dataclasses import asdict, replace

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_classroom import (
    R0_TEACHER_HASH,
    R0_WORLD_HASH,
    coached_receiving_cells,
    r0_receiving_configuration,
)


def test_r0_explicit_config_includes_historical_script_changes():
    world, teacher = r0_receiving_configuration()
    assert world.config_hash == R0_WORLD_HASH
    assert hash_json(asdict(teacher)) == R0_TEACHER_HASH
    assert world.receive_pacing_ratio == 0.2
    assert world.loose_ball_capture_follow_navigation
    assert teacher.committed_receive_ankle_lateral_offset_m == 0.12
    assert not teacher.one_touch_finish_enabled


def test_drift_is_rejected_not_renamed_r0(monkeypatch):
    import rosclaw_soccer.training.receiving_classroom as classroom

    original = classroom.collection_world
    monkeypatch.setattr(
        classroom,
        "collection_world",
        lambda *a: replace(original(*a), minimum_pelvis_height_m=0.56),
    )
    with pytest.raises(ValueError, match="drift"):
        r0_receiving_configuration()


def test_instance_coach_has_no_global_decide_patch(monkeypatch):
    import test_s199_independent_agent_cells as fixtures

    from rosclaw_soccer.growth.independent_agent_cell import RosclawSoccerAgentCell
    from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
    from rosclaw_soccer.training.role_receiving_courses import receiving_practice_task

    monkeypatch.setattr(
        fixtures,
        "_LAYOUT",
        (
            *fixtures._LAYOUT,
            ("red.defender", "red", MatchRole.DEFENDER, (0.5, 0.6, 0.0)),
            ("blue.defender", "blue", MatchRole.DEFENDER, (5.5, -0.6, 0.0)),
        ),
    )
    original = RosclawSoccerAgentCell.decide
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
    coached = coached_receiving_cells(cells, focal_agent_id="blue.playmaker")
    for before, after, obs in zip(cells, coached, observations, strict=True):
        expected = receiving_practice_task(
            before, obs, before.decide(obs), focal_agent_id="blue.playmaker"
        )
        assert after.decide(obs) == expected
        if before.agent_id == "blue.playmaker":
            assert after.cell_hash != before.cell_hash
            assert after.decide(obs).intent is TacticalIntent.RECEIVE
        else:
            assert before is after
    assert RosclawSoccerAgentCell.decide is original
    with pytest.raises(ValueError, match="wrap"):
        coached_receiving_cells(coached, focal_agent_id="blue.playmaker")
