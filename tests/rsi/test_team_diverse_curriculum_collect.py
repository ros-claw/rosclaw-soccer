"""Training curriculum varies physical ball states, not just evidence identities."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rsi_team_diverse_curriculum_collect import training_courses


def test_training_courses_are_reproducible_and_physically_distinct() -> None:
    root = Path(__file__).parents[2]
    protocol = json.loads(
        (root / "docs/rsi/protocols/team-diverse-curriculum-v29.json").read_text()
    )
    first = training_courses(protocol)
    second = training_courses(protocol)
    assert first == second
    assert len(first) == 64
    assert (
        len({(scene.ball_initial_position_m, scene.ball_initial_velocity_mps) for scene in first})
        == 64
    )
    assert all(3.45 <= scene.ball_initial_position_m[0] <= 4.1 for scene in first)
    assert all(-0.9 <= scene.ball_initial_position_m[1] <= -0.5 for scene in first)
    assert all(-0.7 <= scene.ball_initial_velocity_mps[0] <= -0.3 for scene in first)


def test_three_arm_risk_curriculum_has_512_distinct_physical_states() -> None:
    root = Path(__file__).parents[2]
    protocol = json.loads(
        (root / "docs/rsi/protocols/team-paired-risk-curriculum-v56.json").read_text()
    )
    assert [arm["name"] for arm in protocol["arms"]] == [
        "parent",
        "baseline",
        "gate22_cap10",
    ]
    scenes = [scene for batch in range(16) for scene in training_courses(protocol, batch)]
    assert len(scenes) == 512
    assert len({scene.scenario_hash for scene in scenes}) == 512
    assert (
        len({(scene.ball_initial_position_m, scene.ball_initial_velocity_mps) for scene in scenes})
        == 512
    )
