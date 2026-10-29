from dataclasses import dataclass

import pytest

from rosclaw_soccer.world.field import G1TrainingGoalSpec
from scripts.rsi_four_v_four_assisted_returns import ball_fixture


@dataclass(frozen=True)
class Fixture:
    goal: G1TrainingGoalSpec
    other: str = "retained"


def test_standard_ball_is_explicit_and_preserves_legacy_fixture():
    fixture = Fixture(G1TrainingGoalSpec())
    assert ball_fixture(fixture, standard_ball=False) is fixture
    updated = ball_fixture(fixture, standard_ball=True)
    assert updated.goal.ball_radius_m == 0.110 and updated.goal.ball_mass_kg == 0.430
    assert updated.other == fixture.other and fixture.goal.ball_radius_m == 0.115
    with pytest.raises(ValueError):
        ball_fixture(fixture, standard_ball=1)
