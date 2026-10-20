import pytest

from rosclaw_soccer.rsi.ball_aware_learning_budget import BallAwareLearningBudget


def test_smoke_is_unchanged_and_not_full_kick_horizon():
    b = BallAwareLearningBudget()
    assert (b.worlds, b.iterations, b.seed, b.controls) == (32, 2, 2601, 48)
    assert (b.samples, b.optimizer_updates, b.gpu_world_steps) == (1536, 42, 15360)


def test_course_covers_many_complete_kicks_with_explicit_budget():
    b = BallAwareLearningBudget(True)
    assert (b.worlds, b.iterations, b.seed, b.controls) == (128, 200, 2701, 4800)
    assert (b.samples, b.optimizer_updates, b.gpu_world_steps) == (614400, 4200, 6144000)
    assert b.controls > 15 * 300


@pytest.mark.parametrize("value", [1, 0, "course", None])
def test_no_implicit_protocol_coercion(value):
    with pytest.raises(ValueError):
        BallAwareLearningBudget(value)
