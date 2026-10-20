import numpy as np
import pytest

from rosclaw_soccer.rsi.mjlab_motion_evaluation import MotionFirstEpisode


def test_first_episode_masks_repeated_reset_successes_and_rewards():
    score = MotionFirstEpisode(3)
    score.observe(
        np.array([1.0, 2.0, 3.0]),
        np.array([True, False, False]),
        np.array([False, True, False]),
        np.array([False, True, False]),
    )
    score.observe(
        np.array([100.0, 100.0, 4.0]),
        np.array([False, False, False]),
        np.array([True, True, False]),
        np.array([True, True, False]),
    )
    result = score.summary()
    assert result["first_episode_returns"] == [1, 2, 7]
    assert result["first_episode_steps"] == [1, 1, 2]
    assert result["completed"] == [False, True, False]
    assert result["failed"] == [True, False, False]
    assert result["right_censored"] == [False, False, True]
    assert result["football_skill_proven"] is False


def test_simultaneous_failure_vetoes_exhaustion_and_other_timeout_is_censored():
    score = MotionFirstEpisode(2)
    score.observe(
        np.ones(2), np.array([True, False]), np.ones(2, dtype=bool), np.array([True, False])
    )
    assert score.summary()["completed_count"] == 0
    assert score.summary()["failed_count"] == 1
    assert score.summary()["right_censored_count"] == 1
    assert not score.active.any()


@pytest.mark.parametrize("value", [0, True, 513, 2.0])
def test_bounded_batch(value):
    with pytest.raises(ValueError):
        MotionFirstEpisode(value)


@pytest.mark.parametrize(
    "field,value",
    [
        (0, np.array([np.nan])),
        (0, np.array([1.0, 2.0])),
        (1, np.array([1])),
        (2, np.array([1])),
        (3, np.array([1])),
        (3, np.array([True])),
    ],
)
def test_nonfinite_wrong_shape_or_nonboundary_exhaustion_is_rejected(field, value):
    score = MotionFirstEpisode(1)
    args = [np.ones(1), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool)]
    args[field] = value
    with pytest.raises(ValueError):
        score.observe(*args)
    assert score.summary()["first_episode_steps"] == [0]
