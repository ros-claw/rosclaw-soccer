import numpy as np
import pytest

from rosclaw_soccer.rsi.outcome_forecast_kick_player import (
    candidate_native_budget,
    choose_window_rescue,
)


def candidate(
    name, *, passing=False, observed=True, healthy=True, clean=True, reward=1.0, delta=0.0
):
    target = np.zeros(29)
    target[0] = delta
    return (
        name,
        target,
        None,
        {},
        dict(
            window_pass=passing,
            quality_observed=observed,
            predicted_body_healthy=healthy,
            predicted_nonfoot_clean=clean,
            reward=reward,
        ),
    )


def test_unchanged_parent_success_never_sacrificed_for_higher_predicted_reward():
    assert (
        choose_window_rescue(
            [
                candidate("parent", passing=True),
                candidate("new", passing=True, reward=100.0, delta=0.03),
            ]
        )
        == 0
    )


def test_incomplete_or_unsafe_windows_never_select_an_intervention():
    assert (
        choose_window_rescue(
            [
                candidate("parent"),
                candidate("censored", passing=True, observed=False),
                candidate("fall", passing=True, healthy=False),
                candidate("shin", passing=True, clean=False),
            ]
        )
        == 0
    )


def test_eligible_observed_rescue_then_reward_and_smallest_change_tie_break():
    assert (
        choose_window_rescue(
            [
                candidate("parent"),
                candidate("no_pass", reward=100.0),
                candidate("rescue1", passing=True, reward=3.0, delta=0.03),
                candidate("rescue2", passing=True, reward=3.0, delta=0.02),
            ]
        )
        == 3
    )


def test_axis_scope_has_exact_declared_forecast_budget():
    assert candidate_native_budget((0, 3, 4, 6, 9, 10)) == 8320
    assert candidate_native_budget(tuple(range(12))) == 16000
    assert candidate_native_budget(tuple(range(12)), response=True) == 16640


@pytest.mark.parametrize("axes", [(), (0, 0), (12,), (True,), [0, 1]])
def test_invalid_axis_scope_cannot_expand_forecast_authority(axes):
    with pytest.raises(ValueError):
        candidate_native_budget(axes)
