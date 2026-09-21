from copy import deepcopy

import pytest

from rosclaw_soccer.training.receiving_bias_search import receiving_bias_is_clean
from rosclaw_soccer.training.receiving_retention_geometry import receiving_retention_feasible


def forecast(**changes):
    row = dict(
        unsafe=False,
        foot_contact_samples=0,
        nonfoot_contact_samples=0,
        minimum_shin_gap_m=0.1,
        terminal_ball_speed=0.1,
        terminal_distance_m=0.3,
        terminal_height_m=0.11,
    )
    row.update(changes)
    return row


def check(row, *, touch=1.0, now=2.0):
    return receiving_retention_feasible(row, prior_own_touch_sec=touch, observation_time_sec=now)


def test_retention_does_not_require_repeated_collision_or_mutate_input():
    row = forecast()
    before = deepcopy(row)
    assert not receiving_bias_is_clean(row)
    assert check(row)
    assert row == before
    assert not check(row, touch=None)
    assert not check(forecast(foot_contact_samples=5), touch=None)
    assert check(row, touch=0, now=0)


@pytest.mark.parametrize(
    "changes",
    [
        dict(unsafe=True),
        dict(nonfoot_contact_samples=1),
        dict(terminal_ball_speed=0.350001),
        dict(terminal_distance_m=0.350001),
        dict(terminal_height_m=0.200001),
    ],
)
def test_each_original_predicted_constraint_is_required(changes):
    assert not check(forecast(**changes))


def test_thresholds_inclusive_not_relaxed():
    assert check(
        forecast(terminal_ball_speed=0.35, terminal_distance_m=0.35, terminal_height_m=0.2)
    )


@pytest.mark.parametrize("field", ["terminal_distance_m", "terminal_height_m"])
@pytest.mark.parametrize("value", [True, "0.1", -0.01, float("nan"), float("inf"), 10001])
def test_bad_geometry_rejected_even_without_contact(field, value):
    with pytest.raises(ValueError):
        check(forecast(**{field: value}), touch=None)


@pytest.mark.parametrize("field", ["terminal_distance_m", "terminal_height_m", "unsafe"])
def test_missing_fields_rejected(field):
    row = forecast()
    del row[field]
    with pytest.raises(ValueError):
        check(row)


@pytest.mark.parametrize("touch", [True, "1", -1, 2.000001, float("inf"), float("nan")])
def test_predicted_future_or_invalid_touch_cannot_substitute_for_measured_past(touch):
    with pytest.raises(ValueError):
        check(forecast(), touch=touch)


@pytest.mark.parametrize("now", [True, "2", -1, float("inf"), float("nan")])
def test_invalid_observation_time(now):
    with pytest.raises(ValueError):
        check(forecast(), now=now)


@pytest.mark.parametrize(
    "changes", [dict(unsafe=1), dict(nonfoot_contact_samples=-1), dict(foot_contact_samples=True)]
)
def test_inherited_diagnostics_validated(changes):
    with pytest.raises(ValueError):
        check(forecast(**changes))
