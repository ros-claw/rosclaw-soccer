from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from rosclaw_soccer.training.receiving_bias_search import (
    ReceivingBiasForecast,
    receiving_acquisition_confirmed,
    receiving_bias_beam_seeds,
    receiving_bias_candidates,
    receiving_bias_is_clean,
    receiving_bias_search_reason,
)


def diagnostics(**changes):
    values = dict(
        unsafe=False,
        foot_contact_samples=5,
        nonfoot_contact_samples=2,
        minimum_shin_gap_m=-0.001,
        terminal_ball_speed=0.2,
    )
    values.update(changes)
    return values


def row(*, nonfoot=2, gap=-0.001, speed=0.2, change=0.001, unsafe=False, foot=5):
    return dict(
        diagnostics=dict(
            unsafe=unsafe,
            foot_contact_samples=foot,
            nonfoot_contact_samples=nonfoot,
            minimum_shin_gap_m=gap,
            terminal_ball_speed=speed,
        ),
        change_squared=change,
    )


def test_original_coordinate_order_bounds_and_owned_arrays():
    source = np.zeros(12)
    candidates = receiving_bias_candidates(source)
    assert len(candidates) == 24
    assert [label for label, _ in candidates[:4]] == [
        "joint_6_-1_0.02",
        "joint_6_1_0.02",
        "joint_6_-1_0.06",
        "joint_6_1_0.06",
    ]
    assert all(np.count_nonzero(value) == 1 and not value[:6].any() for _, value in candidates)
    assert all(np.max(abs(value)) <= 0.06 for _, value in candidates)
    candidates[0][1][:] = 1
    assert not source.any() and np.count_nonzero(candidates[1][1]) == 1
    edge = receiving_bias_candidates([0.06] * 12)
    assert all(np.max(abs(value)) <= 0.06 for _, value in edge)


@pytest.mark.parametrize(
    "value",
    [
        [0.0] * 11,
        [float("nan")] * 12,
        [float("inf")] * 12,
        [0.061] * 12,
        [True] * 12,
        [0.0] * 11 + [False],
        ["0"] * 12,
        [0j] * 12,
        np.zeros((1, 12)),
    ],
)
def test_invalid_bias(value):
    with pytest.raises(ValueError):
        receiving_bias_candidates(value)


def test_clean_forecast_is_not_partial_seed_or_execution_certificate():
    value = ReceivingBiasForecast.from_mapping(diagnostics())
    assert not value.clean
    assert replace(value, nonfoot_contact_samples=0).clean
    assert not replace(value, nonfoot_contact_samples=0, unsafe=True).clean
    assert not replace(value, nonfoot_contact_samples=0, foot_contact_samples=0).clean
    assert not replace(value, nonfoot_contact_samples=0, terminal_ball_speed=0.350001).clean
    assert receiving_bias_is_clean(diagnostics(nonfoot_contact_samples=0))
    with pytest.raises(ValueError):
        receiving_bias_is_clean(diagnostics(unsafe=1))


@pytest.mark.parametrize(
    "changes",
    [
        {"unsafe": 1},
        {"foot_contact_samples": True},
        {"nonfoot_contact_samples": -1},
        {"minimum_shin_gap_m": float("nan")},
        {"terminal_ball_speed": float("inf")},
        {"terminal_ball_speed": -0.1},
        {"terminal_ball_speed": True},
    ],
)
def test_invalid_forecast(changes):
    values = diagnostics()
    values.update(changes)
    with pytest.raises(ValueError):
        ReceivingBiasForecast.from_mapping(values)


def test_missing_forecast_is_rejected():
    with pytest.raises(ValueError):
        ReceivingBiasForecast.from_mapping({})


def test_beam_order_and_no_input_mutation():
    rows = [
        row(nonfoot=2),
        row(nonfoot=1, gap=-0.002),
        row(nonfoot=1, gap=-0.001),
        row(nonfoot=0, unsafe=True),
        row(nonfoot=0, foot=0),
    ]
    before = repr(rows)
    assert receiving_bias_beam_seeds(rows) == (2, 1)
    assert repr(rows) == before
    assert receiving_bias_beam_seeds([row(), row(), row()]) == (0, 1)
    assert receiving_bias_beam_seeds([]) == ()
    assert receiving_bias_beam_seeds([row(unsafe=True)]) == ()


@pytest.mark.parametrize("width", [True, 0, 9, 2.0])
def test_invalid_width(width):
    with pytest.raises(ValueError):
        receiving_bias_beam_seeds([row()], width=width)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_corrupt_candidate_does_not_silently_disappear(value):
    with pytest.raises(ValueError):
        receiving_bias_beam_seeds([row(nonfoot=0), row(change=value)])


def test_batch_is_bounded_and_complete():
    with pytest.raises(ValueError):
        receiving_bias_beam_seeds([row()] * 257)
    with pytest.raises(ValueError):
        receiving_bias_beam_seeds([{"change_squared": 0}])


def test_search_trigger_matches_old_rule_and_explicit_acquisition_extension():
    for unsafe, foot, nonfoot, speed, observed in product(
        (False, True), (0, 3), (0, 2), (0.0, 0.35, 0.350001, 0.8), (False, True)
    ):
        values = diagnostics(
            unsafe=unsafe,
            foot_contact_samples=foot,
            nonfoot_contact_samples=nonfoot,
            terminal_ball_speed=speed,
        )
        old = unsafe or nonfoot > 0 or (foot > 0 and speed > 0.35)
        assert bool(receiving_bias_search_reason(values, first_contact_observed=observed)) == old
        new = receiving_bias_search_reason(
            values, first_contact_observed=observed, seek_first_contact=True
        )
        assert bool(new) == (old or (not observed and foot == 0))


def test_missing_touch_is_not_ball_escape_or_synthetic_observation():
    values = diagnostics(nonfoot_contact_samples=0, foot_contact_samples=0, terminal_ball_speed=0.8)
    original = values.copy()
    assert receiving_bias_search_reason(values, first_contact_observed=False) is None
    assert (
        receiving_bias_search_reason(values, first_contact_observed=False, seek_first_contact=True)
        == "missing_first_touch"
    )
    assert (
        receiving_bias_search_reason(values, first_contact_observed=True, seek_first_contact=True)
        is None
    )
    assert values == original


@pytest.mark.parametrize("observed,enabled", [(None, True), (1, True), (False, 1), (False, None)])
def test_search_requires_explicit_contact_semantics(observed, enabled):
    with pytest.raises(ValueError):
        receiving_bias_search_reason(
            diagnostics(), first_contact_observed=observed, seek_first_contact=enabled
        )


def test_search_validates_all_forecast_fields_before_short_circuit():
    with pytest.raises(ValueError):
        receiving_bias_search_reason(
            diagnostics(unsafe=True, terminal_ball_speed=float("nan")),
            first_contact_observed=False,
            seek_first_contact=True,
        )
    assert (
        receiving_bias_search_reason(diagnostics(unsafe=True), first_contact_observed=False)
        == "body_unsafe"
    )
    assert (
        receiving_bias_search_reason(diagnostics(), first_contact_observed=False)
        == "nonfoot_contact"
    )
    assert (
        receiving_bias_search_reason(
            diagnostics(nonfoot_contact_samples=0, terminal_ball_speed=0.36),
            first_contact_observed=True,
        )
        == "ball_escape"
    )


def test_horizon_edge_contact_is_not_settled_acquisition():
    values = diagnostics(nonfoot_contact_samples=0, first_foot_contact_sec=0.478)
    assert receiving_bias_is_clean(values)
    assert not receiving_acquisition_confirmed(values, horizon_sec=0.5)
    assert receiving_acquisition_confirmed(values, horizon_sec=1.0)
    values["terminal_ball_speed"] = 0.676
    assert not receiving_acquisition_confirmed(values, horizon_sec=1.0)
    values.update(terminal_ball_speed=0.2, first_foot_contact_sec=0.5)
    assert receiving_acquisition_confirmed(values, horizon_sec=1.0)
    values["first_foot_contact_sec"] = 0.500001
    assert not receiving_acquisition_confirmed(values, horizon_sec=1.0)


def test_missing_contact_cannot_be_confirmed_and_does_not_mutate_input():
    values = diagnostics(foot_contact_samples=0, first_foot_contact_sec=None)
    before = values.copy()
    assert not receiving_acquisition_confirmed(values, horizon_sec=1.0)
    assert values == before
    with pytest.raises(ValueError):
        receiving_acquisition_confirmed(diagnostics(), horizon_sec=1.0)


@pytest.mark.parametrize("contact", [None, True, -0.001, 1.01, float("nan"), float("inf"), "0.1"])
def test_invalid_or_inconsistent_predicted_contact_timing(contact):
    with pytest.raises(ValueError):
        receiving_acquisition_confirmed(
            diagnostics(first_foot_contact_sec=contact), horizon_sec=1.0
        )


@pytest.mark.parametrize("value", [True, 0, -0.1, 2.01, float("nan"), float("inf")])
def test_invalid_prediction_and_settling_horizons(value):
    values = diagnostics(first_foot_contact_sec=0.1)
    with pytest.raises(ValueError):
        receiving_acquisition_confirmed(values, horizon_sec=value)
    with pytest.raises(ValueError):
        receiving_acquisition_confirmed(values, horizon_sec=1.0, minimum_post_contact_sec=value)


def test_inconsistent_prediction_is_rejected_even_if_already_unsafe():
    with pytest.raises(ValueError):
        receiving_acquisition_confirmed(
            diagnostics(unsafe=True, foot_contact_samples=0, first_foot_contact_sec=0.1),
            horizon_sec=1.0,
        )
    with pytest.raises(ValueError):
        receiving_acquisition_confirmed(diagnostics(first_foot_contact_sec=0.1), horizon_sec=0.2)
