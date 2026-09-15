import math

import numpy as np
import pytest

from rosclaw_soccer.training.exploration_change import plan_exploration_increase


def plan(previous, request=-2.0, **kwargs):
    return plan_exploration_increase(
        previous,
        requested_log_std=request,
        **(dict(minimum_log_std=-4.0, maximum_log_std=-0.3) | kwargs),
    )


def test_reports_actual_checkpoint_change_without_mutation():
    previous = np.full(29, -3.0, dtype=np.float32)
    report = plan(previous)
    np.testing.assert_array_equal(previous, np.full(29, -3.0))
    assert report.previous_effective_log_std == (-3.0,) * 29
    np.testing.assert_allclose(report.standard_deviation_multipliers, math.e)


@pytest.mark.parametrize(
    "previous,requested",
    [([-3.0], -3.0), ([-6.0], -5.0), ([0.0], 1.0), ([-1.0, -3.0], -2.0)],
)
def test_noop_after_clipping_and_partial_decreases_are_rejected(previous, requested):
    with pytest.raises(ValueError, match="no-op or decreases"):
        plan(np.array(previous), requested)


def test_per_dimension_effective_multipliers_use_the_real_sampler_bounds():
    report = plan(np.array([-6.0, -3.0]), -2.0)
    assert report.previous_effective_log_std == (-4.0, -3.0)
    np.testing.assert_allclose(report.standard_deviation_multipliers, [math.exp(2), math.e])


@pytest.mark.parametrize(
    "previous",
    [
        [],
        np.zeros(0),
        np.zeros((1, 2)),
        np.zeros(2, dtype=int),
        np.array([np.nan]),
        np.array([21.0]),
    ],
)
def test_invalid_checkpoint_rejected(previous):
    with pytest.raises(ValueError):
        plan(previous)


@pytest.mark.parametrize("requested", [True, float("nan"), float("inf"), -21.0, 3.0])
def test_invalid_request_rejected(requested):
    with pytest.raises(ValueError):
        plan(np.array([-3.0]), requested)


def test_invalid_sampler_interval_rejected():
    with pytest.raises(ValueError):
        plan(np.array([-3.0]), minimum_log_std=-0.3, maximum_log_std=-0.3)
