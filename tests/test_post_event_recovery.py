from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.post_event_recovery import (
    PostEventRecoveryConfig,
    post_event_recovery_cost,
)


def observations():
    return dict(
        event_observed=np.ones(4, dtype=bool),
        event_age_sec=np.full(4, 0.5),
        clearance_m=np.full(4, 0.6),
        forward_velocity_mps=np.array([-0.4, 0.4, -0.1, 0.0]),
        angular_speed_radps=np.zeros(4),
        upright_cosine=np.ones(4),
    )


def test_backward_cost_preserves_forward_motion_and_tolerance():
    inputs = observations()
    before = {k: v.copy() for k, v in inputs.items()}
    result = post_event_recovery_cost(**inputs)
    np.testing.assert_allclose(result.cost, [0.036, 0, 0, 0], atol=1e-12)
    assert result.enabled.all()
    assert not result.cost.flags.writeable and not result.enabled.flags.writeable
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], before[key])


def test_measured_event_delay_and_clearance_all_required():
    inputs = observations()
    inputs["forward_velocity_mps"][:] = -1
    inputs["event_observed"][0] = False
    inputs["event_age_sec"][1] = 0.2
    inputs["clearance_m"][2] = 0.35
    result = post_event_recovery_cost(**inputs)
    np.testing.assert_array_equal(result.enabled, [False, False, False, True])
    np.testing.assert_array_equal(result.cost[:3], np.zeros(3))


def test_angular_tilt_cost_and_cap():
    inputs = observations()
    inputs["forward_velocity_mps"][:] = 0
    inputs["angular_speed_radps"][:] = 1.5
    inputs["upright_cosine"][:] = np.cos(0.3)
    result = post_event_recovery_cost(**inputs)
    np.testing.assert_allclose(result.cost, 0.02 * (2 * 0.5**2 + 4 * 0.15**2))
    inputs["angular_speed_radps"][:] = 1000
    np.testing.assert_array_equal(post_event_recovery_cost(**inputs).cost, np.full(4, 2.0))


@pytest.mark.parametrize("key", list(observations())[1:])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e7])
def test_nonfinite_or_unbounded_observations_rejected(key, bad):
    inputs = observations()
    inputs[key][0] = bad
    with pytest.raises(ValueError):
        post_event_recovery_cost(**inputs)


@pytest.mark.parametrize("key", ["event_age_sec", "clearance_m", "angular_speed_radps"])
def test_negative_magnitudes_rejected(key):
    inputs = observations()
    inputs[key][0] = -0.1
    with pytest.raises(ValueError):
        post_event_recovery_cost(**inputs)


@pytest.mark.parametrize("bad", [True, -1, np.nan, np.inf, "0.2"])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        replace(PostEventRecoveryConfig(), delay_sec=bad)


def test_shapes_dtypes_and_config_rejected():
    for replacement in [np.ones((4, 1)), np.ones(3), np.ones(4, dtype=np.int64)]:
        inputs = observations()
        inputs["event_age_sec"] = replacement
        with pytest.raises(ValueError):
            post_event_recovery_cost(**inputs)
    with pytest.raises(ValueError):
        post_event_recovery_cost(**observations(), config={})
    with pytest.raises(ValueError):
        replace(PostEventRecoveryConfig(), dt_sec=0)


def test_float32_supported_without_aliasing():
    inputs = observations()
    inputs.update({k: v.astype(np.float32) for k, v in inputs.items() if k != "event_observed"})
    result = post_event_recovery_cost(**inputs)
    assert result.cost.dtype == np.float64
    assert not np.shares_memory(result.enabled, inputs["event_observed"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("delay_sec", 61),
        ("minimum_clearance_m", 101),
        ("backward_tolerance_mps", 101),
        ("angular_tolerance_radps", 101),
        ("tilt_tolerance_rad", 4),
        ("backward_weight", 1001),
        ("dt_sec", 1.1),
        ("per_step_cap", 0),
    ],
)
def test_each_configuration_bound(field, value):
    with pytest.raises(ValueError):
        replace(PostEventRecoveryConfig(), **{field: value})


@pytest.mark.parametrize("mask", [np.ones(4), np.ones((4, 1), dtype=bool), np.array([], bool)])
def test_invalid_event_mask(mask):
    inputs = observations()
    inputs["event_observed"] = mask
    with pytest.raises(ValueError):
        post_event_recovery_cost(**inputs)


def test_invalid_upright_and_roundoff():
    inputs = observations()
    inputs["upright_cosine"][:] = 1.01
    with pytest.raises(ValueError):
        post_event_recovery_cost(**inputs)
    inputs["upright_cosine"][:] = 1 + 1e-7
    np.testing.assert_allclose(post_event_recovery_cost(**inputs).cost, [0.036, 0, 0, 0])


def test_linear_time_scaling_before_cap():
    full = post_event_recovery_cost(**observations())
    half = post_event_recovery_cost(**observations(), config=PostEventRecoveryConfig(dt_sec=0.01))
    np.testing.assert_array_equal(full.cost, 2 * half.cost)
