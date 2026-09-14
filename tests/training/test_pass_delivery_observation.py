import dataclasses

import numpy as np
import pytest

from rosclaw_soccer.training.pass_delivery_observation import observe_pass_delivery


def arguments():
    return dict(
        time_sec=np.array([1.0, 2.0, 3.0, 4.0]),
        position_m=np.array([[4.0, 0.0, 0.1], [3.0, 0.2, 0.1], [1.0, 0.4, 0.2], [0.0, 0.6, 0.3]]),
        velocity_mps=np.array(
            [[-1.0, 0.2, 0.0], [-2.0, 0.2, 0.1], [-3.0, 0.4, 0.2], [-1.0, 0.2, 0.0]]
        ),
        plane_origin_xy_m=np.array([2.0, 0.3]),
        plane_forward_xy=np.array([-1.0, 0.0]),
        receiver_contact_sec=3.1,
    )


def test_interpolation_is_readonly_and_records_bracket():
    kwargs = arguments()
    before = {key: value.copy() for key, value in kwargs.items() if isinstance(value, np.ndarray)}
    result = observe_pass_delivery(**kwargs)
    assert result is not None
    assert result.segment_index == 1
    assert result.fraction == 0.5
    assert result.sample_interval_sec == (2.0, 3.0)
    assert result.time_sec == 2.5
    assert result.position_m == pytest.approx((2.0, 0.3, 0.15))
    assert result.velocity_mps == pytest.approx((-2.5, 0.3, 0.15))
    assert result.signed_lateral_error_m == pytest.approx(0.0)
    for key, value in before.items():
        np.testing.assert_array_equal(kwargs[key], value)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.time_sec = 1.0


@pytest.mark.parametrize("contact", [0.0, 2.4, 2.5, 2.9, 3.0])
def test_contact_inside_or_at_bracket_rejects_even_if_interpolated_time_precedes_it(contact):
    kwargs = arguments()
    kwargs["receiver_contact_sec"] = contact
    assert observe_pass_delivery(**kwargs) is None


def test_no_contact_and_scaled_direction():
    kwargs = arguments()
    kwargs["receiver_contact_sec"] = None
    kwargs["plane_forward_xy"] *= 10
    assert observe_pass_delivery(**kwargs).time_sec == 2.5


def test_no_crossing_is_explicit():
    kwargs = arguments()
    kwargs["position_m"][:, 0] = 4.0
    assert observe_pass_delivery(**kwargs) is None


def test_on_plane_initial_sample_not_a_crossing():
    kwargs = arguments()
    kwargs["position_m"][:, 0] = [2.0, 1.0, 0.0, -1.0]
    assert observe_pass_delivery(**kwargs) is None


@pytest.mark.parametrize("key", ["time_sec", "position_m", "velocity_mps"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_telemetry_refused(key, value):
    kwargs = arguments()
    kwargs[key].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        observe_pass_delivery(**kwargs)


@pytest.mark.parametrize(
    "clock", [[1.0, 1.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0], [-1.0, 0.0, 1.0, 2.0]]
)
def test_bad_clock(clock):
    kwargs = arguments()
    kwargs["time_sec"] = np.array(clock)
    with pytest.raises(ValueError, match="clock"):
        observe_pass_delivery(**kwargs)


@pytest.mark.parametrize("contact", [True, -1.0, float("nan"), float("inf"), "3"])
def test_bad_contact_time(contact):
    kwargs = arguments()
    kwargs["receiver_contact_sec"] = contact
    with pytest.raises(ValueError, match="contact"):
        observe_pass_delivery(**kwargs)


@pytest.mark.parametrize("key", ["time_sec", "position_m", "velocity_mps"])
def test_integer_telemetry_refused(key):
    kwargs = arguments()
    kwargs[key] = kwargs[key].astype(np.int64)
    with pytest.raises(ValueError, match="floating-point"):
        observe_pass_delivery(**kwargs)


def test_misaligned_velocity_refused():
    kwargs = arguments()
    kwargs["velocity_mps"] = kwargs["velocity_mps"][:-1]
    with pytest.raises(ValueError, match="aligned"):
        observe_pass_delivery(**kwargs)


def test_plane_is_validated_even_if_contact_early():
    kwargs = arguments()
    kwargs["receiver_contact_sec"] = 0.0
    kwargs["plane_forward_xy"][:] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        observe_pass_delivery(**kwargs)
