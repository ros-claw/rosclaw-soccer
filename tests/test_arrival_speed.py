import math

import pytest

from rosclaw_soccer.sim.arrival_speed import delayed_arrival_speed

PARAMETERS = dict(
    maximum_speed_mps=1.5,
    assumed_deceleration_mps2=2.0,
    response_delay_sec=0.5,
    arrival_margin_m=0.15,
)


def test_arrival_margin_and_speed_cap():
    assert delayed_arrival_speed(remaining_distance_m=0.0, **PARAMETERS) == 0
    assert delayed_arrival_speed(remaining_distance_m=0.15, **PARAMETERS) == 0
    assert delayed_arrival_speed(remaining_distance_m=1000.0, **PARAMETERS) == 1.5


def test_uncapped_cue_satisfies_declared_delay_equation():
    for distance in (0.16, 0.3, 0.5, 0.8):
        speed = delayed_arrival_speed(remaining_distance_m=distance, **PARAMETERS)
        assert 0 < speed < 1.5
        assert math.isclose(speed * 0.5 + speed**2 / 4, distance - 0.15, abs_tol=1e-12)


def test_larger_delay_reduces_reference_speed_and_zero_delay_is_supported():
    fast = delayed_arrival_speed(
        remaining_distance_m=0.5, **(PARAMETERS | dict(response_delay_sec=0.0))
    )
    nominal = delayed_arrival_speed(remaining_distance_m=0.5, **PARAMETERS)
    delayed = delayed_arrival_speed(
        remaining_distance_m=0.5, **(PARAMETERS | dict(response_delay_sec=1.0))
    )
    assert 0 < delayed < nominal < fast <= 1.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("remaining_distance_m", float("nan")),
        ("remaining_distance_m", -1.0),
        ("remaining_distance_m", True),
        ("maximum_speed_mps", 0.0),
        ("maximum_speed_mps", 11.0),
        ("assumed_deceleration_mps2", 0.0),
        ("response_delay_sec", -1.0),
        ("response_delay_sec", float("inf")),
        ("arrival_margin_m", 6.0),
    ],
)
def test_invalid_parameters_rejected(field, value):
    args = dict(PARAMETERS, remaining_distance_m=1.0)
    args[field] = value
    with pytest.raises(ValueError):
        delayed_arrival_speed(**args)
