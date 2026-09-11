import numpy as np
import pytest

from rosclaw_soccer.training.strike_entry_geometry import evaluate_strike_entry_geometry


def state(sign=1):
    p = np.zeros((1, 43))
    v = np.zeros((1, 41))
    p[0, 2] = 0.75
    p[0, 3] = 1
    p[0, 36] = sign * 0.59
    v[0, 0] = sign * 0.2
    return p, v


@pytest.mark.parametrize("sign", [-1, 1])
def test_mirrored_physical_geometry_and_no_alias(sign):
    p, v = state(sign)
    result = evaluate_strike_entry_geometry(p, v, attack_sign=sign)
    assert result.eligible.tolist() == [True]
    assert result.depth_m[0] == 0.59
    p[0, 36] = 5
    assert result.depth_m[0] == 0.59
    with pytest.raises(ValueError):
        result.eligible[0] = False


@pytest.mark.parametrize(
    "column,value,expected",
    [
        (36, 0.5, True),
        (36, 0.68, True),
        (36, 0.4999, False),
        (36, 0.6801, False),
        (37, 0.3, True),
        (37, -0.3, True),
        (37, 0.3001, False),
        (2, 0.65, False),
        (2, 0.6501, True),
    ],
)
def test_original_position_boundaries(column, value, expected):
    p, v = state()
    p[0, column] = value
    assert bool(evaluate_strike_entry_geometry(p, v, attack_sign=1).eligible[0]) is expected


def test_speed_is_strict_and_geometry_is_not_a_shot_receipt():
    p, v = state()
    v[0, 0] = 0.1
    assert not evaluate_strike_entry_geometry(p, v, attack_sign=1).eligible[0]
    v[0, 0] = 0.1001
    # Legacy predicate has no heading or ball-height check: don't silently add
    # one, or claim this test covers those separate downstream requirements.
    p[0, 3] = 0
    p[0, 6] = 1
    p[0, 38] = 2
    assert evaluate_strike_entry_geometry(p, v, attack_sign=1).eligible[0]


@pytest.mark.parametrize(
    "problem", ["nan", "inf", "shape", "integer", "quaternion", "direction", "boolean_direction"]
)
def test_bad_observations_fail_closed(problem):
    p, v = state()
    sign = 1
    if problem == "nan":
        p[0, 7] = np.nan
    elif problem == "inf":
        v[0, 8] = np.inf
    elif problem == "shape":
        p = p[:, :42]
    elif problem == "integer":
        v = v.astype(int)
    elif problem == "quaternion":
        p[0, 3] = 2
    elif problem == "direction":
        sign = 0
    else:
        sign = True
    with pytest.raises(ValueError):
        evaluate_strike_entry_geometry(p, v, attack_sign=sign)
