import numpy as np
import pytest

from rosclaw_soccer.rsi.initial_ball_factorial import initial_ball_factorial


@pytest.mark.parametrize("ground", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_explicit_factors_only_change_owned_ball_initial_components(ground, zero):
    position = np.array([1.5, -1.2, 0.159])
    spin = np.array([2.9, -17.6, -13.0])
    before_position, before_spin = position.copy(), spin.copy()
    p, w = initial_ball_factorial(position, spin, ground_ball=ground, zero_spin=zero)
    assert np.array_equal(position, before_position)
    assert np.array_equal(spin, before_spin)
    assert np.array_equal(p[:2], position[:2])
    assert p[2] == (0.11 if ground else position[2])
    assert np.array_equal(w, np.zeros(3) if zero else spin)


@pytest.mark.parametrize("position", [[0, 0, 0.11], [0, 0, 0.109], [0, 0, 0.3], [0, 0, np.nan]])
def test_rejects_invalid_or_nonarray_state(position):
    value = np.array(position) if position != [0, 0, 0.11] else position
    with pytest.raises(ValueError):
        initial_ball_factorial(value, np.zeros(3), ground_ball=True, zero_spin=False)


def test_rejects_implicit_flags_and_unbounded_spin():
    with pytest.raises(ValueError):
        initial_ball_factorial(
            np.array([0.0, 0.0, 0.11]), np.zeros(3), ground_ball=1, zero_spin=False
        )
    with pytest.raises(ValueError):
        initial_ball_factorial(
            np.array([0.0, 0.0, 0.11]),
            np.array([101.0, 0.0, 0.0]),
            ground_ball=False,
            zero_spin=True,
        )
