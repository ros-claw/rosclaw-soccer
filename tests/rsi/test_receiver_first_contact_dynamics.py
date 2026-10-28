import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_first_contact_dynamics import first_contact_dynamics


def sample(n=30):
    return (
        np.zeros((n, 10, 79), dtype=np.float64),
        np.zeros((n, 10, 76), dtype=np.float64),
        np.zeros((n, 10, 2, 3), dtype=np.float64),
    )


def run(q, v, f):
    return first_contact_dynamics(q, v, f, ball_q=0, ball_v=0, receiver_q=43, receiver_v=41)


def test_first_contact_simultaneous_and_actual_half_second():
    q, v, f = sample()
    f[1, 2, 1, 0] = 5
    f[1, 2, 1, 2] = 6
    v[1, 1, 0] = 2
    v[1, 2, 0] = 1
    v[26, 2, 0] = 0.5
    q[26, 2, 0] = 0.2
    result = run(q, v, f)
    assert result["first_contact_substep"] == 12
    assert result["first_contact_groups"] == ["left_foot", "nonfoot"]
    assert result["precontact_relative_speed_mps"] == 2
    assert result["postcontact_relative_speed_mps"] == 1
    assert result["half_second_observed"] is True
    assert result["half_second_relative_speed_mps"] == 0.5
    assert result["half_second_ball_displacement_m"] == 0.2


def test_censoring_and_absent_contact_are_explicit():
    q, v, f = sample(2)
    assert run(q, v, f)["first_contact_substep"] is None
    f[0, 0, 1, 1] = 2
    result = run(q, v, f)
    assert result["precontact_observed"] is False
    assert result["precontact_relative_speed_mps"] is None
    assert result["half_second_observed"] is False
    assert result["half_second_relative_speed_mps"] is None


def test_nonfinite_or_overlapping_layout_rejected():
    q, v, f = sample()
    v[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        run(q, v, f)
