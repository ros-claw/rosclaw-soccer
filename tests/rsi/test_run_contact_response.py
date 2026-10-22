import numpy as np
import pytest

from rosclaw_soccer.rsi.run_contact_response import bounded_proposals, fit_response


def test_recovers_physical_response_direction_not_a_policy():
    rng = np.random.default_rng(45)
    x = rng.uniform(-0.5, 0.5, (80, 12))
    gradient = np.arange(24).reshape(12, 2) / 24
    y = x @ gradient + np.array([2.0, 4.0])
    report = fit_response(x, y)
    assert np.allclose(report["latent_gradient"], gradient, atol=0.003)
    assert max(report["leave_one_out_rmse"]) < 0.01
    assert report["runtime_motor_policy"] is False


def test_proposals_bounded_and_not_called_successes():
    base = np.full(12, 0.95)
    candidates = np.array(bounded_proposals(base, 2.0, -np.ones(12)))
    assert candidates.shape == (11, 12)
    assert (np.abs(candidates) <= 1).all()
    assert np.linalg.norm(candidates[4] - base) <= 0.5 + 1e-10
    assert np.array_equal(candidates[0], base)


@pytest.mark.parametrize("n", [0, 5, 13])
def test_insufficient_data_rejected(n):
    with pytest.raises(ValueError):
        fit_response(np.zeros((n, 12)), np.zeros((n, 2)))


def test_nonfinite_data_rejected():
    x = np.zeros((20, 12))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_response(x, np.zeros((20, 2)))


def test_signed_left_and_right_errors_have_mirrored_response():
    base = np.zeros(12)
    gradient = np.ones(12)
    left = np.array(bounded_proposals(base, 1.0, gradient))
    right = np.array(bounded_proposals(base, -1.0, gradient))
    assert np.allclose(left[1:5], -right[1:5])
    assert (left[4] @ gradient) < 0
    assert (right[4] @ gradient) > 0
