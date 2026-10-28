import numpy as np
import pytest
from test_g1_velocity_policy import parameters

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.receiver_counterfactual_residual import receiver_counterfactual_residual


def test_zero_is_exact_and_nonzero_is_encoded_as_executed_action():
    params = parameters()
    nominal = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
    original = nominal.copy()
    delta = np.linspace(-0.1, 0.1, 12, dtype=np.float64)
    action, target = receiver_counterfactual_residual(nominal, delta, params, amplitude=0.0)
    np.testing.assert_array_equal(action, original)
    np.testing.assert_array_equal(target, target29(original, params))
    action, target = receiver_counterfactual_residual(nominal, delta, params, amplitude=0.5)
    np.testing.assert_array_equal(nominal, original)
    np.testing.assert_array_equal(action[12:], original[12:])
    np.testing.assert_array_equal(target, target29(action, params))
    np.testing.assert_allclose(
        target[:12] - target29(original, params)[:12], 0.5 * delta, atol=1e-7
    )


@pytest.mark.parametrize("amplitude", [-0.1, 1.1, float("nan")])
def test_invalid_amplitude_is_rejected(amplitude):
    with pytest.raises(ValueError):
        receiver_counterfactual_residual(
            np.zeros(29, np.float32), np.zeros(12, np.float64), parameters(), amplitude=amplitude
        )


def test_nonfinite_or_excessive_leg_correction_is_rejected():
    for value in (float("nan"), 0.151):
        delta = np.zeros(12, np.float64)
        delta[0] = value
        with pytest.raises(ValueError):
            receiver_counterfactual_residual(
                np.zeros(29, np.float32), delta, parameters(), amplitude=1.0
            )
