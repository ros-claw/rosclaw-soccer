import numpy as np
import pytest

from scripts.rsi_audit_current_memory_update import numeric_change


def test_latent_change_is_not_reported_as_actual_physical_motion():
    old = np.zeros((4, 12))
    new = np.full((4, 12), 0.1)
    new[0] = 0
    result = numeric_change(old, new, np.array([0, 1, 1, 1]))
    assert result["states"] == 4
    assert result["exact_guard_zero_states"] == 1
    assert result["protected_raw_mean_exact"] is True
    assert result["raw_latent_delta_absolute_max"] == 0.1
    assert result["pre_slew_target_delta_absolute_max_rad"] == pytest.approx(0.25 * np.tanh(0.1))
    assert result["physically_applied"] is False


def test_protected_shift_is_exposed_not_hidden():
    result = numeric_change(np.zeros((4, 12)), np.ones((4, 12)), np.zeros(4))
    assert result["protected_raw_mean_exact"] is False


@pytest.mark.parametrize(
    "old,new,gate",
    [
        (np.zeros((0, 12)), np.zeros((0, 12)), np.zeros(0)),
        (np.zeros((4, 12)), np.full((4, 12), np.nan), np.ones(4)),
        (np.zeros((4, 12)), np.ones((4, 12)), np.full(4, 1.1)),
        (np.zeros((4, 13)), np.ones((4, 13)), np.ones(4)),
    ],
)
def test_invalid_diagnostic_does_not_become_a_zero_shift(old, new, gate):
    with pytest.raises(ValueError):
        numeric_change(old, new, gate)
