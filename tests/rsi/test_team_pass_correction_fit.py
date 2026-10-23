import numpy as np
import pytest

from scripts.rsi_diagnose_team_pass_correction_fit import weighted_motor_mse


def test_motor_error_is_radians_squared_with_measured_contact_weights():
    data = dict(
        targets=np.zeros((2, 29), np.float32),
        action_scale=np.ones(29, np.float32) * 0.5,
        loss_weights=np.array([1.0, 3.0], np.float32),
    )
    mean = np.stack([np.ones(29, np.float32), np.ones(29, np.float32) * 2])
    assert weighted_motor_mse(mean, data) == (0.25 + 3 * 1.0) / 4
    assert weighted_motor_mse(data["targets"], data) == 0


@pytest.mark.parametrize("fault", ["negative_weight", "nan", "partial", "scale"])
def test_invalid_same_history_comparison_rejected(fault):
    data = dict(
        targets=np.zeros((2, 29), np.float32),
        action_scale=np.ones(29, np.float32),
        loss_weights=np.ones(2, np.float32),
    )
    mean = data["targets"].copy()
    if fault == "negative_weight":
        data["loss_weights"][0] = -1
    elif fault == "nan":
        mean[0, 0] = np.nan
    elif fault == "partial":
        mean = mean[:1]
    else:
        data["action_scale"][0] = 0
    with pytest.raises(ValueError):
        weighted_motor_mse(mean, data)
