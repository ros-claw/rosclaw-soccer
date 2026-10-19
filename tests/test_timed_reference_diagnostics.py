import numpy as np

from scripts.rsi_verify_timed_reference import tracking_metrics


def test_fixed_windows_and_all29_channels_are_counted():
    actual = np.zeros((300, 43))
    reference = np.zeros((300, 36))
    actual[50:251, 7:19] = 0.1
    actual[:50, 19:22] = 0.2
    actual[251:, 22:36] = 0.3
    result = tracking_metrics(actual, reference, np)
    assert result["source"]["legs_rmse_rad"] == np.sqrt(np.mean(np.full((201, 12), 0.1) ** 2))
    assert result["source"]["arms_rmse_rad"] == 0
    assert result["entry"]["legs_rmse_rad"] == 0
    assert result["terminal_hold"]["waist_rmse_rad"] == 0
    assert len(result["per_joint_rmse_rad"]) == 29
    assert result["root_xyz_is_tracking_authority"] is False
    assert result["physical_skill_qualified"] is False


def test_root_translation_not_misreported_as_joint_tracking():
    actual = np.zeros((300, 43))
    reference = np.zeros((300, 36))
    actual[:, :3] = 100
    assert tracking_metrics(actual, reference, np)["whole"]["all_rmse_rad"] == 0
