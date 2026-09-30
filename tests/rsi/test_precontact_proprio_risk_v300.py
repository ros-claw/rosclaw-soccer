"""Precontact risk features are finite and strictly before first collision."""

import numpy as np
import pytest

from scripts.rsi_train_proprio_risk_v300 import FEATURE_NAMES, precontact_features


def _archive(folder) -> None:
    root = np.zeros((31, 1, 7))
    velocity = np.zeros((31, 1, 6))
    ball = np.zeros((31, 1, 3))
    ball[:, 0, 0] = 2.0
    ball[:, 0, 2] = 0.11
    ball_velocity = np.zeros((31, 1, 3))
    ball_velocity[:, 0, 0] = -0.6
    geometry = np.zeros((31, 1, 4, 3))
    geometry[:, 0, :, 2] = 0.2
    np.savez_compressed(
        folder / "body_trace.npz",
        root_pose_xyzw_m=root,
        root_velocity_world=velocity,
        ball_position_before_step_m=ball,
        ball_linear_velocity_before_step_m_s=ball_velocity,
        foot_geometry_position_before_step_m=geometry,
    )


def test_precontact_feature_contract(tmp_path) -> None:
    _archive(tmp_path)
    report = {"environments": [{"first_contact_frame": 31}]}
    features = precontact_features(tmp_path, report)
    assert len(features) == len(FEATURE_NAMES) == 13
    assert features[0] == 2.0
    assert features[2] == -0.6


def test_frame30_at_contact_rejected(tmp_path) -> None:
    _archive(tmp_path)
    with pytest.raises(ValueError, match="physical contact"):
        precontact_features(tmp_path, {"environments": [{"first_contact_frame": 30}]})
