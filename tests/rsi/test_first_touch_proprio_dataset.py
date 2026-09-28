"""Proprioceptive datasets align physics before stepping and preserve episode grain."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_proprio_dataset import (
    FEATURE_NAMES,
    aligned_episode_features,
)


def _episode_inputs() -> dict:
    return {
        "ball_position_after_step": np.asarray(
            ((1.9, 0.0, 0.11), (1.8, 0.0, 0.11), (1.7, 0.0, 0.11))
        ),
        "initial_ball_xyz_m": (2.0, 0.0, 0.13),
        "initial_ball_vx_m_s": -0.5,
        "root_pose_xyzw_m": np.zeros((3, 7)),
        "root_velocity_world": np.zeros((3, 6)),
        "joint_position_rad": np.zeros((3, 29)),
        "joint_velocity_rad_s": np.zeros((3, 29)),
        "navigation_speed_mps": np.full(3, 1.4),
    }


def test_ball_and_proprioception_align_before_step() -> None:
    inputs = _episode_inputs()
    inputs["joint_position_rad"][1, 0] = 0.25
    feature = aligned_episode_features(**inputs)
    assert feature.shape == (3, len(FEATURE_NAMES))
    assert feature[0, 0] == pytest.approx(2.0)
    assert feature[1, 0] == pytest.approx(1.9)
    assert feature[2, 0] == pytest.approx(1.8)
    assert feature[0, 3] == pytest.approx(-0.5)
    assert feature[1, 3] == pytest.approx(-5.0)
    assert feature[1, FEATURE_NAMES.index("leg_joint_position_0_rad")] == pytest.approx(0.25)
    assert feature[1, -1] == pytest.approx(1.4)


def test_nonfinite_body_observation_is_rejected() -> None:
    inputs = _episode_inputs()
    inputs["joint_velocity_rad_s"][0, 3] = float("nan")
    with pytest.raises(ValueError, match="invalid aligned"):
        aligned_episode_features(**inputs)
