import numpy as np
import pytest

from scripts.rsi_replay_retargeted_cftrack_partial import validate_raw


def test_partial_failed_trajectory_is_allowed_but_must_be_complete():
    n = 7
    raw = {
        "qpos": np.zeros((n * 10, 43)),
        "qvel": np.zeros((n * 10, 41)),
        "torque": np.zeros((n * 10, 29)),
        "ball_forces": np.zeros((n * 10, 1, 3)),
        "joint_targets": np.zeros((n, 29)),
    }
    validate_raw(raw, n)
    raw["qvel"] = raw["qvel"][:-1]
    with pytest.raises(ValueError, match="qvel"):
        validate_raw(raw, n)
