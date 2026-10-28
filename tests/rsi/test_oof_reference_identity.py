import numpy as np
import pytest

from rosclaw_soccer.rsi.oof_reference_identity import exact_reference_arrays


def arrays():
    return {
        k: np.zeros((2,), dtype=np.float64)
        for k in (
            "qpos",
            "qvel",
            "torque",
            "ball_forces",
            "executed_receiver_action",
            "passer_action",
            "receiver_observation",
            "receiver_command",
            "receiver_return_action",
            "receiver_return_observation",
            "receiver_return_target",
            "passer_follow_action",
            "passer_follow_command",
            "passer_follow_control",
            "passer_follow_detector_delta",
            "passer_follow_observation",
            "passer_follow_target",
        )
    }


def test_exact_requires_all_actor_history_and_identical_physics():
    a, b = arrays(), arrays()
    assert exact_reference_arrays(a, b)
    b["qvel"][0] = 1
    assert not exact_reference_arrays(a, b)
    b = arrays()
    b["receiver_observation"][0] = np.nan
    assert not exact_reference_arrays(a, b)
    del b["passer_action"]
    with pytest.raises(ValueError):
        exact_reference_arrays(a, b)
