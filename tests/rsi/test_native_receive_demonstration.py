import numpy as np

from rosclaw_soccer.rsi.native_receive_demonstration import qualify_receive_window


def example():
    return dict(
        qpos=np.zeros((12, 10, 14)),
        qvel=np.zeros((12, 10, 12)),
        forces=np.zeros((12, 10, 3)),
        valid=np.ones(12, dtype=bool),
        terminated=np.zeros(12, dtype=bool),
        ball_q=0,
        root_q=7,
        ball_v=0,
        root_v=6,
    )


def test_clean_control_must_be_native_continuous_after_real_foot_contact():
    data = example()
    data["forces"][0, 0, 0] = 11
    result = qualify_receive_window(**data)
    assert result["qualified"] and result["longest_controlled_native_samples"] == 119
    data["qvel"][5, 0, 0] = 0.6
    assert not qualify_receive_window(**data)["qualified"]


def test_any_nonfoot_or_censored_horizon_rejects_demonstration():
    data = example()
    data["forces"][0, 0, 0] = 11
    data["forces"][11, 9, 2] = 1.01
    assert qualify_receive_window(**data)["reason"] == "nonfoot_contact"
    data["forces"][11, 9, 2] = 0
    data["valid"][-1] = False
    assert qualify_receive_window(**data)["reason"] == "incomplete_or_body_guard"
