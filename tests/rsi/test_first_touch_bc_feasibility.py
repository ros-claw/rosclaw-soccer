import numpy as np
import pytest

from scripts.rsi_first_touch_bc_feasibility import causal_examples, kick_phase_examples


def test_examples_use_previous_native_state_and_previous_applied_target():
    q = np.arange(30 * 43, dtype=float).reshape(30, 43)
    v = np.arange(30 * 41, dtype=float).reshape(30, 41)
    actions = np.arange(3 * 29, dtype=float).reshape(3, 29)
    x, y = causal_examples(dict(qpos=q, qvel=v, joint_targets=actions))
    assert x.shape == (2, 104)
    np.testing.assert_array_equal(x[0, :34], q[9, 9:43])
    np.testing.assert_array_equal(x[0, 34:69], v[9, 6:41])
    np.testing.assert_array_equal(x[0, 69:72], q[9, :3] - q[9, 7:10])
    np.testing.assert_array_equal(x[0, 72:75], v[9, :3] - v[9, 6:9])
    np.testing.assert_array_equal(x[0, 75:], actions[0])
    np.testing.assert_array_equal(y[0], actions[1])
    # Changing the post-action state of control 1 must not change its input.
    q[19] += 1000
    altered, _ = causal_examples(dict(qpos=q, qvel=v, joint_targets=actions))
    np.testing.assert_array_equal(x[0], altered[0])


def test_examples_reject_missing_or_nonfinite_data():
    raw = dict(qpos=np.zeros((20, 43)), qvel=np.zeros((20, 41)), joint_targets=np.zeros((2, 29)))
    raw["qvel"][9, 0] = np.nan
    with pytest.raises(ValueError):
        causal_examples(raw)
    raw["qvel"] = np.zeros((10, 41))
    with pytest.raises(ValueError):
        causal_examples(raw)


def test_kick_phase_clock_is_causal_and_excludes_velocity():
    raw = dict(
        qpos=np.zeros((50, 43)),
        qvel=np.zeros((50, 41)),
        joint_targets=np.arange(145, dtype=float).reshape(5, 29),
        velocity_control_active=np.array([True, True, False, False, False]),
    )
    x, y = kick_phase_examples(raw)
    assert x.shape == (3, 105)
    np.testing.assert_array_equal(x[:, -1], [0.0, 0.02, 0.04])
    np.testing.assert_array_equal(x[:, -30:-1], raw["joint_targets"][1:4])
    np.testing.assert_array_equal(y, raw["joint_targets"][2:])
    raw["velocity_control_active"][4] = True
    with pytest.raises(ValueError):
        kick_phase_examples(raw)
