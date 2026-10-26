import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_action_history import describe_previous_target_encoding


def test_inverse_codec_respects_nontrivial_joint_permutation_and_float32():
    lab2mj = np.roll(np.arange(29), 4)
    mj2lab = np.argsort(lab2mj)
    default = np.linspace(-0.3, 0.3, 29, dtype=np.float32)
    scale = np.linspace(0.1, 0.4, 29, dtype=np.float32)
    action = np.linspace(-1, 1, 29, dtype=np.float32)
    target = (action * scale + default)[lab2mj]
    result = describe_previous_target_encoding(target, default, scale, mj2lab, lab2mj)
    np.testing.assert_allclose(result["implied_action_lab"], action, atol=2e-7)
    np.testing.assert_allclose(result["reconstructed_target_hg"], target, atol=3e-8)
    assert not result["prior_policy_action_proven"]
    assert not result["historical_observation_installed"]
    assert not result["physical_state_modified"]


def test_invalid_permutations_scales_and_nonfinite_targets_fail_closed():
    target, default, scale = np.zeros(29), np.zeros(29), np.ones(29)
    order = np.arange(29)
    bad = order.copy()
    bad[0] = bad[1]
    with pytest.raises(ValueError):
        describe_previous_target_encoding(target, default, scale, order, bad)
    scale[4] = 0
    with pytest.raises(ValueError):
        describe_previous_target_encoding(target, default, scale, order, order)
    scale[4] = 1
    target[4] = np.nan
    with pytest.raises(ValueError):
        describe_previous_target_encoding(target, default, scale, order, order)
