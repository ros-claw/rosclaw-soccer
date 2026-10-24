import numpy as np
import pytest

from rosclaw_soccer.rsi.native_role_residual_data import causal_gait_memory


def test_current_command_and_command_masked_phase_are_never_critic_features():
    obs = np.arange(8 * 98, dtype=np.float32).reshape(8, 98)
    phase = np.float32(0.2)
    before = causal_gait_memory(obs, phase)
    obs[:, 6:11] = -10
    np.testing.assert_array_equal(before, causal_gait_memory(obs, phase))
    np.testing.assert_array_equal(before[:, :29], obs[:, 69:98])
    assert before.shape == (8, 31)


def test_causal_memory_does_not_alias_or_mutate_caller():
    obs = np.zeros((8, 98), np.float32)
    memory = causal_gait_memory(obs, np.float32(0.1))
    memory[:, 0] = 10
    assert not obs.any()


def test_nonfinite_memory_is_rejected_not_sanitized():
    obs = np.zeros((8, 98), np.float32)
    obs[0, 69] = np.nan
    with pytest.raises(ValueError):
        causal_gait_memory(obs, np.float32(0.1))
