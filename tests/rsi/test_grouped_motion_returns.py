import numpy as np
import pytest

from rosclaw_soccer.rsi.grouped_motion_returns import leave_one_out_returns


def test_own_reward_does_not_enter_own_baseline():
    r = np.zeros((2, 4, 300), dtype=np.float32)
    r[0, :, 299] = [1, 2, 3, 4]
    a, meta = leave_one_out_returns(r)
    np.testing.assert_allclose(a[0, :, 299], [-2, -2 / 3, 2 / 3, 2], atol=1e-7)
    r[0, 0, 299] += 3
    b, _ = leave_one_out_returns(r)
    assert b[0, 0, 299] - a[0, 0, 299] == pytest.approx(3)
    np.testing.assert_allclose(b[0, 1:, 299] - a[0, 1:, 299], -1, atol=1e-7)
    assert meta["independent_contexts"] == 2 and meta["episode_executions"] == 8
    assert meta["zero_variation_contexts"] == 1
    assert not a[:, :, :20].any()


def test_group_permutation_and_common_reward_offset_invariance():
    rng = np.random.default_rng(31)
    r = rng.normal(0, 0.1, (3, 4, 300)).astype(np.float32)
    a, _ = leave_one_out_returns(r)
    perm = [2, 0, 3, 1]
    b, _ = leave_one_out_returns(r[:, perm])
    np.testing.assert_array_equal(b, a[:, perm])
    c, _ = leave_one_out_returns(r + 0.1)
    np.testing.assert_allclose(c, a, atol=2e-6)


@pytest.mark.parametrize("shape", [(4, 300), (1, 1, 300), (0, 4, 300), (1, 4, 299)])
def test_incomplete_or_ungrouped_data_rejected(shape):
    with pytest.raises(ValueError):
        leave_one_out_returns(np.zeros(shape, dtype=np.float32))
