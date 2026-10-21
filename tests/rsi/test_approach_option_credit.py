import numpy as np
import pytest

from rosclaw_soccer.rsi.approach_option_credit import (
    GAMMA,
    approach_head_penalty,
    approach_option_gae,
)


def args():
    z = np.zeros((95, 2), dtype=np.float32)
    valid = np.ones((95, 2), dtype=bool)
    valid[40:, 1] = False
    handoff = np.zeros_like(valid)
    handoff[94, 0] = True
    terminated = np.zeros_like(valid)
    terminated[39, 1] = True
    tr = np.zeros((60, 2), dtype=np.float32)
    tr[-1, 0] = 12
    tv = np.zeros((60, 2), dtype=bool)
    tv[:, 0] = True
    tt = np.zeros_like(tv)
    tt[-1, 0] = True
    return [z.copy(), z.copy(), z.copy(), terminated, handoff, valid], dict(
        tail_reward=tr, tail_valid=tv, tail_terminal=tt
    )


def test_actual_long_head_tail_credit_and_zero_failure_bootstrap():
    head, tail = args()
    head[2][:] = 9999.0
    advantage, _ = approach_option_gae(*head, **tail)
    assert advantage[94, 0] == pytest.approx(12 * float(GAMMA) ** 60, rel=2e-6)
    assert advantage[39, 1] == 0
    assert not advantage[40:, 1].any()


@pytest.mark.parametrize("mutation", ["alive", "tail_dense", "hole", "double_boundary"])
def test_rejects_reward_hacks_and_nonphysical_boundaries(mutation):
    head, tail = args()
    if mutation == "alive":
        head[0][0, 0] = 0.01
    elif mutation == "tail_dense":
        tail["tail_reward"][0, 0] = 1
    elif mutation == "hole":
        head[-1][20, 0] = False
    else:
        head[3][94, 0] = True
    with pytest.raises(ValueError):
        approach_option_gae(*head, **tail)


def test_imitation_cost_never_rewards_waiting_and_penalizes_actual_failure_once():
    info = dict(
        valid=np.array([True, True, False]),
        tracking=np.ones((3, 5)),
        action_rate=np.zeros(3),
        limits=np.zeros(3),
        terminated=np.array([False, True, False]),
    )
    assert np.array_equal(approach_head_penalty(info), [0, -20, 0])
    info["tracking"][:] = 0
    assert np.array_equal(
        approach_head_penalty(info), np.array([-0.01, -20.01, 0], dtype=np.float32)
    )
