import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_onpolicy_episode import (
    build_receiver_episode,
    receiver_terminal_reward,
)


def test_real_outcome_reward_and_temporal_placement():
    assert (
        receiver_terminal_reward(handoff=True, precise=True, kick=True, body_failure=False) == 3.25
    )
    assert (
        receiver_terminal_reward(handoff=False, precise=False, kick=False, body_failure=True) == -2
    )
    with pytest.raises(ValueError):
        receiver_terminal_reward(handoff=False, precise=True, kick=False, body_failure=False)
    n = 3
    trace = dict(
        control=np.arange(n),
        obs=np.zeros((n, 98), np.float32),
        context14=np.zeros((n, 14), np.float32),
        reference_action=np.zeros((n, 29), np.float32),
        action=np.zeros((n, 29), np.float32),
        SAM_executed_action=np.zeros((n, 29), np.float32),
        SAM_logp=np.zeros(n, np.float32),
        SAM_rng_before=np.zeros((n, 32), np.uint8),
        SAM_rng_after=np.ones((n, 32), np.uint8),
        SAM_active=np.ones(n, bool),
    )
    episode = build_receiver_episode(trace, reward=3.25)
    assert episode["inputs"].shape == (n, 112)
    assert episode["rewards"].tolist() == [0, 0, 3.25]
    trace["control"][1] = 2
    with pytest.raises(ValueError):
        build_receiver_episode(trace, reward=0)
