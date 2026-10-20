import numpy as np
import pytest

from rosclaw_soccer.rsi.motion_actor_critic import create_motion_actor_critic
from scripts.rsi_motion_ppo_rollout import SamplingPolicy


def test_owned_sampling_stream_warms_prefix_without_advancing_rng():
    import torch

    torch.set_num_threads(1)
    model = create_motion_actor_critic().eval()
    adapter = SamplingPolicy(model, seed=9, stochastic=True)
    old = adapter.generator.get_state().clone()
    hidden = None
    x = torch.zeros(1, 1, 108)
    for _ in range(20):
        _, _, hidden = adapter(x, hidden)
    assert torch.equal(old, adapter.generator.get_state())
    for _ in range(280):
        _, _, hidden = adapter(x, hidden)
    trace = adapter.recording()
    assert trace["noise"].shape == trace["raw"].shape == (300, 64)
    assert trace["log_prob"].shape == (300,)
    assert np.count_nonzero(trace["noise"][:20]) == 0
    assert np.count_nonzero(trace["noise"][20:]) > 0
    for row in trace.values():
        assert row.dtype == np.float32 and np.isfinite(row).all()
    with pytest.raises(ValueError):
        adapter(x, hidden)


def test_empty_gae_batch_rejected():
    from rosclaw_soccer.rsi.motion_ppo_math import full_horizon_gae

    x = np.empty((0, 300), dtype=np.float32)
    with pytest.raises(ValueError):
        full_horizon_gae(x, x)
