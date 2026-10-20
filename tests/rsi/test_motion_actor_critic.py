import pytest

from rosclaw_soccer.rsi.motion_actor_critic import create_motion_actor_critic
from rosclaw_soccer.rsi.motion_ppo_math import clipped_ppo_loss, full_horizon_gae


def test_deterministic_wrapper_exactly_preserves_supervised_policy_and_rng():
    import torch

    m = create_motion_actor_critic()
    m.eval()
    m.base.eval()
    m.base.initialize_training_mean(torch.linspace(-0.5, 0.5, 64))
    x = torch.randn(1, 20, 108)
    gen = torch.Generator().manual_seed(42)
    before = gen.get_state().clone()
    with torch.no_grad():
        expected, value, h = m.base(x)
        actual = m.sample(x, None, generator=gen, stochastic=False)
    assert (
        torch.equal(expected, actual["token"])
        and torch.equal(value, actual["value"])
        and torch.equal(h, actual["hidden"])
    )
    assert torch.equal(before, gen.get_state())


def test_sampling_noise_is_reproducible_and_not_a_stored_action_bypass():
    import torch

    m = create_motion_actor_critic()
    m.eval()
    x = torch.zeros(1, 1, 108)
    a = m.sample(x, None, generator=torch.Generator().manual_seed(9), stochastic=True)
    b = m.sample(x, None, generator=torch.Generator().manual_seed(9), stochastic=True)
    for key in ("noise", "raw", "token", "log_prob", "value"):
        assert torch.equal(a[key], b[key])
    assert torch.equal(a["token"] * 16, torch.round(a["token"] * 16))
    assert not torch.equal(a["noise"], torch.zeros_like(a["noise"]))


@pytest.mark.parametrize("quantized", [False, True])
def test_gae_and_ppo_produce_actor_critic_gradients_without_updating_frozen_memory(quantized):
    import numpy as np
    import torch

    r = np.zeros((2, 300), dtype=np.float32)
    r[:, 299] = 5
    advantage, returns = full_horizon_gae(r, np.zeros_like(r))
    np.testing.assert_allclose(advantage[:, -1], 5)
    np.testing.assert_allclose(advantage[:, -2], 5 * 0.99 * 0.95)
    assert np.count_nonzero(advantage[:, :20]) == 0
    m = create_motion_actor_critic()
    ref = create_motion_actor_critic()
    ref.load_state_dict(m.state_dict())
    ref.requires_grad_(False)
    x = torch.randn(2, 300, 108)
    with torch.no_grad():
        s = m.sample(x, None, generator=torch.Generator().manual_seed(4), stochastic=True)
    loss, stats = clipped_ppo_loss(
        m,
        ref,
        x,
        s["raw"],
        s["log_prob"],
        s["value"],
        torch.from_numpy(advantage),
        torch.from_numpy(returns),
        quantized_likelihood=quantized,
    )
    loss.backward()
    assert all(p.grad is None for p in m.base.memory.parameters())
    assert all(p.grad is None for p in m.base.encoder.parameters())
    for p in (m.base.actor.weight, m.base.critic.weight, m.log_std):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert abs(float(stats["prior_kl"].detach())) < 1e-6


def test_nonfinite_observation_rejected():
    import torch

    m = create_motion_actor_critic()
    with pytest.raises(ValueError):
        m.parameters_for_state(torch.full((1, 1, 108), float("nan")))
