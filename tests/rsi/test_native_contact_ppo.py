"""Timeouts bootstrap; true failures and invalid tails do not leak advantages."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_contact_ppo import course_gae


def test_timeout_bootstrap_distinct_from_true_failure_and_invalid_tail():
    reward = np.array([[1, 1], [0, 1]], dtype=np.float32)
    value = np.full((2, 2), 0.5, dtype=np.float32)
    following = np.array([[0.5, 0.5], [999, 9]], dtype=np.float32)
    terminated = np.array([[True, False], [False, False]])
    truncated = np.array([[False, False], [False, True]])
    valid = np.array([[True, True], [False, True]])
    adv, returns = course_gae(reward, value, following, terminated, truncated, valid)
    assert adv[0, 0] == 0.5
    assert returns[0, 0] == 1
    assert adv[1, 0] == returns[1, 0] == 0
    assert adv[1, 1] == pytest.approx(1 + 0.99 * 9 - 0.5)
    assert adv[0, 1] == pytest.approx(1 + 0.99 * 0.5 - 0.5 + 0.99 * 0.95 * adv[1, 1])


def test_consecutive_timeouts_stop_gae_recursion():
    r = np.ones((2, 1), dtype=np.float32)
    v = np.zeros_like(r)
    mask = np.ones_like(r, dtype=bool)
    adv, _ = course_gae(r, v, v + 2, ~mask, mask, mask)
    assert np.allclose(adv, 1 + 0.99 * 2)
    with pytest.raises(ValueError, match="exactly one valid"):
        course_gae(r, v, v, mask, mask, mask)


def test_40_control_course_requires_explicit_opt_in():
    reward = np.ones((40, 1), dtype=np.float32)
    value = np.zeros_like(reward)
    valid = np.ones_like(reward, dtype=bool)
    terminal = np.zeros_like(valid)
    timeout = terminal.copy()
    timeout[-1] = True
    with pytest.raises(ValueError, match="bounded"):
        course_gae(reward, value, value, terminal, timeout, valid)
    advantage, returns = course_gae(
        reward, value, value, terminal, timeout, valid, maximum_controls=40
    )
    assert advantage[-1, 0] == 1
    assert advantage[0, 0] > advantage[-1, 0]
    assert np.array_equal(advantage, returns)


def test_60_control_course_requires_explicit_opt_in():
    reward = np.ones((60, 1), dtype=np.float32)
    value = np.zeros_like(reward)
    valid = np.ones_like(reward, dtype=bool)
    terminal = np.zeros_like(valid)
    timeout = terminal.copy()
    timeout[-1] = True
    with pytest.raises(ValueError, match="bounded"):
        course_gae(reward, value, value, terminal, timeout, valid, maximum_controls=40)
    advantage, _ = course_gae(reward, value, value, terminal, timeout, valid, maximum_controls=60)
    assert advantage[-1, 0] == 1
    assert np.isfinite(advantage).all()


def test_post_update_kl_guard_rolls_back_network_and_adam():
    import torch
    from torch import nn

    from rosclaw_soccer.rsi.native_contact_ppo import (
        guarded_ppo_optimizer_step,
        make_contact_actor_critic,
    )

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("prior_mean", torch.zeros(1, 98))
            self.register_buffer("prior_denominator", torch.ones(1, 98))
            self.head = nn.Linear(177, 29)

        def forward(self, obs):
            return self.head(obs)

    torch.manual_seed(1)
    policy = make_contact_actor_critic(Actor())
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=0.01)
    obs = torch.zeros(8, 177)
    obs[:, 0] = 1
    with torch.no_grad():
        dist = policy.distribution(obs)
        actions = dist.loc.clone()
        old_logp = dist.log_prob(actions).sum(-1)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    (-policy.actor(obs).sum()).backward()
    accepted, kl = guarded_ppo_optimizer_step(policy, optimizer, obs, actions, old_logp)
    assert not accepted and kl > 0.02
    assert not optimizer.state
    assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items())
    optimizer.param_groups[0]["lr"] = 1e-5
    accepted, kl = guarded_ppo_optimizer_step(policy, optimizer, obs, actions, old_logp)
    assert accepted and kl < 0.02
    assert optimizer.state
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items())


def test_value_warmup_cannot_touch_actor_or_exploration():
    import torch
    from torch import nn

    from rosclaw_soccer.rsi.native_contact_ppo import make_contact_actor_critic, native_ppo_loss

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("prior_mean", torch.zeros(1, 98))
            self.register_buffer("prior_denominator", torch.ones(1, 98))
            self.head = nn.Linear(177, 29)

        def forward(self, obs):
            return self.head(obs)

    torch.manual_seed(29)
    policy = make_contact_actor_critic(Actor())
    optimizer = torch.optim.Adam(
        [*policy.actor.parameters(), *policy.critic.parameters(), policy.log_std], lr=0.001
    )
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    obs = torch.randn(8, 177)
    zeros = torch.zeros(8)
    loss, _ = native_ppo_loss(
        policy, obs, torch.zeros(8, 29), zeros, zeros, zeros, zeros + 2, obs, value_only=True
    )
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(v, before[k])
        for k, v in policy.state_dict().items()
        if k.startswith("critic.")
    )
    assert all(
        torch.equal(v, before[k])
        for k, v in policy.state_dict().items()
        if not k.startswith("critic.")
    )
    assert not any(p in optimizer.state for p in policy.actor.parameters())
    assert policy.log_std not in optimizer.state
