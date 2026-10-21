"""Function-preserving initialization and protected far/absent-contact behavior."""

import torch
from torch import nn

from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor


class Base(nn.Module):
    reference_features = 79

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(177, 29)
        self.register_buffer("prior_mean", torch.zeros(1, 98))
        self.register_buffer("prior_denominator", torch.ones(1, 98))

    def forward(self, obs):
        return self.head(obs)


def test_initial_function_exact_everywhere_and_backbone_frozen():
    torch.manual_seed(62)
    base = Base()
    actor = NativeResidualContactActor(base)
    obs = torch.randn(128, 177)
    assert torch.equal(actor(obs), base(obs))
    assert all(not p.requires_grad for p in base.parameters())
    assert all(p.requires_grad for p in actor.adapter.parameters())


def test_modified_adapter_cannot_change_far_or_missing_perception():
    actor = NativeResidualContactActor(Base())
    with torch.no_grad():
        for p in actor.adapter.parameters():
            p.uniform_(-2, 2)
    obs = torch.randn(5, 177)
    obs[:3, 168:170] = torch.tensor([[0.7, 0], [1, 1], [-0.71, 0]])
    obs[3, 168:] = 0
    obs[4, 168:170] = torch.tensor([0.1, 0.1])
    actual, base = actor(obs), actor.base(obs)
    assert torch.equal(actual[:4], base[:4])
    assert not torch.equal(actual[4], base[4])
    assert (actual - base).abs().max() <= 0.200001
    loss = (actual[4] - base[4].detach()).square().sum()
    loss.backward()
    assert all(p.grad is None for p in actor.base.parameters())


def test_gate_is_smooth_monotonic_on_declared_transition():
    actor = NativeResidualContactActor(Base())
    obs = torch.zeros(4, 177)
    obs[:, 168] = torch.tensor([0.1, 0.35, 0.525, 0.7])
    gate = actor.measured_gate(obs)
    assert torch.allclose(gate, torch.tensor([1, 1, 0.5, 0]), atol=1e-6)


def test_real_ppo_gradient_updates_adapter_but_preserves_foundation_and_far_domain():
    from rosclaw_soccer.rsi.native_contact_ppo import make_contact_actor_critic, native_ppo_loss

    torch.manual_seed(7602)
    policy = make_contact_actor_critic(NativeResidualContactActor(Base()))
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=0.001)
    before = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    obs = torch.randn(16, 177)
    obs[:, 168:170] = torch.tensor([0.1, 0.1])
    with torch.no_grad():
        distribution = policy.distribution(obs)
        action = distribution.sample()
        old_logp = distribution.log_prob(action).sum(-1)
        old_value = policy.value(obs)
    advantage = torch.linspace(-1, 1, 16)
    rehearsal = obs.clone()
    rehearsal[:, 168:] = 0
    loss, _ = native_ppo_loss(
        policy, obs, action, old_logp, old_value, advantage, old_value + advantage, rehearsal
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert all(p.grad is None and p not in optimizer.state for p in policy.actor.base.parameters())
    assert all(
        torch.equal(v, before[k])
        for k, v in policy.actor.state_dict().items()
        if k.startswith("base.")
    )
    assert any(
        not torch.equal(v, before[k])
        for k, v in policy.actor.state_dict().items()
        if k.startswith("adapter.")
    )
    with torch.no_grad():
        far = obs.clone()
        far[:, 168:170] = torch.tensor([0.8, 0.0])
        assert torch.equal(policy.actor(far), policy.actor.base(far))
        assert torch.equal(policy.actor(rehearsal), policy.actor.base(rehearsal))
