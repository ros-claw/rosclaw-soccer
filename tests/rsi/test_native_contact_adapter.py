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
