"""Optional trainer dependency tests; run explicitly in the qualified RSL env."""

import pytest
import torch

pytest.importorskip("rsl_rl")

from scripts.rsi_foot_only_value_warmup import FootOnlyValueWarmupPPO  # noqa: E402
from scripts.rsi_mjlab_prior_rl import PPO, RehearsalPPO  # noqa: E402


def staged():
    obj = FootOnlyValueWarmupPPO.__new__(FootOnlyValueWarmupPPO)
    obj.actor = torch.nn.Linear(2, 1)
    obj.critic = torch.nn.Linear(2, 1)
    obj.optimizer = torch.optim.Adam(list(obj.actor.parameters()) + list(obj.critic.parameters()))
    obj.schedule = "adaptive"
    obj.value_warmup_iterations = 16
    obj.value_warmup_completed = 0
    return obj


def test_real_adam_does_not_mutate_actor_or_use_stale_gradients(monkeypatch):
    obj = staged()
    before = {k: v.clone() for k, v in obj.actor.state_dict().items()}
    critic = {k: v.clone() for k, v in obj.critic.state_dict().items()}
    for p in obj.actor.parameters():
        p.grad = torch.ones_like(p)

    def update(self):
        assert self.schedule == "fixed"
        assert all(not p.requires_grad and p.grad is None for p in self.actor.parameters())
        x = torch.ones(4, 2)
        loss = (self.actor(x).square() + self.critic(x).square()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {}

    monkeypatch.setattr(PPO, "update", update)
    obj.update()
    assert all(torch.equal(v, before[k]) for k, v in obj.actor.state_dict().items())
    assert any(not torch.equal(v, critic[k]) for k, v in obj.critic.state_dict().items())
    assert obj.value_warmup_completed == 1
    assert obj.schedule == "adaptive"
    assert all(p.requires_grad for p in obj.actor.parameters())
    assert all(p not in obj.optimizer.state for p in obj.actor.parameters())


def test_exception_restores_flags_without_counting_success(monkeypatch):
    obj = staged()
    next(obj.actor.parameters()).requires_grad_(False)
    flags = [p.requires_grad for p in obj.actor.parameters()]

    def fail(self):
        raise RuntimeError("injected failed value update")

    monkeypatch.setattr(PPO, "update", fail)
    with pytest.raises(RuntimeError, match="injected"):
        obj.update()
    assert obj.value_warmup_completed == 0
    assert obj.schedule == "adaptive"
    assert [p.requires_grad for p in obj.actor.parameters()] == flags


def test_after_boundary_returns_to_rehearsal_ppo(monkeypatch):
    obj = staged()
    obj.value_warmup_completed = 16
    monkeypatch.setattr(RehearsalPPO, "update", lambda self: {"ppo": 1})
    assert obj.update() == {"ppo": 1}
    assert obj.value_warmup_completed == 16


def test_unqualified_resume_is_explicitly_rejected():
    with pytest.raises(ValueError, match="resume is not qualified"):
        staged().load({})
