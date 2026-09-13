import copy

import pytest

from rosclaw_soccer.training.self_imitation_update import update_self_imitation

torch = pytest.importorskip("torch")


class Learner(torch.nn.Module):
    def __init__(self, observation_size=169):
        super().__init__()
        self.actor = torch.nn.Linear(observation_size, 32)
        self.critic = torch.nn.Linear(observation_size, 1)
        self.logstd = torch.nn.Parameter(torch.full((32,), -3.5))
        for head in (self.actor, self.critic):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


def setup():
    model = Learner()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = dict(
        obs=torch.ones(8, 169),
        raw=torch.full((8, 32), 0.1),
        returns=torch.ones(8),
        actor_mask=torch.ones(8),
    )
    return model, optimizer, batch


def test_positive_replay_learns_actor_and_critic():
    model, optimizer, batch = setup()
    result = update_self_imitation(model, optimizer, batch)
    mean, value = model(batch["obs"])
    assert (mean > 0).all() and (value > 0).all()
    assert result["positive_actor_samples"] == 8
    assert result["optimizer_steps"] == 1
    assert not result["on_policy"] and not result["promotion_eligible"]


def test_negative_replay_does_not_advance_existing_adam_momentum():
    model, optimizer, batch = setup()
    update_self_imitation(model, optimizer, batch)
    before = copy.deepcopy(model.state_dict())
    steps = [state["step"].clone() for state in optimizer.state.values()]
    batch["returns"].fill_(-100)
    assert update_self_imitation(model, optimizer, batch)["optimizer_steps"] == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(
        torch.equal(step, state["step"])
        for step, state in zip(steps, optimizer.state.values(), strict=True)
    )


def test_masked_actor_not_updated_even_with_existing_momentum():
    model, optimizer, batch = setup()
    update_self_imitation(model, optimizer, batch)
    before = copy.deepcopy(model.actor.state_dict())
    std = model.logstd.detach().clone()
    batch["actor_mask"].zero_()
    result = update_self_imitation(model, optimizer, batch)
    assert result["positive_actor_samples"] == 0
    assert result["positive_value_samples"] == 8
    assert all(torch.equal(before[k], v) for k, v in model.actor.state_dict().items())
    assert torch.equal(std, model.logstd)


@pytest.mark.parametrize("fault", ["nan", "mask", "grad", "extra", "shape"])
def test_invalid_batch_rejected_without_weight_changes(fault):
    model, optimizer, batch = setup()
    before = copy.deepcopy(model.state_dict())
    if fault == "nan":
        batch["raw"][0, 0] = float("nan")
    elif fault == "mask":
        batch["actor_mask"][0] = 0.5
    elif fault == "grad":
        batch["returns"].requires_grad_()
    elif fault == "extra":
        batch["logp"] = torch.zeros(8)
    else:
        batch["obs"] = torch.zeros(8, 170)
    with pytest.raises(ValueError):
        update_self_imitation(model, optimizer, batch)
    assert not optimizer.state
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_exact_replay_and_exclusive_optimizer():
    first, opt, batch = setup()
    second, other, _ = setup()
    for _ in range(3):
        assert update_self_imitation(first, opt, batch) == update_self_imitation(
            second, other, batch
        )
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    opt.add_param_group({"params": [torch.nn.Parameter(torch.zeros(1))]})
    with pytest.raises(ValueError, match="exclusively"):
        update_self_imitation(first, opt, batch)


@pytest.mark.parametrize("size", [True, 169.0, 0, 170, 183, 513])
def test_unknown_observation_contract_is_rejected_before_optimizer(size):
    model, optimizer, batch = setup()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="observation contract"):
        update_self_imitation(model, optimizer, batch, observation_size=size)
    assert not optimizer.state
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_explicit_legacy_contract_preserves_exact_optimizer_math():
    first, opt, batch = setup()
    second, other, _ = setup()
    for _ in range(3):
        assert update_self_imitation(first, opt, batch) == update_self_imitation(
            second, other, batch, observation_size=169
        )
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    for a, b in zip(opt.state.values(), other.state.values(), strict=True):
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_contextual_contract_requires_opt_in_and_replays_actor_critic_and_adam():
    first = Learner(182)
    second = copy.deepcopy(first)
    opt = torch.optim.Adam(first.parameters(), lr=1e-4)
    other = torch.optim.Adam(second.parameters(), lr=1e-4)
    batch = dict(
        obs=torch.ones(8, 182),
        raw=torch.full((8, 32), 0.1),
        returns=torch.ones(8),
        actor_mask=torch.ones(8),
    )
    with pytest.raises(ValueError, match="aligned"):
        update_self_imitation(first, opt, batch)
    assert not opt.state
    for _ in range(3):
        result = update_self_imitation(first, opt, batch, observation_size=182)
        assert result == update_self_imitation(second, other, batch, observation_size=182)
        assert not result["on_policy"] and not result["promotion_eligible"]
    mean, value = first(batch["obs"])
    assert (mean > 0).all() and (value > 0).all()
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    for a, b in zip(opt.state.values(), other.state.values(), strict=True):
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_real_full_body_rehearsal_preserves_frozen_parent_and_normalization():
    from rosclaw_soccer.training.coupled_ball_residual import (
        build_coupled_ball_residual_actor_critic,
    )
    from rosclaw_soccer.training.task_space_contact_actor import build_full_body_carry_actor_critic

    parent = build_coupled_ball_residual_actor_critic()
    zero, one = torch.zeros(169), torch.ones(169)
    model = build_full_body_carry_actor_critic(
        parent.state_dict(), zero, one, critic_mean=zero, critic_scale=one
    )
    obs = torch.zeros(8, 182)
    obs[:, 136] = 1
    obs[:, 138] = 0.2
    obs[4:, 169] = 1
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    with torch.no_grad():
        mean, value = model(obs)
    batch = dict(obs=obs, raw=mean + 0.1, returns=value + 1, actor_mask=torch.ones(8))
    result = update_self_imitation(model, optimizer, batch, observation_size=182)
    assert result["positive_actor_samples"] == 8 and result["optimizer_steps"] == 1
    after = model.state_dict()
    protected = [k for k in before if k.startswith("parent.") or k.endswith((".mean", ".scale"))]
    assert protected and all(torch.equal(before[k], after[k]) for k in protected)
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("actor."))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("critic."))
    assert not any(p.requires_grad or p.grad is not None for p in model.parent.parameters())
