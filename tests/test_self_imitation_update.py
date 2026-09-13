import copy

import pytest

from rosclaw_soccer.training.self_imitation_update import update_self_imitation

torch = pytest.importorskip("torch")


class Learner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(169, 32)
        self.critic = torch.nn.Linear(169, 1)
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
