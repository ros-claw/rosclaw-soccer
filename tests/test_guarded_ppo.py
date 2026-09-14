import copy

import pytest

import rosclaw_soccer.training.guarded_ppo as guarded
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig


def setup(lr=0.1, count=2):
    torch = pytest.importorskip("torch")

    class Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(133, 29)
            self.critic = torch.nn.Linear(133, 1)
            self.logstd = torch.nn.Parameter(torch.full((29,), -0.3))
            self.register_buffer("counter", torch.zeros(1))
            with torch.no_grad():
                for layer in (self.actor, self.critic):
                    layer.weight.zero_()
                    layer.bias.zero_()

        def forward(self, obs):
            return self.actor(obs), self.critic(obs).squeeze(-1)

    model = Agent()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    data = dict(obs=torch.zeros(1, count, 133), alive=torch.ones(1, count))
    return torch, model, optimizer, data


def test_overshoot_restores_momentum_rng_and_backtracks_same_trial(monkeypatch):
    torch, model, optimizer, data = setup()
    draws = []

    def step(agent, opt, *_):
        draws.append(float(torch.rand(1)))
        opt.zero_grad()
        agent.actor.bias.grad = torch.ones(29)
        opt.step()
        agent.counter.add_(1)
        return dict(optimizer_steps=1, approx_kl=0.0)  # Deliberately untrustworthy summary.

    monkeypatch.setattr(guarded, "update_full_body_ppo", step)
    result = guarded.guarded_full_body_ppo_update(model, optimizer, data, FullBodyPPOUpdateConfig())
    assert result["accepted"] and len(result["attempts"]) == 2
    assert result["optimizer_steps"] == 2 and result["applied_optimizer_steps"] == 1
    assert not result["attempts"][0]["accepted"]
    assert draws[0] == draws[1]
    torch.testing.assert_close(model.actor.bias, torch.full((29,), -0.01))
    assert float(model.counter) == 1 and optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert (
        result["trust_region_mean_kl"] <= 0.01 and result["trust_region_maximum_state_kl"] <= 0.05
    )


@pytest.mark.parametrize("raise_error", [False, True])
def test_rejection_or_exception_restores_complete_numeric_state(monkeypatch, raise_error):
    torch, model, optimizer, data = setup()
    model.actor.bias.grad = torch.full((29,), 3.0)
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()

    def step(agent, opt, *_):
        torch.rand(7)
        agent.train()
        opt.zero_grad()
        agent.actor.bias.grad = torch.ones(29)
        opt.step()
        agent.counter.add_(1)
        if raise_error:
            raise RuntimeError("failed trial")
        return dict(optimizer_steps=1)

    monkeypatch.setattr(guarded, "update_full_body_ppo", step)
    region = guarded.PPOTrustRegion(
        maximum_mean_kl=1e-12, maximum_state_kl=1e-12, maximum_attempts=1
    )
    if raise_error:
        with pytest.raises(RuntimeError, match="failed trial"):
            guarded.guarded_full_body_ppo_update(
                model, optimizer, data, FullBodyPPOUpdateConfig(), trust_region=region
            )
    else:
        result = guarded.guarded_full_body_ppo_update(
            model, optimizer, data, FullBodyPPOUpdateConfig(), trust_region=region
        )
        assert (
            result["status"] == "REJECTED_AND_RESTORED" and result["applied_optimizer_steps"] == 0
        )
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert optimizer.state_dict() == opt_state
    assert torch.equal(rng, torch.get_rng_state()) and not model.training
    assert torch.equal(model.actor.bias.grad, torch.full((29,), 3.0))
    assert model.actor.weight.grad is None


def test_rare_state_cannot_hide_behind_small_mean_kl(monkeypatch):
    torch, model, optimizer, data = setup(lr=0.5, count=100)
    data["obs"][0, 0, 0] = 1

    def step(agent, opt, *_):
        opt.zero_grad()
        agent.actor.weight.grad = torch.zeros_like(agent.actor.weight)
        agent.actor.weight.grad[0, 0] = 1
        opt.step()
        return dict(optimizer_steps=1)

    monkeypatch.setattr(guarded, "update_full_body_ppo", step)
    result = guarded.guarded_full_body_ppo_update(
        model,
        optimizer,
        data,
        FullBodyPPOUpdateConfig(),
        trust_region=guarded.PPOTrustRegion(maximum_attempts=1),
    )
    trial = result["attempts"][0]
    assert trial["mean_kl"] < 0.01 and trial["maximum_state_kl"] > 0.05
    assert not result["accepted"] and torch.count_nonzero(model.actor.weight) == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(maximum_attempts=0),
        dict(maximum_attempts=9),
        dict(maximum_attempts=True),
        dict(backtrack_factor=1),
        dict(maximum_mean_kl=float("nan")),
        dict(maximum_state_kl=0.001),
    ],
)
def test_invalid_region(options):
    with pytest.raises(ValueError):
        guarded.PPOTrustRegion(**options)


def test_foreign_optimizer_and_false_config_rejected():
    torch, model, optimizer, data = setup()
    foreign = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="distinct learner"):
        guarded.guarded_full_body_ppo_update(
            model, torch.optim.SGD([foreign], lr=0.1), data, FullBodyPPOUpdateConfig()
        )
    with pytest.raises(ValueError, match="configs"):
        guarded.guarded_full_body_ppo_update(
            model, optimizer, data, FullBodyPPOUpdateConfig(), trust_region=False
        )


def test_actual_ppo_update_remains_on_policy():
    torch, model, optimizer, data = setup(lr=1e-4, count=8)
    torch.set_num_threads(1)
    with torch.no_grad():
        mean, value = model(data["obs"].reshape(8, 133))
        distribution = torch.distributions.Normal(mean, model.logstd.exp())
        raw = distribution.sample()
        logp = distribution.log_prob(raw).sum(-1)
    data.update(
        raw=raw.reshape(1, 8, 29),
        value=value.reshape(1, 8),
        logp=logp.reshape(1, 8),
        reward=torch.linspace(0, 1, 8).reshape(1, 8),
        next_alive=torch.zeros(1, 8),
    )
    result = guarded.guarded_full_body_ppo_update(
        model, optimizer, data, FullBodyPPOUpdateConfig(epochs=1, minibatch_size=8)
    )
    assert result["accepted"] and result["on_policy_inputs_verified"]
    assert result["applied_optimizer_steps"] == 1
