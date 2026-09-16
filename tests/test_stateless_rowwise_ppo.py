"""Singleton arithmetic is opt-in; probability mismatches still fail closed."""

import copy

import pytest

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.stateless_forward import stateless_actor_critic_rows

torch = pytest.importorskip("torch")


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(150, 128), torch.nn.Tanh(), torch.nn.Linear(128, 29)
        )
        self.critic = torch.nn.Linear(150, 1)
        self.logstd = torch.nn.Parameter(torch.full((29,), -6.0), requires_grad=False)
        self.calls = 0

    def forward(self, obs):
        self.calls += 1
        return self.actor(obs), self.critic(obs).squeeze(-1)


def case():
    torch.manual_seed(2472)
    actor = Actor()
    obs = torch.randn(4, 2, 150) * 0.05
    obs[:, :, 135] = 1
    obs[:, :, 149] = 1
    obs[0, :, 149] = 0
    with torch.no_grad():
        mean, value = stateless_actor_critic_rows(actor, obs.flatten(0, 1))
        distribution = torch.distributions.Normal(mean, actor.logstd.exp())
        raw = distribution.sample()
        batch = dict(
            obs=obs,
            raw=raw.reshape(4, 2, 29),
            logp=distribution.log_prob(raw).sum(1).reshape(4, 2),
            value=value.reshape(4, 2),
            reward=torch.randn(4, 2),
            alive=torch.ones(4, 2),
            next_alive=torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        )
    return actor, batch


def config(**kwargs):
    return FullBodyPPOUpdateConfig(
        observation_size=150,
        minimum_log_std=-6.0,
        learning_observation_index=149,
        critic_all_active=True,
        stateless_rowwise_forward=True,
        epochs=2,
        minibatch_size=4,
        **kwargs,
    )


def test_rows_preserve_singleton_outputs_inputs_weights_and_gradients():
    actor, batch = case()
    other = copy.deepcopy(actor)
    obs = batch["obs"].flatten(0, 1)
    before = obs.clone()
    weights = copy.deepcopy(actor.state_dict())
    mean, value = stateless_actor_critic_rows(actor, obs)
    individual = [other(row[None]) for row in obs]
    expected_mean = torch.cat([item[0] for item in individual])
    expected_value = torch.cat([item[1] for item in individual])
    assert torch.equal(mean, expected_mean) and torch.equal(value, expected_value)
    assert torch.equal(obs, before)
    assert all(torch.equal(v, weights[k]) for k, v in actor.state_dict().items())
    (mean.square().sum() + value.square().sum()).backward()
    (expected_mean.square().sum() + expected_value.square().sum()).backward()
    for a, b in zip(actor.parameters(), other.parameters(), strict=True):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)


def test_low_noise_rowwise_ppo_repeats_exactly_with_full_critic_and_mask():
    actor, batch = case()
    repeat = copy.deepcopy(actor)
    optimizer = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=1e-6)
    repeated_optimizer = torch.optim.Adam(
        (p for p in repeat.parameters() if p.requires_grad), lr=1e-6
    )
    rng = torch.get_rng_state().clone()
    stats = update_full_body_ppo(actor, optimizer, batch, config())
    torch.set_rng_state(rng)
    assert update_full_body_ppo(repeat, repeated_optimizer, batch, config()) == stats
    assert all(torch.equal(v, repeat.state_dict()[k]) for k, v in actor.state_dict().items())
    assert stats["stateless_rowwise_forward"] and stats["on_policy_inputs_verified"]
    assert stats["optimizer_steps"] > 0 and not stats["promotion_eligible"]


@pytest.mark.parametrize("fault", ["logp", "value"])
def test_rowwise_mode_does_not_relax_starting_policy_validation(fault):
    actor, batch = case()
    before = copy.deepcopy(actor.state_dict())
    batch[fault][0, 0] += 0.01
    optimizer = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=1e-6)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="starting learner"):
        update_full_body_ppo(actor, optimizer, batch, config())
    assert not optimizer.state and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_rowwise_mode_requires_explicit_boolean(value):
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(stateless_rowwise_forward=value)


@pytest.mark.parametrize("kind", ["dropout", "batchnorm", "gru", "attention"])
def test_known_stochastic_or_coupled_layers_rejected_before_forward(kind):
    actor = Actor()
    actor.coupled = {
        "dropout": torch.nn.Dropout(0.1),
        "batchnorm": torch.nn.BatchNorm1d(150),
        "gru": torch.nn.GRU(4, 4),
        "attention": torch.nn.MultiheadAttention(4, 1),
    }[kind]
    with pytest.raises(ValueError, match="stateless"):
        stateless_actor_critic_rows(actor, torch.zeros(2, 150))
    assert actor.calls == 0


@pytest.mark.parametrize("fault", ["empty", "nonfinite", "double"])
def test_invalid_rows_rejected_before_forward(fault):
    actor = Actor()
    obs = torch.zeros(2, 150)
    if fault == "empty":
        obs = obs[:0]
    elif fault == "nonfinite":
        obs[0, 0] = float("nan")
    else:
        obs = obs.double()
    with pytest.raises(ValueError):
        stateless_actor_critic_rows(actor, obs)
    assert actor.calls == 0


def test_default_mode_keeps_historical_result_schema():
    actor, batch = case()
    with torch.no_grad():
        mean, value = actor(batch["obs"].flatten(0, 1))
        distribution = torch.distributions.Normal(mean, actor.logstd.exp())
        batch["logp"] = distribution.log_prob(batch["raw"].flatten(0, 1)).sum(1).reshape(4, 2)
        batch["value"] = value.reshape(4, 2)
    result = update_full_body_ppo(
        actor,
        torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=1e-6),
        batch,
        FullBodyPPOUpdateConfig(observation_size=150, minimum_log_std=-6.0, epochs=1),
    )
    assert "stateless_rowwise_forward" not in result
