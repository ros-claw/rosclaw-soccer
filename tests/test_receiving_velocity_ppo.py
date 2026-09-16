"""Explicit velocity-feature schema; optimizer evidence is not physical success."""

import copy

import pytest

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

torch = pytest.importorskip("torch")


class Actor(torch.nn.Module):
    def __init__(self, observation_size=149):
        super().__init__()
        self.actor = torch.nn.Linear(observation_size, 29)
        self.critic = torch.nn.Linear(observation_size, 1)
        self.logstd = torch.nn.Parameter(torch.full((29,), -2.5))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


def case(observation_size=149):
    torch.manual_seed(2424)
    actor = Actor(observation_size)
    obs = torch.randn(4, 2, observation_size) * 0.05
    obs[:, :, 135] = 1
    obs[0, :, 135] = 0
    if observation_size == 150:
        obs[:, :, 149] = obs[:, :, 135]
    with torch.no_grad():
        mean, value = actor(obs.flatten(0, 1))
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


@pytest.mark.parametrize("observation_size", [149, 150])
def test_velocity_schema_update_replays_exactly_with_event_mask(observation_size):
    actor, batch = case(observation_size)
    repeat = copy.deepcopy(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    repeated_optimizer = torch.optim.Adam(repeat.parameters(), lr=1e-5)
    rng = torch.get_rng_state().clone()
    config = FullBodyPPOUpdateConfig(
        observation_size=observation_size,
        learning_observation_index=149 if observation_size == 150 else 135,
        critic_all_active=True,
        epochs=2,
        minibatch_size=4,
    )
    stats = update_full_body_ppo(actor, optimizer, batch, config)
    torch.set_rng_state(rng)
    assert update_full_body_ppo(repeat, repeated_optimizer, batch, config) == stats
    assert all(torch.equal(v, repeat.state_dict()[k]) for k, v in actor.state_dict().items())
    assert stats["optimizer_steps"] > 0
    assert stats["activation_ceiling"] == "SIM_ONLY" and not stats["promotion_eligible"]


@pytest.mark.parametrize("fault", ["old_shape", "nonfinite_velocity", "wrong_logp"])
@pytest.mark.parametrize("observation_size", [149, 150])
def test_velocity_schema_faults_rejected_before_update(fault, observation_size):
    actor, batch = case(observation_size)
    before = copy.deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    if fault == "old_shape":
        batch["obs"] = batch["obs"][:, :, :143]
    elif fault == "nonfinite_velocity":
        batch["obs"][0, 0, 143] = float("nan")
    else:
        batch["logp"][0, 0] += 1
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        update_full_body_ppo(
            actor, optimizer, batch, FullBodyPPOUpdateConfig(observation_size=observation_size)
        )
    assert not optimizer.state and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())


@pytest.mark.parametrize("observation_size", [149, 150])
def test_velocity_schema_does_not_implicitly_enable_other_action_dimensions(observation_size):
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(observation_size=observation_size, action_size=3)
