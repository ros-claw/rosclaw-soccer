import copy

import pytest

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.handoff_policy import build_handoff_actor_critic
from rosclaw_soccer.training.reception_navigation import build_reception_navigation_actor_critic


def case(observation_size=136, action_size=3):
    torch = pytest.importorskip("torch")
    torch.manual_seed(525)
    agent = (
        build_handoff_actor_critic(action_size=action_size)
        if observation_size == 143
        else build_reception_navigation_actor_critic()
    )
    with torch.no_grad():
        agent.logstd.fill_(-5.3)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-6)
    obs = torch.randn(4, 2, observation_size) * 0.1
    with torch.no_grad():
        mean, value = agent(obs.flatten(0, 1))
        normal = torch.distributions.Normal(mean, agent.logstd.clamp(-5.5, -0.3).exp())
        raw = normal.sample()
        data = dict(
            obs=obs,
            raw=raw.reshape(4, 2, action_size),
            logp=normal.log_prob(raw).sum(1).reshape(4, 2),
            value=value.reshape(4, 2),
            reward=torch.randn(4, 2),
            alive=torch.ones(4, 2),
            next_alive=torch.ones(4, 2),
        )
    return torch, agent, optimizer, data


@pytest.mark.parametrize("observation_size,action_size", [(136, 3), (143, 3), (143, 29)])
def test_precision_exploration_has_matching_likelihood_and_exact_replay(
    observation_size, action_size
):
    torch, agent, optimizer, data = case(observation_size, action_size)
    original = copy.deepcopy(agent.state_dict())
    rng = torch.get_rng_state().clone()
    config = FullBodyPPOUpdateConfig(
        observation_size=observation_size,
        action_size=action_size,
        minimum_log_std=-5.5,
        epochs=2,
        minibatch_size=4,
    )
    result = update_full_body_ppo(agent, optimizer, data, config)
    repeated = (
        build_handoff_actor_critic(action_size=action_size)
        if observation_size == 143
        else build_reception_navigation_actor_critic()
    )
    repeated.load_state_dict(original)
    repeated_optimizer = torch.optim.Adam(repeated.parameters(), lr=1e-6)
    torch.set_rng_state(rng)
    assert update_full_body_ppo(repeated, repeated_optimizer, data, config) == result
    assert all(torch.equal(v, repeated.state_dict()[k]) for k, v in agent.state_dict().items())
    assert result["minimum_log_std"] == -5.5 and result["optimizer_steps"] > 0
    assert result["activation_ceiling"] == "SIM_ONLY" and not result["promotion_eligible"]


@pytest.mark.parametrize("observation_size,action_size", [(136, 3), (143, 3), (143, 29)])
def test_default_floor_rejects_fine_noise_rollout_before_any_mutation(
    observation_size, action_size
):
    torch, agent, optimizer, data = case(observation_size, action_size)
    original = copy.deepcopy(agent.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="likelihood"):
        update_full_body_ppo(
            agent,
            optimizer,
            data,
            FullBodyPPOUpdateConfig(observation_size=observation_size, action_size=action_size),
        )
    assert all(torch.equal(v, original[k]) for k, v in agent.state_dict().items())
    assert not optimizer.state and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("floor", [True, "-5", float("nan"), float("inf"), -6.1, -2.4])
def test_exploration_floor_is_explicit_and_bounded(floor):
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(minimum_log_std=floor)


def test_default_floor_is_legacy():
    assert FullBodyPPOUpdateConfig().minimum_log_std == -2.5
