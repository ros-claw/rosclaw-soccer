import copy

import pytest

from rosclaw_soccer.training.binary_learning_window import attach_binary_learning_window
from rosclaw_soccer.training.coupled_ball_residual import build_coupled_ball_residual_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.task_space_contact_actor import (
    build_task_space_carry_actor_critic,
    build_task_space_contact_actor_critic,
)

torch = pytest.importorskip("torch")


def build(actions=3):
    parent = build_coupled_ball_residual_actor_critic()
    zero, one = torch.zeros(169), torch.ones(169)
    factory = (
        build_task_space_contact_actor_critic
        if actions == 3
        else build_task_space_carry_actor_critic
    )
    learner = factory(parent.state_dict(), zero, one, critic_mean=zero, critic_scale=one)
    obs = torch.zeros(4, 170)
    obs[:, 136] = 1
    obs[:, 138] = 0.48
    obs[2:, 169] = 1
    return learner, obs


@pytest.mark.parametrize("actions", [3, 6])
def test_initial_force_zero_parent_and_normalization_frozen(actions):
    learner, obs = build(actions)
    mean, value = learner(obs)
    _, expected = learner.parent(obs[:, :169])
    assert torch.equal(mean, torch.zeros(4, actions))
    assert torch.equal(value, expected)
    assert not any(p.requires_grad for p in learner.parent.parameters())
    assert learner.actor.mean.shape == (170,)
    assert learner.actor.mean[-1] == learner.actor.scale[-1] == 0.5


@pytest.mark.parametrize("fault", ["missing_foot", "extra_bit", "foot", "nan", "history"])
def test_no_implicit_state_or_foot_contract(fault):
    learner, obs = build()
    if fault == "missing_foot":
        obs = obs[:, :169]
    elif fault == "extra_bit":
        obs = torch.cat((obs, torch.ones(4, 1)), 1)
    elif fault == "foot":
        obs[:, 169] = 0.5
    elif fault == "nan":
        obs[:, 0] = float("nan")
    else:
        obs[:, 145] = 1  # Velocity cannot be available while validity is zero.
    with pytest.raises(ValueError):
        learner(obs)


@pytest.mark.parametrize("actions", [3, 6])
def test_explicit_171_ppo_replays_without_changing_parent(actions):
    learner, obs = build(actions)
    agent = attach_binary_learning_window(learner, observation_size=170)
    starting = copy.deepcopy(agent.state_dict())
    observation = torch.cat((obs, torch.ones(4, 1)), 1).repeat(4, 1, 1)
    observation[2:, :, -1] = 0
    with torch.no_grad():
        mean, value = agent(observation.reshape(-1, 171))
        normal = torch.distributions.Normal(mean, agent.logstd.exp())
        raw = normal.sample()
        data = dict(
            obs=observation,
            raw=raw.reshape(4, 4, actions),
            logp=normal.log_prob(raw).sum(1).reshape(4, 4),
            value=value.reshape(4, 4),
            reward=torch.arange(16).reshape(4, 4).float() / 16,
            alive=torch.ones(4, 4),
            next_alive=torch.ones(4, 4),
        )
        data["next_alive"][-1] = 0
    config = FullBodyPPOUpdateConfig(
        epochs=2,
        minibatch_size=4,
        observation_size=171,
        action_size=actions,
        minimum_log_std=-4,
        learning_observation_index=170,
        critic_all_active=True,
    )
    outcomes = []
    for _ in range(2):
        agent.load_state_dict(starting)
        optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=1e-4)
        torch.manual_seed(1234)
        report = update_full_body_ppo(agent, optimizer, data, config)
        after = copy.deepcopy(agent.state_dict())
        for key, item in starting.items():
            if key.startswith("learner.parent.") or key.endswith((".mean", ".scale")):
                assert torch.equal(item, after[key])
        assert any(
            not torch.equal(item, after[key])
            for key, item in starting.items()
            if key.startswith("learner.actor.network.")
        )
        outcomes.append((report, after))
    assert outcomes[0][0] == outcomes[1][0]
    for key in starting:
        assert torch.equal(outcomes[0][1][key], outcomes[1][1][key])


def test_force_only_weights_do_not_silently_load_as_force_and_navigation():
    contact, _ = build(3)
    carry, _ = build(6)
    with pytest.raises(RuntimeError):
        carry.load_state_dict(contact.state_dict())
