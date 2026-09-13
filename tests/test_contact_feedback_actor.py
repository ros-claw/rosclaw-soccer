import copy

import pytest

from rosclaw_soccer.training.contact_feedback_actor import (
    build_contact_feedback_actor_critic,
    contact_features,
)
from rosclaw_soccer.training.coupled_ball_residual import build_coupled_ball_residual_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

torch = pytest.importorskip("torch")


def inputs():
    return dict(
        foot_position=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        foot_velocity=torch.zeros(1, 2, 3),
        ball_position=torch.tensor([[1.0, 0.0, 0.0]]),
        ball_velocity=torch.tensor([[5.0, 0.0, 0.0]]),
        launch_direction=torch.tensor([[1.0, 0.0]]),
        stance_force=torch.tensor([[300.0, 600.0]]),
        interval_valid=torch.ones(1, 1),
    )


def model_and_obs():
    parent = build_coupled_ball_residual_actor_critic()
    agent = build_contact_feedback_actor_critic(parent.state_dict())
    observation = torch.zeros(6, 169)
    observation[:, 136] = 1
    observation[:, 138] = 0.48
    return parent, agent, observation


def test_launch_frame_features_and_reset_validity():
    data = inputs()
    actual = contact_features(**data)
    torch.testing.assert_close(
        actual,
        torch.tensor(
            [[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
        ),
    )
    data["interval_valid"].zero_()
    actual = contact_features(**data)
    assert torch.count_nonzero(actual[:, 6:]) == 0
    assert torch.count_nonzero(actual[:, :6]) > 0


def test_rigid_rotation_and_translation_invariance():
    data = inputs()
    expected = contact_features(**data)
    for name in ("foot_position", "foot_velocity", "ball_position", "ball_velocity"):
        value = data[name].clone()
        data[name][..., 0] = -value[..., 1]
        data[name][..., 1] = value[..., 0]
        if name.endswith("position"):
            data[name] += torch.tensor([4.0, 7.0, 2.0])
    data["launch_direction"] = torch.tensor([[0.0, 1.0]])
    torch.testing.assert_close(contact_features(**data), expected)


@pytest.mark.parametrize("name", list(inputs()))
def test_nonfinite_measurements_rejected(name):
    data = inputs()
    data[name].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        contact_features(**data)


def test_parent_initial_mean_value_and_storage_preserved():
    parent, agent, observation = model_and_obs()
    before = copy.deepcopy(parent.state_dict())
    actual = agent(observation)
    expected = parent(observation[:, :139])
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not any(p.requires_grad for p in agent.parent.parameters())
    with torch.no_grad():
        agent.parent.actor[-1].bias.add_(1)
    for key, value in parent.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize("size", [139, 140, 168, 170])
def test_no_implicit_legacy_or_metadata_contract(size):
    _, agent, _ = model_and_obs()
    with pytest.raises(ValueError):
        agent(torch.zeros(2, size))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.int64, torch.bool])
def test_parent_dtype_is_not_silently_converted(dtype):
    state = build_coupled_ball_residual_actor_critic().state_dict()
    key = next(iter(state))
    state[key] = state[key].to(dtype)
    with pytest.raises(ValueError, match="float32 parent"):
        build_contact_feedback_actor_critic(state)


@pytest.mark.parametrize("state", [None, {}, {"weight": [1.0]}])
def test_parent_requires_nonempty_tensor_mapping(state):
    with pytest.raises(ValueError, match="float32 parent"):
        build_contact_feedback_actor_critic(state)


def test_actor_residual_envelope_and_nonfinite_parent_rejection():
    parent, agent, observation = model_and_obs()
    with torch.no_grad():
        agent.actor[-1].bias.fill_(1000)
    delta = agent(observation)[0] - parent(observation[:, :139])[0]
    assert bool((delta.abs() <= 0.300001).all())
    state = parent.state_dict()
    state["actor.0.weight"][0, 0] = float("inf")
    with pytest.raises(ValueError):
        build_contact_feedback_actor_critic(state)


def test_real_ppo_updates_actor_critic_but_not_parent():
    torch.manual_seed(106001)
    _, agent, observation = model_and_obs()
    observation[:, 139:151] = torch.randn(6, 12) * 0.1
    observation[:, 153] = 1
    observation[:, 168] = 1
    protected = copy.deepcopy(agent.parent.state_dict())
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = copy.deepcopy(agent.critic.state_dict())
    with torch.no_grad():
        mean, value = agent(observation)
        normal = torch.distributions.Normal(mean, agent.logstd.clamp(-4, -0.3).exp())
        raw = normal.sample()
        rollout = dict(
            obs=observation.reshape(3, 2, 169),
            raw=raw.reshape(3, 2, 32),
            logp=normal.log_prob(raw).sum(1).reshape(3, 2),
            value=value.reshape(3, 2),
            reward=torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]]),
            alive=torch.ones(3, 2),
            next_alive=torch.ones(3, 2),
        )
    config = FullBodyPPOUpdateConfig(
        epochs=1, minibatch_size=6, observation_size=169, action_size=32, minimum_log_std=-4
    )
    optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=1e-4)
    result = update_full_body_ppo(agent, optimizer, rollout, config)
    assert result["optimizer_steps"] > 0
    assert any(
        not torch.equal(value, actor_before[key]) for key, value in agent.actor.state_dict().items()
    )
    assert any(
        not torch.equal(value, critic_before[key])
        for key, value in agent.critic.state_dict().items()
    )
    assert all(
        torch.equal(value, protected[key]) for key, value in agent.parent.state_dict().items()
    )
    assert all(p.grad is None for p in agent.parent.parameters())


def test_stance_and_validity_are_not_arbitrary_floats():
    _, agent, observation = model_and_obs()
    for index, value in ((151, -1), (152, 2), (153, 0.5), (168, 0.5)):
        invalid = observation.clone()
        invalid[0, index] = value
        with pytest.raises(ValueError):
            agent(invalid)


def test_unavailable_interval_and_overflow_inputs_rejected():
    _, agent, observation = model_and_obs()
    observation[0, 145] = 1
    with pytest.raises(ValueError, match="unavailable"):
        agent(observation)
    data = inputs()
    data["foot_position"].fill_(3e38)
    with pytest.raises(ValueError):
        contact_features(**data)
