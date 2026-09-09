import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.directed_ball_residual import (
    build_directed_ball_residual_actor_critic,
    initialize_directed_from_capture,
)

torch = pytest.importorskip("torch")


def test_directed_ppo_requires_explicit_observation_contract():
    from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

    agent = build_directed_ball_residual_actor_critic()
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    obs = torch.cat((torch.zeros(2, 133), torch.tensor([[1.0, 0.0, 0.5]]).repeat(2, 1)), 1)
    with torch.no_grad():
        mean, value = agent(obs)
        distribution = torch.distributions.Normal(mean, agent.logstd.clamp(-2.5, -0.3).exp())
        raw = distribution.sample()
        logp = distribution.log_prob(raw).sum(1)
    rollout = dict(
        obs=obs[:, None],
        raw=raw[:, None],
        logp=logp[:, None],
        value=value[:, None],
        reward=torch.tensor([[0.0], [1.0]]),
        alive=torch.ones(2, 1),
        next_alive=torch.ones(2, 1),
    )
    with pytest.raises(ValueError, match="matching shapes"):
        update_full_body_ppo(agent, optimizer, rollout)
    result = update_full_body_ppo(
        agent, optimizer, rollout, FullBodyPPOUpdateConfig(observation_size=136, epochs=1)
    )
    assert result["on_policy_inputs_verified"]
    assert result["optimizer_steps"] == 1
    for size in (True, 134, 137, 100000):
        with pytest.raises(ValueError):
            FullBodyPPOUpdateConfig(observation_size=size)


def test_migration_preserves_source_and_initial_outputs():
    capture = build_ball_residual_actor_critic()
    with torch.no_grad():
        capture.actor[-1].weight.normal_(0, 0.01)
    source = {k: v.clone() for k, v in capture.state_dict().items()}
    directed = build_directed_ball_residual_actor_critic()
    initialize_directed_from_capture(directed, capture)
    x = torch.randn(8, 133).clamp(-10, 10)
    goal = torch.tensor([1.0, 0.0, 0.5]).repeat(8, 1)
    a, v = capture(x)
    b, w = directed(torch.cat((x, goal), 1))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(v, w, atol=1e-6, rtol=1e-6)
    assert all(torch.equal(value, capture.state_dict()[k]) for k, value in source.items())
    assert torch.count_nonzero(directed.actor[0].weight[:, -3:]) == 0


@pytest.mark.parametrize(
    "goal",
    [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 1.1], [float("nan"), 0.0, 0.5]],
)
def test_invalid_goal_rejected(goal):
    model = build_directed_ball_residual_actor_critic()
    with pytest.raises(ValueError):
        model(torch.cat((torch.zeros(1, 133), torch.tensor([goal])), 1))


def test_migration_validates_before_mutating_destination():
    source = build_ball_residual_actor_critic()
    destination = build_directed_ball_residual_actor_critic()
    original = {k: v.clone() for k, v in destination.state_dict().items()}
    with torch.no_grad():
        source.logstd.fill_(float("nan"))
    with pytest.raises(ValueError):
        initialize_directed_from_capture(destination, source)
    assert all(torch.equal(v, destination.state_dict()[k]) for k, v in original.items())


def test_goal_inputs_can_learn_without_mutating_capture():
    capture = build_ball_residual_actor_critic()
    with torch.no_grad():
        capture.actor[-1].weight.fill_(0.01)
    original = {k: v.clone() for k, v in capture.state_dict().items()}
    directed = build_directed_ball_residual_actor_critic()
    initialize_directed_from_capture(directed, capture)
    optimizer = torch.optim.Adam(directed.parameters(), lr=1e-4)
    observation = torch.cat((torch.zeros(1, 133), torch.tensor([[1.0, 0.0, 0.5]])), 1)
    mean, _ = directed(observation)
    loss = (mean - 0.1).square().mean()
    loss.backward()
    assert directed.actor[0].weight.grad[:, -3:].abs().sum() > 0
    optimizer.step()
    assert torch.count_nonzero(directed.actor[0].weight[:, -3:]) > 0
    assert all(torch.equal(v, capture.state_dict()[k]) for k, v in original.items())
