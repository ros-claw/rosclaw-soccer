import pytest

from rosclaw_soccer.training.coupled_ball_residual import (
    advance_coupled_residual,
    build_coupled_ball_residual_actor_critic,
    compose_coupled_navigation,
)
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

torch = pytest.importorskip("torch")


def observations(n):
    x = torch.zeros(n, 139)
    x[:, 136] = 1
    x[:, 138] = 0.32
    return x


def test_new_contract_zero_mean_and_finite_critic():
    agent = build_coupled_ball_residual_actor_critic()
    mean, value = agent(observations(4))
    assert mean.shape == (4, 32) and torch.count_nonzero(mean) == 0
    assert value.shape == (4,) and torch.isfinite(value).all()
    with pytest.raises(ValueError):
        agent(torch.zeros(4, 136))


def test_joint_and_navigation_envelopes_are_independently_bounded():
    raw = torch.full((4, 32), 100.0)
    joint, nav = torch.zeros(4, 29), torch.zeros(4, 3)
    for _ in range(100):
        following_joint, following_nav = advance_coupled_residual(raw, joint, nav)
        assert (following_joint - joint).abs().max() <= 0.025001
        assert torch.linalg.vector_norm(following_nav[:, :2] - nav[:, :2], dim=1).max() <= 0.012001
        assert (following_nav[:, 2] - nav[:, 2]).abs().max() <= 0.040001
        joint, nav = following_joint, following_nav
    assert joint.abs().max() <= 0.250001
    output = compose_coupled_navigation(torch.tensor([[0.7, 0, 0.8]]).repeat(4, 1), nav)
    assert torch.linalg.vector_norm(output[:, :2], dim=1).max() <= 0.700001
    assert output[:, 2].abs().max() <= 0.800001


def test_zero_residual_preserves_nominal_command():
    nominal = torch.tensor([[0.3, -0.4, 0.2]])
    assert torch.equal(compose_coupled_navigation(nominal, torch.zeros(1, 3)), nominal)


@pytest.mark.parametrize(
    "bad_history", [None, torch.zeros(1, 29, dtype=torch.float64), torch.zeros(2, 29)]
)
def test_invalid_joint_history_rejected(bad_history):
    with pytest.raises(ValueError):
        advance_coupled_residual(torch.zeros(1, 32), bad_history, torch.zeros(1, 3))


@pytest.mark.parametrize("index,value", [(136, 0.0), (133, 2.0), (138, 0.1), (0, float("nan"))])
def test_invalid_coupled_observation_rejected(index, value):
    agent = build_coupled_ball_residual_actor_critic()
    x = observations(2)
    x[:, index] = value
    with pytest.raises(ValueError):
        agent(x)


def test_explicit_paired_ppo_dimensions_required():
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(observation_size=139)
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(action_size=32)
    assert FullBodyPPOUpdateConfig(observation_size=139, action_size=32).action_size == 32


def test_real_optimizer_update_accepts_verified_139_by_32_rollout():
    torch.manual_seed(405)
    agent = build_coupled_ball_residual_actor_critic()
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    obs = observations(16)
    with torch.no_grad():
        mean, value = agent(obs)
        distribution = torch.distributions.Normal(mean, agent.logstd.clamp(-2.5, -0.3).exp())
        raw = distribution.sample()
        data = dict(
            obs=obs.reshape(4, 4, 139),
            raw=raw.reshape(4, 4, 32),
            logp=distribution.log_prob(raw).sum(1).reshape(4, 4),
            value=value.reshape(4, 4),
            reward=torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16,
            alive=torch.ones(4, 4),
            next_alive=torch.ones(4, 4),
        )
    result = update_full_body_ppo(
        agent,
        optimizer,
        data,
        FullBodyPPOUpdateConfig(observation_size=139, action_size=32, epochs=1, minibatch_size=8),
    )
    assert result["optimizer_steps"] == 2 and result["on_policy_inputs_verified"]
