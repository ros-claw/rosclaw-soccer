import pytest

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo


@pytest.mark.parametrize(
    "options",
    [
        dict(gamma=1, trace_decay=1),
        dict(binary_terminal_outcome=1, gamma=1, trace_decay=1),
        dict(binary_terminal_outcome=True, gamma=0.995, trace_decay=1),
        dict(binary_terminal_outcome=True, gamma=1, trace_decay=0.999),
        dict(binary_terminal_outcome=True, gamma=True, trace_decay=1),
    ],
)
def test_undiscounted_mode_requires_explicit_complete_contract(options):
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(**options)


def setup():
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.moving_contact_learning import build_moving_contact_actor_critic

    torch.manual_seed(37)
    agent = build_moving_contact_actor_critic()
    with torch.no_grad():
        agent.critic[-1].bias.fill_(0.25)
    obs = torch.zeros(4, 2, 136)
    alive = torch.ones(4, 2)
    alive[2:, 0] = 0
    following = alive.clone()
    following[1, 0] = 0
    following[-1] = 0
    with torch.no_grad():
        mean, value = agent(obs.reshape(-1, 136))
        distribution = torch.distributions.Normal(mean, agent.logstd.exp())
        raw = distribution.sample().reshape(4, 2, 29)
        logp = distribution.log_prob(raw.reshape(-1, 29)).sum(-1).reshape(4, 2)
    data = dict(
        obs=obs,
        raw=raw,
        logp=logp,
        value=value.reshape(4, 2),
        reward=torch.zeros(4, 2),
        alive=alive,
        next_alive=following,
    )
    for value in data.values():
        value[2:, 0] = 0
    data["reward"][1, 0] = 1
    config = FullBodyPPOUpdateConfig(
        epochs=1,
        minibatch_size=32,
        observation_size=136,
        action_size=29,
        gamma=1,
        trace_decay=1,
        binary_terminal_outcome=True,
        critic_all_active=True,
    )
    return torch, agent, data, config


def test_complete_episode_monte_carlo_critic_targets(monkeypatch):
    torch, agent, data, config = setup()
    captured = []
    original = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        captured.append(float(agent.critic[-1].bias.grad))
        return original(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0)
    result = update_full_body_ppo(agent, optimizer, data, config)
    # Two live successful samples, four live failed samples; V=.25 everywhere.
    assert captured == pytest.approx([0.25 - 2 / 6], abs=1e-7)
    assert result["binary_terminal_outcome"] and result["on_policy_inputs_verified"]


@pytest.mark.parametrize("kind", ["truncated", "dense", "fractional", "negative", "dead"])
def test_reject_non_outcome_data_before_optimizer(kind):
    torch, agent, data, config = setup()
    if kind == "truncated":
        data["next_alive"][-1, 1] = 1
    elif kind == "dense":
        data["reward"][0, 1] = 1
    elif kind == "fractional":
        data["reward"][1, 0] = 0.5
    elif kind == "negative":
        data["reward"][1, 0] = -1
    else:
        data["obs"][3, 0, 0] = 1
    before = {key: value.clone() for key, value in agent.state_dict().items()}
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="binary terminal outcome"):
        update_full_body_ppo(agent, optimizer, data, config)
    assert not optimizer.state
    assert all(torch.equal(value, agent.state_dict()[key]) for key, value in before.items())
