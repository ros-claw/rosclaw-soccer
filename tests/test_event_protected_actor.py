import copy

import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.event_protected_actor import build_event_protected_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.goal_reference_actor import build_goal_reference_actor_critic

torch = pytest.importorskip("torch")


def case():
    torch.manual_seed(90101)
    seed = build_goal_reference_actor_critic(build_ball_residual_actor_critic().state_dict())
    with torch.no_grad():
        seed.actor[-1].weight.normal_(0, 0.03)
        seed.reference_actor[-1].weight.normal_(0, 0.03)
    model = build_event_protected_actor_critic(seed, observation_size=136, action_size=30)
    obs = torch.randn(8, 137) * 0.1
    obs[:, -1] = 0
    obs[4:, -1] = 1
    return seed, model, obs


def test_initial_both_branches_preserve_seed_means_and_value_exactly():
    seed, model, obs = case()
    expected = seed(obs[:, :-1].contiguous())
    actual = model(obs)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert all(not p.requires_grad for p in model.anchor.parameters())
    assert not model.logstd.requires_grad and not model.plastic.logstd.requires_grad
    assert all(
        p.data_ptr() != q.data_ptr()
        for p, q in zip(seed.parameters(), model.anchor.parameters(), strict=True)
    )


def test_real_ppo_update_changes_successor_but_not_protected_means_or_distribution():
    seed, model, obs = case()
    initial = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        mean, value = model(obs)
        normal = torch.distributions.Normal(mean, model.logstd.clamp(-2.5, -0.3).exp())
        raw = normal.sample()
        data = dict(
            obs=obs.reshape(4, 2, 137),
            raw=raw.reshape(4, 2, 30),
            value=value.reshape(4, 2),
            logp=normal.log_prob(raw).sum(1).reshape(4, 2),
            reward=torch.randn(4, 2),
            alive=torch.ones(4, 2),
            next_alive=torch.ones(4, 2),
        )
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    config = FullBodyPPOUpdateConfig(
        observation_size=137, action_size=30, epochs=2, minibatch_size=4
    )
    rng = torch.get_rng_state()
    result = update_full_body_ppo(model, optimizer, data, config)
    updated, _ = model(obs)
    assert result["on_policy_inputs_verified"]
    assert torch.equal(updated[:4], mean[:4])
    assert not torch.equal(updated[4:], mean[4:])
    assert all(
        torch.equal(v, model.state_dict()[k])
        for k, v in initial.items()
        if k.startswith("anchor.") or k in {"logstd", "plastic.logstd"}
    )
    replay = build_event_protected_actor_critic(seed, observation_size=136, action_size=30)
    replay.load_state_dict(initial, strict=True)
    opt = torch.optim.Adam((p for p in replay.parameters() if p.requires_grad), lr=1e-3)
    torch.set_rng_state(rng)
    assert update_full_body_ppo(replay, opt, data, config) == result
    assert all(torch.equal(v, replay.state_dict()[k]) for k, v in model.state_dict().items())


def test_protected_actor_has_no_action_gradient_even_when_critic_learns():
    _, model, obs = case()
    obs[:, -1] = 0
    mean, value = model(obs)
    (mean.square().sum() + value.square().sum()).backward()
    assert all(p.grad is None for p in model.anchor.parameters())
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.plastic.actor.parameters()
    )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.plastic.critic.parameters()
    )


def test_train_keeps_anchor_in_evaluation_mode():
    _, model, _ = case()
    model.train()
    assert model.training and model.plastic.training and not model.anchor.training


def test_non_robot_seed_and_parent_mutation_are_isolated():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(5, 2)
            self.critic = torch.nn.Linear(5, 1)
            self.logstd = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x):
            return self.actor(x), self.critic(x).squeeze(-1)

    seed = Toy()
    model = build_event_protected_actor_critic(seed, observation_size=5, action_size=2)
    obs = torch.zeros(3, 6)
    before = model(obs)[0].detach().clone()
    with torch.no_grad():
        seed.actor.bias.add_(100)
        model.plastic.actor.bias.add_(1)
    assert torch.equal(model(obs)[0], before)
    obs[:, -1] = 1
    assert torch.equal(model(obs)[0], before + 1)


@pytest.mark.parametrize("flag", [-1, 0.5, 2, float("nan"), float("inf")])
def test_invalid_event_feature_rejected(flag):
    _, model, obs = case()
    obs[0, -1] = flag
    with pytest.raises(ValueError):
        model(obs)


@pytest.mark.parametrize("shape", [(0, 137), (1, 136), (137,)])
def test_invalid_shape_rejected(shape):
    _, model, _ = case()
    with pytest.raises(ValueError):
        model(torch.zeros(shape))


@pytest.mark.parametrize("size", [True, 0, 1025, 136.0])
def test_invalid_dimensions_rejected(size):
    seed, _, _ = case()
    with pytest.raises(ValueError):
        build_event_protected_actor_critic(seed, observation_size=size, action_size=30)


def test_invalid_seed_and_input_rejected():
    seed, model, obs = case()
    with pytest.raises(ValueError):
        model(obs.double())
    obs[0, 0] = 11
    with pytest.raises(ValueError):
        model(obs)
    with torch.no_grad():
        seed.logstd[0] = float("nan")
    with pytest.raises(ValueError):
        build_event_protected_actor_critic(seed, observation_size=136, action_size=30)
