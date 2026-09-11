"""Entry adaptation must not silently rewrite the frozen motor seed."""

import copy

import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.entry_conditioned_actor import build_entry_conditioned_actor_critic
from rosclaw_soccer.training.goal_reference_actor import build_goal_reference_actor_critic

torch = pytest.importorskip("torch")


def make_pair():
    torch.manual_seed(73201)
    parent = build_goal_reference_actor_critic(build_ball_residual_actor_critic().state_dict())
    with torch.no_grad():
        parent.reference_actor[-1].weight.normal_(0, 0.1)
    child = build_entry_conditioned_actor_critic(
        parent.state_dict(), torch.zeros(6), torch.full((6,), 0.01)
    )
    return parent, child


def test_zero_adapter_exact_parent_and_trainable_context():
    parent, child = make_pair()
    obs = torch.randn(32, 142)
    expected = parent(obs[:, :136].contiguous())
    actual = child(obs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert torch.equal(child.logstd, parent.logstd)
    (actual[0].sum() + actual[1].sum()).backward()
    assert all(p.grad is None and not p.requires_grad for p in child.parent.parameters())
    assert child.entry_actor[-1].weight.grad.abs().sum() > 0
    assert child.entry_critic[-1].weight.grad.abs().sum() > 0


def test_calibration_copied_and_action_correction_bounded():
    parent, child = make_pair()
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    with torch.no_grad():
        child.entry_actor[-1].bias.fill_(100)
    obs = torch.randn(3, 142)
    expected, _ = parent(obs[:, :136].contiguous())
    actual, _ = child(obs)
    assert torch.allclose(actual - expected, torch.full_like(actual, 0.5), atol=1e-7)
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("bad", [0.0, -1.0, 11.0, float("nan"), float("inf")])
def test_invalid_scale_at_creation_and_after_load(bad):
    parent, child = make_pair()
    scale = torch.ones(6)
    scale[0] = bad
    with pytest.raises(ValueError):
        build_entry_conditioned_actor_critic(parent.state_dict(), torch.zeros(6), scale)
    state = child.state_dict()
    state["entry_scale"] = scale
    child.load_state_dict(state)
    with pytest.raises(ValueError):
        child(torch.zeros(1, 142))


@pytest.mark.parametrize("shape", [(1, 136), (0, 142), (142,), (1, 143)])
def test_invalid_input_shape(shape):
    _, child = make_pair()
    with pytest.raises(ValueError):
        child(torch.zeros(shape))


@pytest.mark.parametrize("bad", [11.0, float("nan"), float("inf")])
def test_invalid_input_value(bad):
    _, child = make_pair()
    obs = torch.zeros(1, 142)
    obs[0, -1] = bad
    with pytest.raises(ValueError):
        child(obs)


def test_roundtrip_and_calibration_isolation():
    parent, child = make_pair()
    center, scale = torch.zeros(6), torch.ones(6)
    restored = build_entry_conditioned_actor_critic(parent.state_dict(), center, scale)
    center.fill_(9)
    scale.fill_(9)
    assert torch.equal(restored.entry_center, torch.zeros(6))
    restored.load_state_dict(child.state_dict(), strict=True)
    obs = torch.randn(4, 142)
    assert all(torch.equal(a, b) for a, b in zip(child(obs), restored(obs), strict=True))


def test_entry_ppo_exact_replay_keeps_parent_frozen():
    from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

    _, child = make_pair()
    before = copy.deepcopy(child.state_dict())
    optimizer = torch.optim.Adam([p for p in child.parameters() if p.requires_grad], lr=0.001)
    observations = torch.randn(4, 2, 142) * 0.01
    with torch.no_grad():
        mean, value = child(observations.reshape(-1, 142))
        dist = torch.distributions.Normal(mean, child.logstd.clamp(-2.5, -0.3).exp())
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(1)
    alive = torch.ones(4, 2)
    following = alive.clone()
    following[-1] = 0
    data = dict(
        obs=observations,
        raw=raw.reshape(4, 2, 30),
        logp=logp.reshape(4, 2),
        value=value.reshape(4, 2),
        reward=torch.randn(4, 2),
        alive=alive,
        next_alive=following,
    )
    rng = torch.get_rng_state()
    config = FullBodyPPOUpdateConfig(observation_size=142, action_size=30, epochs=2)
    result = update_full_body_ppo(child, optimizer, data, config)
    _, clone = make_pair()
    clone.load_state_dict(before)
    clone_optimizer = torch.optim.Adam([p for p in clone.parameters() if p.requires_grad], lr=0.001)
    torch.set_rng_state(rng)
    assert update_full_body_ppo(clone, clone_optimizer, data, config) == result
    for name, value in child.state_dict().items():
        assert torch.equal(value, clone.state_dict()[name])
        if name.startswith("parent."):
            assert torch.equal(value, before[name])
    assert not torch.equal(before["entry_actor.2.weight"], child.entry_actor[-1].weight)
