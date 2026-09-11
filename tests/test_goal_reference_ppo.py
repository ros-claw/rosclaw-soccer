"""Optimizer arithmetic qualification, not a physical skill success claim."""

import copy

import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.goal_reference_actor import build_goal_reference_actor_critic

torch = pytest.importorskip("torch")


def case():
    torch.manual_seed(69701)
    seed = build_ball_residual_actor_critic().state_dict()
    actor = build_goal_reference_actor_critic(seed)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    observations = torch.randn(4, 2, 136) * 0.1
    with torch.no_grad():
        mean, value = actor(observations.reshape(-1, 136))
        distribution = torch.distributions.Normal(mean, actor.logstd.clamp(-2.5, -0.3).exp())
        raw = distribution.sample()
        logp = distribution.log_prob(raw).sum(1)
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
    return seed, actor, optimizer, data


def test_complete_joint_reference_update_replays_exactly():
    seed, actor, optimizer, data = case()
    before = copy.deepcopy(actor.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state()
    config = FullBodyPPOUpdateConfig(observation_size=136, action_size=30, epochs=2)
    result = update_full_body_ppo(actor, optimizer, data, config)
    assert result["on_policy_inputs_verified"] and result["active_samples"] == 8
    assert not torch.equal(
        before["reference_actor.2.weight"], actor.state_dict()["reference_actor.2.weight"]
    )
    clone = build_goal_reference_actor_critic(seed)
    clone.load_state_dict(before)
    clone_optimizer = torch.optim.Adam(clone.parameters(), lr=3e-4)
    clone_optimizer.load_state_dict(state)
    torch.set_rng_state(rng)
    repeated = update_full_body_ppo(clone, clone_optimizer, data, config)
    assert repeated == result
    for name, value in actor.state_dict().items():
        assert torch.equal(value, clone.state_dict()[name])


def test_stale_reference_action_rejected_before_weights_change():
    _, actor, optimizer, data = case()
    before = copy.deepcopy(actor.state_dict())
    data["raw"][:, :, 29] += 1
    with pytest.raises(ValueError, match="likelihood"):
        update_full_body_ppo(
            actor, optimizer, data, FullBodyPPOUpdateConfig(observation_size=136, action_size=30)
        )
    assert all(torch.equal(value, actor.state_dict()[name]) for name, value in before.items())
