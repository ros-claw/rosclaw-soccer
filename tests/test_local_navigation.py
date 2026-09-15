import copy
from dataclasses import replace

import numpy as np
import pytest
from test_navigation_option import observation

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.local_navigation import (
    build_local_navigation_actor_critic,
    local_navigation_delta,
    local_navigation_features,
)


def test_features_are_heading_explicit_and_translation_invariant():
    o = observation()
    a = local_navigation_features(o)
    assert a.shape == (39,) and a.dtype == np.float32 and np.isfinite(a).all()
    rotated = replace(o, body_pose=(*o.body_pose[:3], 0.0, 0.0, 0.0, 1.0))
    b = local_navigation_features(rotated)
    assert a[1] == 1 and b[1] == -1
    translated = replace(
        o,
        body_pose=(2.0, 3.0, *o.body_pose[2:]),
        ball_position=(2.4, 3.1, 0.115),
        task_target=(4.0, 4.0, 0.0),
        steering_target=(2.2, 3.1),
        neighbors=(("red.defender", 3.0, 4.0),),
    )
    np.testing.assert_allclose(a, local_navigation_features(translated), atol=1e-6)


def test_delta_is_bounded_and_rejects_bad_latents():
    assert local_navigation_delta(np.zeros(3)) == (0.0, 0.0, 0.0)
    delta = local_navigation_delta(np.full(3, 1000.0))
    assert np.linalg.norm(delta[:2]) <= 0.250000001 and abs(delta[2]) <= 0.4
    for value in (np.zeros(4), np.zeros(3, dtype=int), np.full(3, np.nan)):
        with pytest.raises(ValueError):
            local_navigation_delta(value)


def test_initial_actor_is_zero_and_influence_bit_cannot_leak_into_policy():
    torch = pytest.importorskip("torch")
    actor = build_local_navigation_actor_critic(training_influence=True)
    features = torch.tensor(local_navigation_features(observation()))[None]
    a = torch.cat((features, torch.zeros((1, 1))), 1)
    b = torch.cat((features, torch.ones((1, 1))), 1)
    ma, va = actor(a)
    mb, vb = actor(b)
    assert torch.equal(ma, torch.zeros_like(ma)) and torch.equal(ma, mb)
    assert torch.equal(va, vb)
    config = FullBodyPPOUpdateConfig(
        observation_size=40, action_size=3, learning_observation_index=39
    )
    assert config.learning_observation_index == 39
    b[:, -1] = 0.5
    with pytest.raises(ValueError):
        actor(b)


def test_navigation_optimizer_replays_with_actual_influence_selection():
    torch = pytest.importorskip("torch")
    torch.manual_seed(2072)
    actor = build_local_navigation_actor_critic(training_influence=True)
    replay = copy.deepcopy(actor)
    obs = torch.randn(4, 2, 40) * 0.1
    obs[:, :, -1] = 1
    obs[0, :, -1] = 0
    with torch.no_grad():
        mean, value = actor(obs.reshape(-1, 40))
        dist = torch.distributions.Normal(mean, actor.logstd.clamp(-2.5, -0.3).exp())
        raw = dist.sample()
        data = dict(
            obs=obs,
            raw=raw.reshape(4, 2, 3),
            logp=dist.log_prob(raw).sum(1).reshape(4, 2),
            value=value.reshape(4, 2),
            reward=torch.randn(4, 2),
            alive=torch.ones(4, 2),
            next_alive=torch.ones(4, 2),
        )
    data["next_alive"][-1] = 0
    config = FullBodyPPOUpdateConfig(
        epochs=2,
        minibatch_size=4,
        observation_size=40,
        action_size=3,
        learning_observation_index=39,
        critic_all_active=True,
    )
    rng = torch.get_rng_state()
    result = update_full_body_ppo(
        actor, torch.optim.Adam(actor.parameters(), lr=1e-4), data, config
    )
    torch.set_rng_state(rng)
    repeated = update_full_body_ppo(
        replay, torch.optim.Adam(replay.parameters(), lr=1e-4), data, config
    )
    assert result == repeated and result["optimizer_steps"] > 0
    assert not result["promotion_eligible"]
    assert all(torch.equal(v, replay.state_dict()[k]) for k, v in actor.state_dict().items())
