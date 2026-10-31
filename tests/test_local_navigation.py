import copy
from dataclasses import replace

import numpy as np
import pytest
from test_navigation_option import observation

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.local_navigation import (
    build_local_navigation_actor_critic,
    lift_local_navigation_state,
    local_navigation_delta,
    local_navigation_features,
    team_navigation_features,
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


def test_team_context_distinguishes_teammates_opponents_and_missing():
    base = observation()
    # The existing 39 fields are identical when the same neighbor changes team.
    ally = replace(base, neighbors=(("blue.defender", 1.0, 1.0),))
    enemy = replace(base, neighbors=(("red.defender", 1.0, 1.0),))
    ally_features, enemy_features = team_navigation_features(ally), team_navigation_features(enemy)
    np.testing.assert_array_equal(ally_features[:39], enemy_features[:39])
    np.testing.assert_array_equal(ally_features[39:], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(enemy_features[39:], [1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        team_navigation_features(replace(base, neighbors=(("unknown.defender", 1.0, 1.0),)))
    crowded = replace(
        base,
        neighbors=(
            ("blue.defender", 0.1, 0.0),
            ("blue.finisher", 2.0, 0.0),
            ("red.defender", -0.1, 0.0),
            ("red.finisher", 3.0, 0.0),
        ),
    )
    np.testing.assert_array_equal(team_navigation_features(crowded)[39:], [1, 1, 1, 0, 1, 1])


def test_team_context_lift_preserves_old_actor_and_can_learn_new_features():
    torch = pytest.importorskip("torch")
    torch.manual_seed(25)
    parent = build_local_navigation_actor_critic()
    torch.nn.init.normal_(parent.actor[-1].weight, std=0.1)
    state = {k: v.detach().numpy().copy() for k, v in parent.state_dict().items()}
    team = build_local_navigation_actor_critic(team_context=True, training_influence=True)
    team.load_state_dict(
        {k: torch.from_numpy(v) for k, v in lift_local_navigation_state(state).items()}
    )
    ally = replace(observation(), neighbors=(("blue.defender", 1.0, 1.0),))
    enemy = replace(observation(), neighbors=(("red.defender", 1.0, 1.0),))
    inputs = torch.tensor(
        np.stack((team_navigation_features(ally), team_navigation_features(enemy)))
    )
    inputs = torch.cat((inputs, torch.tensor([[0.0], [1.0]])), dim=1)
    mean, value = team(inputs)
    with torch.no_grad():
        expected_mean, expected_value = parent(inputs[:, :39])
    torch.testing.assert_close(mean, expected_mean, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(value, expected_value, atol=1e-7, rtol=1e-6)
    (mean[0, 0] - mean[1, 0]).backward()
    assert float(team.actor[0].weight.grad[:, 39:].abs().sum()) > 0
    assert FullBodyPPOUpdateConfig(
        observation_size=46, action_size=3, learning_observation_index=45
    )
    invalid = inputs.clone()
    invalid[0, 39:41] = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError):
        team(invalid)


def test_team_lift_rejects_incomplete_or_nonfinite_state():
    pytest.importorskip("torch")
    actor = build_local_navigation_actor_critic()
    state = {k: v.detach().numpy().copy() for k, v in actor.state_dict().items()}
    missing = dict(state)
    missing.pop("logstd")
    with pytest.raises(ValueError):
        lift_local_navigation_state(missing)
    state["actor.0.weight"][0, 0] = np.nan
    with pytest.raises(ValueError):
        lift_local_navigation_state(state)


def test_team_navigation_ppo_accepts_causal_features_with_influence_mask():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    actor = build_local_navigation_actor_critic(team_context=True, training_influence=True)
    inputs = torch.zeros((4, 2, 46), dtype=torch.float32)
    inputs[:, 0, 39:41] = torch.tensor([1.0, 1.0])
    inputs[:, 1, 39:41] = torch.tensor([1.0, 0.0])
    inputs[:, :, -1] = 1
    with torch.no_grad():
        mean, value = actor(inputs.reshape(-1, 46))
        distribution = torch.distributions.Normal(mean, actor.logstd.clamp(-2.5, -0.3).exp())
        raw = distribution.sample()
        data = {
            "obs": inputs,
            "raw": raw.reshape(4, 2, 3),
            "logp": distribution.log_prob(raw).sum(1).reshape(4, 2),
            "value": value.reshape(4, 2),
            "reward": torch.randn(4, 2),
            "alive": torch.ones(4, 2),
            "next_alive": torch.ones(4, 2),
        }
    data["next_alive"][-1] = 0
    config = FullBodyPPOUpdateConfig(
        epochs=1,
        minibatch_size=4,
        observation_size=46,
        action_size=3,
        learning_observation_index=45,
    )
    result = update_full_body_ppo(
        actor, torch.optim.Adam(actor.parameters(), lr=1e-4), data, config
    )
    assert result["optimizer_steps"] > 0 and result["on_policy_inputs_verified"]
    assert not result["promotion_eligible"]
