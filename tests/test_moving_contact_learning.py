import numpy as np
import pytest

from rosclaw_soccer.training.moving_contact_learning import (
    advance_moving_contact_residual,
    build_moving_contact_actor_critic,
    moving_contact_observation,
    moving_contact_potential,
)


def observation():
    return dict(
        locomotion_observation=np.zeros(96),
        ball_relative_position=np.ones(3),
        ball_relative_velocity=np.ones(3),
        target_relative_position=np.ones(3),
        phase=0.5,
        learning_active=True,
        previous_residual=np.zeros(29),
    )


def test_feature_contract():
    result = moving_contact_observation(**observation())
    assert result.shape == (136,) and result.dtype == np.float32
    assert result[105] == 0 and result[106] == 1
    np.testing.assert_array_equal(result[96:99], [0.5, 0.5, 0.5])
    assert np.max(np.abs(result)) <= 1


def test_bounded_residual_and_decay():
    last = np.zeros(29)
    for _ in range(30):
        nxt = advance_moving_contact_residual(np.ones(29) * 10, last, learning_active=True)
        assert np.max(np.abs(nxt - last)) <= 0.025001 and np.max(np.abs(nxt)) <= 0.25
        last = nxt
    for _ in range(11):
        last = advance_moving_contact_residual(np.ones(29), last, learning_active=False)
    np.testing.assert_array_equal(last, np.zeros(29))


@pytest.mark.parametrize(
    "change",
    [
        dict(phase=float("nan")),
        dict(learning_active=1),
        dict(previous_residual=np.ones(29)),
        dict(ball_relative_velocity=np.full(3, np.inf)),
    ],
)
def test_reject_bad_observations(change):
    args = observation()
    args.update(change)
    with pytest.raises(ValueError):
        moving_contact_observation(**args)


def test_potential_not_success_and_failure_cannot_gain_over_penalty():
    assert (
        moving_contact_potential(progress=1.0, lateral_error=0.0, height=0.115, failed=False) == 3
    )
    assert moving_contact_potential(progress=1.0, lateral_error=1.0, height=1.2, failed=False) == -6
    assert moving_contact_potential(progress=1.0, lateral_error=0.0, height=0.115, failed=True) == 0
    with pytest.raises(ValueError):
        moving_contact_potential(progress=float("nan"), lateral_error=0.0, height=0.1, failed=False)


def test_new_model_is_zero_residual_not_a_foundation():
    torch = pytest.importorskip("torch")
    model = build_moving_contact_actor_critic()
    mean, value = model(torch.from_numpy(moving_contact_observation(**observation()))[None])
    assert torch.equal(mean, torch.zeros(1, 29)) and torch.equal(value, torch.zeros(1))
    assert all(p.requires_grad for p in model.parameters())


def test_explicit_ppo_window_and_full_critic_update():
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

    torch.manual_seed(19)
    model = build_moving_contact_actor_critic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    obs = torch.zeros(6, 4, 136)
    obs[2:5, :, 106] = 1
    with torch.no_grad():
        mean, value = model(obs.reshape(-1, 136))
        dist = torch.distributions.Normal(mean, model.logstd.exp())
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
    alive = torch.ones(6, 4)
    following = alive.clone()
    following[-1] = 0
    reward = torch.zeros(6, 4)
    reward[-1] = torch.tensor([10.0, -2.0, 10.0, -2.0])
    rollout = dict(
        obs=obs,
        raw=raw.reshape(6, 4, 29),
        logp=logp.reshape(6, 4),
        value=value.reshape(6, 4),
        reward=reward,
        alive=alive,
        next_alive=following,
    )
    result = update_full_body_ppo(
        model,
        optimizer,
        rollout,
        FullBodyPPOUpdateConfig(
            epochs=1,
            minibatch_size=8,
            observation_size=136,
            action_size=29,
            minimum_log_std=-4.0,
            learning_observation_index=106,
            critic_all_active=True,
        ),
    )
    assert result["active_samples"] == 12
    assert result["verified_active_samples"] == 24
    assert result["critic_samples_per_epoch"] == 24
    assert result["on_policy_inputs_verified"]


@pytest.mark.parametrize("raw", [np.zeros(28), np.full(29, np.nan), np.ones(29) * 101])
def test_reject_invalid_residual_latents(raw):
    with pytest.raises(ValueError):
        advance_moving_contact_residual(raw, np.zeros(29), learning_active=True)
