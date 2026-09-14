import numpy as np
import pytest

from rosclaw_soccer.training.full_kick_learning import (
    build_full_kick_actor_critic,
    full_kick_observation,
)


def parameters():
    rng = np.random.default_rng(91)
    widths = (547, 512, 256, 128, 29)
    return {
        f"{2 * i}.{kind}": (rng.normal(size=shape) * 0.01).astype(np.float32)
        for i in range(4)
        for kind, shape in [("weight", (widths[i + 1], widths[i])), ("bias", (widths[i + 1],))]
    }


def test_context_features_preserve_reference_and_are_owned():
    original = np.arange(547, dtype=np.float32)
    result = full_kick_observation(
        reference_observation=original,
        goal_relative_position=np.array([20.0, -5.0, 1.0]),
        local_root_velocity=np.array([6.0, -1.5, 0.3]),
        learning_active=True,
    )
    assert result.shape == (554,) and result.dtype == np.float32
    np.testing.assert_array_equal(result[:547], original)
    np.testing.assert_allclose(result[547:], [1, -0.5, 0.1, 1, -0.5, 0.1, 1])
    result[0] = -9
    assert original[0] == 0


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "large", "bit"])
def test_invalid_features_rejected(kind):
    ref = np.zeros(547, dtype=np.float32)
    if kind == "shape":
        ref = ref[:-1]
    if kind == "dtype":
        ref = ref.astype(np.int64)
    if kind == "nan":
        ref[0] = np.nan
    if kind == "large":
        ref[0] = 1e5
    with pytest.raises(ValueError):
        full_kick_observation(
            reference_observation=ref,
            goal_relative_position=np.zeros(3),
            local_root_velocity=np.zeros(3),
            learning_active=1 if kind == "bit" else False,
        )


@pytest.mark.parametrize("kind", ["keys", "shape", "dtype", "nan"])
def test_reference_parameter_contract(kind):
    state = parameters()
    if kind == "keys":
        state.pop("0.bias")
    if kind == "shape":
        state["0.bias"] = state["0.bias"][:-1]
    if kind == "dtype":
        state["0.bias"] = state["0.bias"].astype(np.float64)
    if kind == "nan":
        state["0.bias"][0] = np.nan
    with pytest.raises(ValueError):
        build_full_kick_actor_critic(state)


def test_import_preserves_actor_rng_and_ignores_training_bit():
    torch = pytest.importorskip("torch")
    state = parameters()
    before = torch.get_rng_state().clone()
    model = build_full_kick_actor_critic(state)
    assert torch.equal(before, torch.get_rng_state())
    obs = torch.randn(3, 554)
    with torch.no_grad():
        actual, value = model(obs)
        torch.testing.assert_close(actual, model.actor(obs[:, :547]), rtol=1e-5, atol=1e-6)
        assert torch.count_nonzero(value) == 0
        changed = obs.clone()
        changed[:, 553] = 500
        assert torch.equal(actual, model(changed)[0])
    state["0.weight"].fill(9)
    assert not torch.all(model.actor[0].weight == 9)


def test_import_does_not_inherit_ambient_float64_default():
    torch = pytest.importorskip("torch")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = build_full_kick_actor_critic(parameters())
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    finally:
        torch.set_default_dtype(previous)


def test_complete_actor_and_context_can_update_with_explicit_ppo_contract():
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

    torch.set_num_threads(1)
    torch.manual_seed(83)
    model = build_full_kick_actor_critic(parameters())
    obs = torch.randn(4, 2, 554)
    obs[..., 553] = 1
    with torch.no_grad():
        mean, value = model(obs.reshape(-1, 554))
        dist = torch.distributions.Normal(mean, model.logstd.exp())
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
    following = torch.ones(4, 2)
    following[-1] = 0
    data = dict(
        obs=obs,
        raw=raw.reshape(4, 2, 29),
        logp=logp.reshape(4, 2),
        value=value.reshape(4, 2),
        reward=torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8,
        alive=torch.ones(4, 2),
        next_alive=following,
    )
    original = model.actor[0].weight.detach().clone()
    result = update_full_body_ppo(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-5),
        data,
        FullBodyPPOUpdateConfig(
            epochs=1,
            minibatch_size=8,
            observation_size=554,
            action_size=29,
            minimum_log_std=-4,
            learning_observation_index=553,
            critic_all_active=True,
        ),
    )
    assert result["optimizer_steps"] == 1 and result["on_policy_inputs_verified"]
    assert not torch.equal(original, model.actor[0].weight)
    assert torch.count_nonzero(model.context_adapter.weight) > 0
