import numpy as np
import pytest

from rosclaw_soccer.training.autoregressive_kick_learning import (
    autoregressive_kick_observation,
    build_autoregressive_kick_actor_critic,
)
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig
from rosclaw_soccer.training.full_kick_learning import build_full_kick_actor_critic
from rosclaw_soccer.training.guarded_ppo import guarded_full_body_ppo_update


def reference():
    rng = np.random.default_rng(1435)
    widths = (547, 512, 256, 128, 29)
    return {
        f"{2 * i}.{kind}": (rng.normal(size=shape) * 0.01).astype(np.float32)
        for i in range(4)
        for kind, shape in (("weight", (widths[i + 1], widths[i])), ("bias", (widths[i + 1],)))
    }


def test_observation_ownership_and_inactive_reset():
    base = np.zeros(554, dtype=np.float32)
    previous = np.ones(29, dtype=np.float32)
    with pytest.raises(ValueError, match="inactive"):
        autoregressive_kick_observation(base_observation=base, previous_raw_action=previous)
    base[553] = 1
    result = autoregressive_kick_observation(base_observation=base, previous_raw_action=previous)
    assert result.shape == (583,) and result.dtype == np.float32
    result[:] = 5
    assert base[0] == 0 and previous[0] == 1


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "large", "bit"])
def test_invalid_observation_rejected(kind):
    base = np.zeros(554, dtype=np.float32)
    previous = np.zeros(29, dtype=np.float32)
    if kind == "shape":
        previous = previous[:-1]
    elif kind == "dtype":
        previous = previous.astype(np.float64)
    elif kind == "nan":
        previous[0] = np.nan
    elif kind == "large":
        previous[0] = 1e5
    else:
        base[553] = 0.5
    with pytest.raises(ValueError):
        autoregressive_kick_observation(base_observation=base, previous_raw_action=previous)


@pytest.mark.parametrize("rho", [-0.1, 0.91, True, float("nan"), float("inf")])
def test_invalid_continuation_rejected_before_torch(rho):
    with pytest.raises(ValueError, match="continuation"):
        build_autoregressive_kick_actor_critic({}, continuation_factor=rho)


def test_zero_factor_exact_base_rng_ownership_and_stateless_history():
    torch = pytest.importorskip("torch")
    state = reference()
    rng = torch.get_rng_state().clone()
    model = build_autoregressive_kick_actor_critic(state, continuation_factor=0)
    original = build_full_kick_actor_critic(state)
    assert torch.equal(rng, torch.get_rng_state())
    obs = torch.randn(4, 583)
    obs[:, 553] = 1
    with torch.no_grad():
        actual, value = model(obs)
        expected, expected_value = original(obs[:, :554])
        assert torch.equal(actual, expected) and torch.equal(value, expected_value)
        changed = obs.clone()
        changed[:, 554:] += 2
        assert torch.equal(model(changed)[0], actual)
        assert torch.equal(model(obs)[0], actual)
    assert model.logstd is model.base.logstd
    assert not model.continuation_factor.requires_grad
    model.base.actor[0].weight.data.zero_()
    assert np.any(state["0.weight"] != 0)


def test_conditional_mean_and_serial_batch_match_without_hidden_cache():
    torch = pytest.importorskip("torch")
    model = build_autoregressive_kick_actor_critic(reference(), continuation_factor=0.25)
    obs = torch.randn(5, 583)
    obs[:, 553] = 1
    obs[0, 553:] = 0
    with torch.no_grad():
        base, _ = model.base(obs[:, :554])
        mean, _ = model(obs)
        expected = (0.75 * base.double() + 0.25 * obs[:, 554:].double()).float()
        expected[0] = base[0]
        assert torch.equal(mean, expected)
        assert torch.equal(torch.cat([model(x[None])[0] for x in obs]), mean)
        changed = obs.clone()
        changed[1, 554] += 1
        assert float(model(changed)[0][1, 0] - mean[1, 0]) == pytest.approx(0.25)
        assert torch.equal(model(obs)[0], mean)
    obs[0, 554] = 1
    with pytest.raises(ValueError, match="inactive"):
        model(obs)
    obs[0, 554] = 0
    model.continuation_factor.fill_(float("nan"))
    with pytest.raises(ValueError):
        model(obs)


def test_conditional_gaussian_rollout_is_accepted_by_real_guarded_ppo():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = build_autoregressive_kick_actor_critic(reference(), continuation_factor=0.25)
    frames = []
    previous = torch.zeros(2, 29)  # Explicit fixed, parameter-independent entry seed.
    for _ in range(8):
        obs = torch.zeros(2, 583)
        obs[:, 553] = 1
        obs[:, 554:] = previous
        with torch.no_grad():
            mean, value = model(obs)
            dist = torch.distributions.Normal(mean, model.logstd.exp())
            raw = dist.sample()
            logp = dist.log_prob(raw).sum(-1)
        frames.append(dict(obs=obs, raw=raw, value=value, logp=logp))
        previous = raw.clone()
    data = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
    assert torch.equal(data["obs"][1:, :, 554:], data["raw"][:-1])
    data.update(reward=torch.ones(8, 2), alive=torch.ones(8, 2), next_alive=torch.ones(8, 2))
    data["next_alive"][-1] = 0
    config = FullBodyPPOUpdateConfig(
        epochs=1,
        minibatch_size=16,
        observation_size=583,
        action_size=29,
        minimum_log_std=-4,
        learning_observation_index=553,
        critic_all_active=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    result = guarded_full_body_ppo_update(model, optimizer, data, config)
    assert result["accepted"] and result["on_policy_inputs_verified"]
    assert float(model.continuation_factor) == 0.25
