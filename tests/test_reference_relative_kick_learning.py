import numpy as np
import pytest

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig
from rosclaw_soccer.training.guarded_ppo import guarded_full_body_ppo_update
from rosclaw_soccer.training.reference_relative_kick_learning import (
    build_reference_relative_kick_actor_critic,
    reference_relative_kick_observation,
)


def reference():
    rng = np.random.default_rng(1450)
    widths = (547, 512, 256, 128, 29)
    return {
        f"{2 * i}.{kind}": (rng.normal(size=shape) * 0.01).astype(np.float32)
        for i in range(4)
        for kind, shape in (("weight", (widths[i + 1], widths[i])), ("bias", (widths[i + 1],)))
    }


def test_explicit_history_difference_and_reset():
    base = np.zeros(554, dtype=np.float32)
    raw = np.ones(29, dtype=np.float32)
    teacher = np.full(29, 0.25, dtype=np.float32)
    with pytest.raises(ValueError, match="reset"):
        reference_relative_kick_observation(
            base_observation=base, previous_raw_action=raw, previous_frozen_reference_mean=raw
        )
    base[553] = 1
    out = reference_relative_kick_observation(
        base_observation=base, previous_raw_action=raw, previous_frozen_reference_mean=teacher
    )
    np.testing.assert_array_equal(out[554:], np.full(29, 0.75, dtype=np.float32))
    out[:] = 0
    assert base[553] == 1 and raw[0] == 1 and teacher[0] == 0.25


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(28, dtype=np.float32),
        np.zeros(29, dtype=np.float64),
        np.full(29, np.nan, dtype=np.float32),
        np.full(29, 5001, dtype=np.float32),
    ],
)
def test_invalid_history_rejected(bad):
    with pytest.raises(ValueError):
        reference_relative_kick_observation(
            base_observation=np.zeros(554, dtype=np.float32),
            previous_raw_action=np.zeros(29, dtype=np.float32),
            previous_frozen_reference_mean=bad,
        )


@pytest.mark.parametrize("rho", [-0.1, 0.91, True, float("nan")])
def test_bad_factor_rejected(rho):
    with pytest.raises(ValueError):
        build_reference_relative_kick_actor_critic({}, continuation_factor=rho)


@pytest.mark.parametrize("rho", [0, 0.25, 0.5, 0.9])
def test_zero_deviation_preserves_fast_reference_means_exactly(rho):
    torch = pytest.importorskip("torch")
    rng = torch.get_rng_state().clone()
    model = build_reference_relative_kick_actor_critic(reference(), continuation_factor=rho)
    assert torch.equal(rng, torch.get_rng_state())
    obs = torch.randn(8, 583)
    obs[:, 553] = 1
    obs[:, 554:] = 0
    with torch.no_grad():
        mean, value = model(obs)
        original, original_value = model.base(obs[:, :554])
        assert torch.equal(mean, original) and torch.equal(value, original_value)
        assert torch.equal(torch.cat([model(x[None])[0] for x in obs]), mean)


def test_conditional_distribution_and_real_ppo_no_reference_mutation():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    state = reference()
    model = build_reference_relative_kick_actor_critic(state, continuation_factor=0.5)
    frozen = build_reference_relative_kick_actor_critic(state, continuation_factor=0)
    frozen_before = {k: v.clone() for k, v in frozen.state_dict().items()}
    history = torch.zeros(2, 29)
    frames = []
    for _ in range(8):
        obs = torch.zeros(2, 583)
        obs[:, 553] = 1
        obs[:, 554:] = history
        with torch.no_grad():
            mean, value = model(obs)
            base, _ = model.base(obs[:, :554])
            assert torch.equal(mean, (base.double() + 0.5 * history.double()).float())
            teacher, _ = frozen(obs)
            dist = torch.distributions.Normal(mean, model.logstd.exp())
            raw = dist.sample()
            frames.append(dict(obs=obs, raw=raw, value=value, logp=dist.log_prob(raw).sum(-1)))
            history = (raw.double() - teacher.double()).float()
    data = {k: torch.stack([x[k] for x in frames]) for k in frames[0]}
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
    result = guarded_full_body_ppo_update(
        model, torch.optim.Adam(model.parameters(), lr=1e-7), data, config
    )
    assert result["accepted"] and result["on_policy_inputs_verified"]
    for key, value in frozen.state_dict().items():
        assert torch.equal(value, frozen_before[key])
