import numpy as np
import pytest

from rosclaw_soccer.training.temporal_native_anchor_learning import (
    build_temporal_native_anchor_actor_critic,
    temporal_native_anchor_observation,
)

torch = pytest.importorskip("torch")


def build(rho=0.75):
    return build_temporal_native_anchor_actor_critic(continuation_factor=rho)


def observation(base, previous):
    return temporal_native_anchor_observation(base_observation=base, previous_residual=previous)


def inputs():
    x = torch.zeros((2, 612), dtype=torch.float32)
    x[:, 553:582] = torch.linspace(-3, 3, 29)
    x[:, 582] = 1
    return x


def test_zero_feedback_preserves_native_exactly():
    for rho in (0, 0.25, 0.75, 0.9):
        actor = build(rho=rho)
        x = inputs()
        for _ in range(120):
            mean, _ = actor(x)
            assert torch.equal(mean, x[:, 553:582])
            x[:, 583:] = mean.detach() - x[:, 553:582]
            x[:, 553:582] *= -0.99


def test_zero_rho_exact_base_after_learning():
    actor = build(rho=0)
    with torch.no_grad():
        actor.base.actor[-1].bias.fill_(0.8)
    x = inputs()
    x[:, 583:] = 0.3
    actual = actor(x)
    expected = actor.base(x[:, :583])
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_history_changes_only_feedback_no_native_lag():
    actor = build(rho=0.75)
    x = inputs()
    x[:, 583:] = 0.04
    mean, _ = actor(x)
    torch.testing.assert_close(mean, x[:, 553:582] + 0.03)
    x[:, 553:582] *= -3
    mean, _ = actor(x)
    torch.testing.assert_close(mean, x[:, 553:582] + 0.03)


def test_no_hidden_history_or_rng():
    actor = build()
    x = inputs()
    rng = torch.get_rng_state().clone()
    first = actor(x)
    altered = x.clone()
    altered[:, 583:] = 0.2
    actor(altered)
    second = actor(x)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("rho", [-0.01, 0.91, float("nan"), float("inf"), True])
def test_bad_rho(rho):
    with pytest.raises(ValueError):
        build(rho=rho)


def test_history_owned_copy_and_reset():
    x = inputs()[0, :583].numpy().copy()
    p = np.full(29, 0.1, np.float32)
    packed = observation(x, p)
    p[:] = 0
    assert (packed[583:] == 0.1).all()
    x[582] = 0
    with pytest.raises(ValueError):
        observation(x, np.ones(29, np.float32))
    bad = inputs()
    bad[:, 582] = 0
    bad[:, 583:] = 0.1
    with pytest.raises(ValueError):
        build()(bad)


def test_actor_and_critic_gradients():
    actor = build()
    x = inputs()
    mean, value = actor(x)
    (mean.square().sum() + (value - 1).square().sum()).backward()
    assert actor.base.actor[-1].bias.grad.abs().sum() > 0
    assert actor.base.critic[-1].bias.grad.abs().sum() > 0
    assert not actor.rho.requires_grad


@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "bound", "context", "bit"])
def test_reject_malformed_context_in_packer_and_model(damage):
    x = inputs()[0].numpy().copy()
    if damage == "shape":
        x = x[:-1]
    elif damage == "dtype":
        x = x.astype(np.float64)
    elif damage == "nan":
        x[600] = np.nan
    elif damage == "bound":
        x[600] = 10001
    elif damage == "context":
        x[547] = 1.01
    else:
        x[582] = 0.5
    with pytest.raises(ValueError):
        observation(x[:583], x[583:])
    with pytest.raises(ValueError):
        build()(torch.from_numpy(x)[None])


def test_inactive_segment_uses_base_and_is_not_an_execution_gate():
    model = build()
    with torch.no_grad():
        model.base.actor[-1].bias.fill_(0.5)
    x = inputs()
    x[:, 582] = 0
    mean, _ = model(x)
    assert torch.equal(mean, model.base(x[:, :583])[0])
    assert not torch.equal(mean, x[:, 553:582])


def test_serialized_rho_is_validated_and_not_optimized():
    model = build()
    assert "rho" in model.state_dict()
    assert all(name != "rho" for name, _ in model.named_parameters())
    with torch.no_grad():
        model.rho.fill_(float("nan"))
    with pytest.raises(ValueError):
        model(inputs())


def test_explicit_history_likelihood_is_accepted_by_bounded_ppo():
    from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

    model = build()
    model.logstd.requires_grad_(False)
    generator = torch.Generator().manual_seed(2556)
    observations, actions, probabilities, values = [], [], [], []
    history = torch.zeros((2, 29))
    for _ in range(4):
        x = inputs()
        x[:, 583:] = history
        with torch.no_grad():
            outputs = [model(row[None]) for row in x]
            mean = torch.cat([out[0] for out in outputs])
            value = torch.cat([out[1] for out in outputs])
            dist = torch.distributions.Normal(mean, model.logstd.exp())
            raw = mean + dist.scale * torch.randn(mean.shape, generator=generator)
            probability = dist.log_prob(raw).sum(1)
        history = raw - x[:, 553:582]
        observations.append(x)
        actions.append(raw)
        probabilities.append(probability)
        values.append(value)
    batch = dict(
        obs=torch.stack(observations),
        raw=torch.stack(actions),
        logp=torch.stack(probabilities),
        value=torch.stack(values),
        reward=torch.zeros(4, 2),
        alive=torch.ones(4, 2),
        next_alive=torch.ones(4, 2),
    )
    batch["next_alive"][-1] = 0
    batch["reward"][-1] = torch.tensor([1.0, -1.0])
    config = FullBodyPPOUpdateConfig(
        observation_size=612,
        action_size=29,
        epochs=1,
        minibatch_size=4,
        minimum_log_std=-6.0,
        learning_observation_index=582,
        stateless_rowwise_forward=True,
    )
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-7)
    old = {k: v.clone() for k, v in model.state_dict().items()}
    damaged = {**batch, "obs": batch["obs"].clone()}
    damaged["obs"][1, 0, 583] += 0.1
    with pytest.raises(ValueError):
        update_full_body_ppo(model, optimizer, damaged, config)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.items())
    stats = update_full_body_ppo(model, optimizer, batch, config)
    assert stats["optimizer_steps"] > 0
    assert torch.equal(old["rho"], model.rho)
    assert not torch.equal(old["base.actor.2.bias"], model.base.actor[-1].bias)
