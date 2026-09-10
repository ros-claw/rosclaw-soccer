import pytest

from rosclaw_soccer.training.coupled_ball_residual import advance_coupled_residual
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from rosclaw_soccer.training.reception_navigation import (
    advance_reception_navigation,
    build_reception_navigation_actor_critic,
    normalized_reception_navigation,
)

torch = pytest.importorskip("torch")


def test_explicit_new_contract_starts_with_zero_navigation():
    model = build_reception_navigation_actor_critic()
    mean, value = model(torch.zeros(8, 136))
    assert mean.shape == (8, 3) and value.shape == (8,)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert FullBodyPPOUpdateConfig(observation_size=136, action_size=3).action_size == 3
    with pytest.raises(ValueError):
        model(torch.zeros(8, 133))


def test_exact_existing_navigation_envelope_without_joint_authority():
    previous = torch.zeros(8, 3)
    generator = torch.Generator().manual_seed(467)
    for _ in range(100):
        raw = torch.randn(8, 3, generator=generator) * 3
        before = previous.clone()
        actual = advance_reception_navigation(raw, previous)
        _, expected = advance_coupled_residual(
            torch.cat((torch.zeros(8, 29), raw), 1), torch.zeros(8, 29), previous
        )
        assert torch.equal(actual, expected) and torch.equal(previous, before)
        assert bool(
            (torch.linalg.vector_norm(actual[:, :2] - previous[:, :2], dim=1) <= 0.0120001).all()
        )
        assert bool(((actual[:, 2] - previous[:, 2]).abs() <= 0.0400001).all())
        assert bool((normalized_reception_navigation(actual).abs() <= 1.000001).all())
        previous = actual


@pytest.mark.parametrize("bad", ["nan", "gradient", "dtype", "shape", "history", "list"])
def test_invalid_action_or_history_is_rejected_without_mutation(bad):
    raw = torch.zeros(2, 3)
    previous = torch.zeros(2, 3)
    if bad == "nan":
        raw[0, 0] = float("nan")
    elif bad == "gradient":
        raw.requires_grad_(True)
    elif bad == "dtype":
        raw = raw.double()
    elif bad == "shape":
        raw = torch.zeros(2, 29)
    elif bad == "history":
        previous[0, 0] = 0.8
    else:
        raw = [[0.0, 0.0, 0.0]]
    before = previous.clone()
    with pytest.raises(ValueError):
        advance_reception_navigation(raw, previous)
    assert torch.equal(previous, before)


def test_navigation_only_update_is_on_policy_and_changes_no_external_motor():
    torch.manual_seed(467)
    model = build_reception_navigation_actor_critic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    obs = torch.randn(4, 8, 136) * 0.1
    with torch.no_grad():
        mean, value = model(obs.flatten(0, 1))
        normal = torch.distributions.Normal(mean, model.logstd.exp())
        raw = normal.sample()
        data = dict(
            obs=obs,
            raw=raw.reshape(4, 8, 3),
            logp=normal.log_prob(raw).sum(1).reshape(4, 8),
            value=value.reshape(4, 8),
            reward=torch.randn(4, 8),
            alive=torch.ones(4, 8),
            next_alive=torch.ones(4, 8),
        )
    update = update_full_body_ppo(
        model,
        optimizer,
        data,
        FullBodyPPOUpdateConfig(observation_size=136, action_size=3, minibatch_size=16),
    )
    assert update["on_policy_inputs_verified"]
    assert update["optimizer_steps"] > 0
    assert update["activation_ceiling"] == "SIM_ONLY"
