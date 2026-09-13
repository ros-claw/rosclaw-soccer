import pytest

from rosclaw_soccer.training.contact_feedback_actor import (
    build_contact_feedback_actor_critic,
    build_normalized_contact_actor_critic,
)
from rosclaw_soccer.training.coupled_ball_residual import build_coupled_ball_residual_actor_critic
from rosclaw_soccer.training.frozen_feature_normalization import normalize_network_inputs

torch = pytest.importorskip("torch")


def test_frozen_copied_buffers_scaling_clipping_and_no_input_write():
    mean = torch.tensor([1.0, 2.0])
    scale = torch.tensor([0.1, 0.2])
    head = normalize_network_inputs(torch.nn.Identity(), mean, scale)
    obs = torch.tensor([[1.1, 200.0]])
    before = obs.clone()
    actual = head(obs)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 10.0]]))
    assert torch.equal(obs, before)
    mean.zero_()
    scale.zero_()
    torch.testing.assert_close(head(obs), actual)
    assert not list(head.parameters())
    assert set(head.state_dict()) == {"mean", "scale"}


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), 1e-9])
def test_invalid_scaling_rejected(scale):
    with pytest.raises(ValueError):
        normalize_network_inputs(torch.nn.Identity(), torch.zeros(2), torch.full((2,), scale))


def test_corrupt_buffer_and_invalid_observation_fail_closed():
    head = normalize_network_inputs(torch.nn.Identity(), torch.zeros(2), torch.ones(2))
    for obs in (None, torch.zeros(2), torch.zeros(1, 3), torch.full((1, 2), float("nan"))):
        with pytest.raises(ValueError):
            head(obs)
    head.scale.zero_()
    with pytest.raises(ValueError):
        head(torch.zeros(1, 2))


def test_normalized_contact_parent_identity_and_explicit_checkpoint_variant():
    parent = build_coupled_ball_residual_actor_critic()
    obs = torch.zeros(4, 169)
    obs[:, 136] = 1
    obs[:, 138] = 0.48
    model = build_normalized_contact_actor_critic(
        parent.state_dict(), torch.ones(169), torch.full((169,), 0.03)
    )
    expected = parent(obs[:, :139].contiguous())
    for actual, reference in zip(model(obs), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert not any(p.requires_grad for p in model.parent.parameters())
    with pytest.raises(RuntimeError):
        build_contact_feedback_actor_critic(parent.state_dict()).load_state_dict(model.state_dict())
    assert not model.actor.mean.requires_grad and not model.critic.scale.requires_grad


def test_normalization_contract_rejects_wrong_dtype_or_contact_size():
    with pytest.raises(ValueError):
        normalize_network_inputs(
            torch.nn.Identity(), torch.zeros(2, dtype=torch.float64), torch.ones(2)
        )
    with pytest.raises(ValueError):
        build_normalized_contact_actor_critic(
            build_coupled_ball_residual_actor_critic().state_dict(),
            torch.zeros(139),
            torch.ones(139),
        )


def test_separate_frozen_actor_and_critic_statistics_preserve_initial_parent():
    parent = build_coupled_ball_residual_actor_critic()
    model = build_normalized_contact_actor_critic(
        parent.state_dict(),
        torch.zeros(169),
        torch.full((169,), 0.03),
        critic_mean=torch.ones(169),
        critic_scale=torch.full((169,), 2.0),
    )
    obs = torch.zeros(4, 169)
    obs[:, 136] = 1
    obs[:, 138] = 0.48
    for actual, expected in zip(model(obs), parent(obs[:, :139].contiguous()), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(model.actor.mean, model.critic.mean)
    assert torch.equal(model.critic.scale, torch.full((169,), 2.0))
    assert all(not value.requires_grad for value in model.buffers())


def test_separate_critic_requires_both_vectors_and_explicit_size():
    state = build_coupled_ball_residual_actor_critic().state_dict()
    for extra in (
        {"critic_mean": torch.zeros(169)},
        {"critic_scale": torch.ones(169)},
        {"critic_mean": torch.zeros(139), "critic_scale": torch.ones(139)},
    ):
        with pytest.raises(ValueError):
            build_normalized_contact_actor_critic(state, torch.zeros(169), torch.ones(169), **extra)
