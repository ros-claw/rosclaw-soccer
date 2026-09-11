"""The new reference channel must preserve the old motor at zero correction."""

import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.goal_reference_actor import (
    ReferenceHeadingEnvelope,
    advance_reference_heading,
    build_goal_reference_actor_critic,
)

torch = pytest.importorskip("torch")


def test_zero_heads_preserve_nonzero_motor_and_critic_exactly():
    torch.manual_seed(69501)
    original = build_ball_residual_actor_critic()
    with torch.no_grad():
        original.actor[-1].weight.normal_(0, 0.1)
        original.actor[-1].bias.normal_(0, 0.1)
    seed = {k: v.detach().clone() for k, v in original.state_dict().items()}
    learner = build_goal_reference_actor_critic(seed)
    observation = torch.randn(32, 136)
    expected, value = original(observation[:, :133].contiguous())
    actual, actual_value = learner(observation)
    assert torch.equal(actual[:, :29], expected)
    assert torch.equal(actual_value, value)
    assert torch.equal(actual[:, 29], torch.zeros(32))
    assert torch.equal(learner.logstd[:29], original.logstd)
    assert all(torch.equal(v, original.state_dict()[k]) for k, v in seed.items())
    actual.sum().backward()
    assert learner.reference_actor[-1].weight.grad.abs().sum() > 0
    assert learner.goal_actor[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_seed_rejected(bad):
    seed = build_ball_residual_actor_critic().state_dict()
    seed["logstd"][0] = bad
    with pytest.raises(ValueError):
        build_goal_reference_actor_critic(seed)


@pytest.mark.parametrize("shape", [(1, 133), (0, 136), (136,)])
def test_invalid_observation_shape(shape):
    learner = build_goal_reference_actor_critic(build_ball_residual_actor_critic().state_dict())
    with pytest.raises(ValueError):
        learner(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 11.0])
def test_invalid_observation_value(value):
    learner = build_goal_reference_actor_critic(build_ball_residual_actor_critic().state_dict())
    obs = torch.zeros(1, 136)
    obs[0, -1] = value
    with pytest.raises(ValueError):
        learner(obs)


def test_dtype_and_envelope_contracts_fail_before_evaluation():
    seed = build_ball_residual_actor_critic().state_dict()
    learner = build_goal_reference_actor_critic(seed)
    with pytest.raises(ValueError):
        learner(torch.zeros(1, 136, dtype=torch.float64))
    with pytest.raises(ValueError):
        build_goal_reference_actor_critic({k: v.double() for k, v in seed.items()})
    with pytest.raises(ValueError):
        advance_reference_heading(torch.zeros(1), torch.tensor([-0.4]), object())


def test_zero_heading_action_keeps_center_and_bounds_hold_over_time():
    previous = torch.full((3,), -0.4)
    assert torch.equal(advance_reference_heading(torch.zeros(3), previous), previous)
    raw = torch.tensor([-1e5, 0.0, 1e5])
    for _ in range(200):
        before = previous.clone()
        current = advance_reference_heading(raw, previous)
        assert torch.equal(previous, before)
        assert bool((abs(current - previous) <= 0.0200001).all())
        assert bool(((current >= -0.6) & (current <= -0.2)).all())
        previous = current


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center_rad": float("nan")},
        {"center_rad": 3.1},
        {"maximum_offset_rad": 0.0},
        {"maximum_step_rad": 0.04},
        {"smoothing": True},
        {"activation_ceiling": "REAL"},
    ],
)
def test_invalid_envelope(kwargs):
    with pytest.raises(ValueError):
        ReferenceHeadingEnvelope(**kwargs)


@pytest.mark.parametrize(
    "raw,previous",
    [
        ([0.0], [-0.4]),
        (torch.zeros(2), torch.zeros(1)),
        (torch.zeros(0), torch.zeros(0)),
        (torch.tensor([float("nan")]), torch.tensor([-0.4])),
        (torch.zeros(1), torch.tensor([-0.7])),
        (torch.zeros(1), torch.tensor([0.0])),
        (torch.zeros(1, dtype=torch.float64), torch.tensor([-0.4])),
    ],
)
def test_invalid_heading_states(raw, previous):
    with pytest.raises(ValueError):
        advance_reference_heading(raw, previous)
