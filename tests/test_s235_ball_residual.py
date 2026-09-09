import pytest

from rosclaw_soccer.training.ball_residual import (
    BallResidualEnvelope,
    advance_ball_residual,
    build_ball_residual_actor_critic,
    precontact_approach_reward,
)


def test_zero_mean_actor_preserves_frozen_teacher_and_has_29_joint_outputs():
    torch = pytest.importorskip("torch")
    actor = build_ball_residual_actor_critic()
    mean, value = actor(torch.zeros((3, 133)))
    assert mean.shape == (3, 29) and value.shape == (3,)
    torch.testing.assert_close(mean, torch.zeros_like(mean))
    output = advance_ball_residual(mean, torch.zeros_like(mean))
    torch.testing.assert_close(output, torch.zeros_like(output))
    for observation in (torch.zeros((1, 132)), torch.full((1, 133), float("nan"))):
        with pytest.raises(ValueError):
            actor(observation)


def test_residual_remains_bounded_under_persistent_extreme_requests():
    torch = pytest.importorskip("torch")
    previous = torch.zeros((2, 29))
    request = torch.full_like(previous, 1000)
    for _ in range(100):
        output = advance_ball_residual(request, previous)
        assert float((output - previous).abs().max()) <= 0.025001
        assert float(output.abs().max()) <= 0.25
        previous = output
    with pytest.raises(ValueError):
        advance_ball_residual(request, previous + 1)
    with pytest.raises(ValueError):
        advance_ball_residual(request * float("nan"), previous)


def test_kicking_ball_away_is_not_penalized_by_approach_shaping_after_contact():
    torch = pytest.importorskip("torch")
    before = torch.tensor([0.4, 0.4, 0.4])
    after = torch.tensor([0.3, 2.0, 2.0])
    mask = torch.tensor([False, True, False])
    reward = precontact_approach_reward(before, after, mask)
    torch.testing.assert_close(reward, torch.tensor([0.4, 0.0, -0.8]))
    torch.testing.assert_close(before, torch.tensor([0.4, 0.4, 0.4]))
    with pytest.raises(ValueError):
        precontact_approach_reward(before, after, mask.float())
    with pytest.raises(ValueError):
        precontact_approach_reward(-before, after, mask)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_offset_rad": 0.36},
        {"maximum_step_rad": 0.031},
        {"smoothing": 0},
        {"smoothing": float("nan")},
        {"activation_ceiling": "REAL"},
    ],
)
def test_residual_envelope_is_simulation_only(kwargs):
    with pytest.raises(ValueError):
        BallResidualEnvelope(**kwargs)
