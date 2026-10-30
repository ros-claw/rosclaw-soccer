import numpy as np
import pytest

from rosclaw_soccer.training.pass_impulse_reward import sampled_pass_impulse_reward


def test_passing_alignment_and_contact_gate():
    base = np.zeros((4, 2))
    target = np.tile((2.0, 0.0), (4, 1))
    after = np.array([[1.0, 0.0], [1.0, 2.0], [-1.0, 0.0], [1.0, 0.0]])
    reward = sampled_pass_impulse_reward(
        ball_xy=base,
        target_xy=target,
        velocity_before_xy=base,
        velocity_after_xy=after,
        valid_contact=np.array([True, True, True, False]),
    )
    np.testing.assert_allclose(reward, [0.04, -0.04, -0.04, 0.0])


def test_preexisting_velocity_and_degenerate_target_give_no_credit():
    v = np.array([[3.0, 1.0], [3.0, 1.0]])
    result = sampled_pass_impulse_reward(
        ball_xy=np.zeros((2, 2)),
        target_xy=np.array([[1.0, 0.0], [0.0, 0.0]]),
        velocity_before_xy=v,
        velocity_after_xy=v + np.array([[0.0, 0.0], [1.0, 0.0]]),
        valid_contact=np.ones(2, dtype=bool),
    )
    np.testing.assert_array_equal(result, np.zeros(2))


@pytest.mark.parametrize(
    "field,bad",
    [("valid_contact", np.ones(2)), ("target_xy", np.array([[float("nan"), 0.0], [0.0, 0.0]]))],
)
def test_invalid_telemetry_fails_closed(field, bad):
    args = dict(
        ball_xy=np.zeros((2, 2)),
        target_xy=np.ones((2, 2)),
        velocity_before_xy=np.zeros((2, 2)),
        velocity_after_xy=np.zeros((2, 2)),
        valid_contact=np.ones(2, dtype=bool),
    )
    args[field] = bad
    with pytest.raises(ValueError):
        sampled_pass_impulse_reward(**args)
