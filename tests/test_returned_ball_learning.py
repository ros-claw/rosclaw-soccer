import numpy as np
import pytest

from rosclaw_soccer.training.returned_ball_learning import returned_ball_segments


def trace(codes):
    return dict(
        time=np.arange(len(codes), dtype=float) * 0.02 + 0.02,
        training_return_event_code=np.array(codes),
        observations=np.ones((len(codes), 8, 56)),
    )


def test_only_observed_live_epochs_are_learning_samples():
    value = trace([0, 1, 0, 2, 0, 3, 0, 1, 2, 3, 0])
    parts = returned_ball_segments(value)
    assert [p["time"].tolist() for p in parts] == [
        value["time"][5:7].tolist(),
        value["time"][9:].tolist(),
    ]
    parts[0]["observations"][0, 0, 0] = 99
    assert value["observations"][5, 0, 0] == 1


def test_failed_throw_and_no_reentry_give_no_reward_segment():
    assert returned_ball_segments(trace([0, 1, 2, 0, 1, 2, 0])) == []
    assert returned_ball_segments(trace([0, 0, 0])) == []


@pytest.mark.parametrize("codes", [[3], [2], [1, 3], [1, 1], [1, 2, 2], [1, 2, 3, 3], [4]])
def test_malformed_lifecycle_is_rejected(codes):
    with pytest.raises(ValueError):
        returned_ball_segments(trace(codes))


def test_clock_and_frame_alignment_required():
    value = trace([1, 2, 3, 0])
    value["observations"] = np.zeros((3, 8, 56))
    with pytest.raises(ValueError):
        returned_ball_segments(value)
    for times in ([0.02, 0.02, 0.06, 0.08], [0.02, float("nan"), 0.06, 0.08]):
        value = trace([1, 2, 3, 0])
        value["time"] = np.array(times)
        with pytest.raises(ValueError):
            returned_ball_segments(value)


def test_noninteger_codes_rejected():
    value = trace([1, 2, 3])
    value["training_return_event_code"] = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        returned_ball_segments(value)


@pytest.mark.parametrize("codes", [[1, 2, 3, 0], [0, 0], [3, 0, 3], [3, 1]])
def test_reward_rejects_unsegmented_or_reset_bearing_coach_trace(codes):
    from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards

    with pytest.raises(ValueError, match="observed returned-live segment"):
        physical_rewards(trace(codes), tuple(f"agent.{i}" for i in range(8)))
