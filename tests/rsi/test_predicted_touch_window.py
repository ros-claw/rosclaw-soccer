import numpy as np
import pytest

from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window


def prediction(n=64, first=2):
    q = np.zeros((n * 10, 43), np.float64)
    q[:, 9] = 0.8
    q[:, 10] = 1.0
    q[:, 3] = 1.0
    q[:, 0] = np.repeat(np.arange(n) * 0.02, 10)
    force = np.zeros((n * 10, 1, 3), np.float64)
    if first is not None:
        force[first * 10, 0, 0] = 2.0
    return dict(qpos=q, ball_forces=force, controls=n)


def score(p, start=100):
    return predicted_touch_window(p, ball_address=0, root_address=7, start_control=start)


def test_complete_observed_window_is_not_full_episode_certificate():
    s = score(prediction())
    assert s["window_pass"] and s["quality_observed"]
    assert s["first_foot_global_control"] == 102
    assert s["forward_60_m"] == pytest.approx(1.2)
    assert s["full_h300_task_pass_proven"] is False


def test_missing_quality_never_fabricated_even_when_ball_moves_far():
    p = prediction(first=4)
    p["qpos"][:, 0] *= 100
    s = score(p)
    assert not s["window_pass"] and not s["quality_observed"]
    assert s["forward_60_m"] is None


@pytest.mark.parametrize("kind", ["nonfoot", "fall", "tilt", "lateral"])
def test_future_unsafe_or_bad_ball_cannot_buy_pass(kind):
    p = prediction()
    if kind == "nonfoot":
        p["ball_forces"][-1, 0, 2] = 2.0
    elif kind == "fall":
        p["qpos"][-1, 9] = 0.64
    elif kind == "tilt":
        p["qpos"][-1, 10:14] = [np.sqrt(0.9), np.sqrt(0.1), 0.0, 0.0]
    else:
        p["qpos"][:, 1] = 5.0
    assert not score(p)["window_pass"]


def test_deadline_and_early_physical_termination_are_not_padded():
    assert score(prediction(n=61, first=0), start=239)["quality_observed"]
    with pytest.raises(ValueError):
        score(prediction(), start=240)
    s = score(prediction(n=8, first=0))
    assert s["forward_60_m"] is None and not s["window_pass"]
