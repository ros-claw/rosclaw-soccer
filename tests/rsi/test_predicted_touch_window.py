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


def test_diagnostic_tail_never_extends_football_deadline_but_checks_body():
    p = prediction(n=96, first=0)

    def tail_score():
        return predicted_touch_window(
            p, ball_address=0, root_address=7, start_control=239, diagnostic_tail=True
        )

    assert tail_score()["window_pass"]
    p["qpos"][-1, 9] = 0.64
    assert not tail_score()["predicted_body_healthy"]
    p = prediction(n=96, first=1)
    assert not tail_score()["quality_observed"]  # first240 cannot borrow control300.
    p = prediction(n=96, first=70)
    assert not tail_score()["foot_observed"]  # contact309 isn't football evidence.
    assert not tail_score()["window_pass"]


def test_longer_forecast_observes_existing_formal_quality_without_extending_exam():
    short = predicted_touch_window(
        prediction(96, 38), ball_address=0, root_address=7, start_control=165, diagnostic_tail=True
    )
    long = predicted_touch_window(
        prediction(128, 38), ball_address=0, root_address=7, start_control=165, diagnostic_tail=True
    )
    assert not short["quality_observed"]
    assert long["quality_observed"] and long["first_foot_global_control"] == 203
    late = predicted_touch_window(
        prediction(128, 38), ball_address=0, root_address=7, start_control=202, diagnostic_tail=True
    )
    assert not late["quality_observed"]  # first240 still cannot borrow control300.
