import pytest

from rosclaw_soccer.training.shot_precision_reward import (
    ShotPrecisionRewardConfig,
    terminal_shot_precision_penalty,
)


def penalty(error, clean=True, config=None):
    return terminal_shot_precision_penalty(
        measured_crossing_error_m=error, complete_clean_episode=clean, config=config
    )


def test_actual_accuracy_has_strict_feedback_inside_declared_cap():
    errors = [0, 0.1, 0.5, 1.0, 1.635, 2.5, 4]
    values = [penalty(x) for x in errors]
    assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))
    assert penalty(0) == 0
    assert penalty(1.635) == pytest.approx(-6.54)
    assert penalty(100) == penalty(4) == -16


@pytest.mark.parametrize("error", [None, 0, 0.1, 2, 10])
def test_missing_or_unclean_episode_cannot_get_accuracy_credit(error):
    assert penalty(error, False) == -16
    if error is None:
        assert penalty(error) == -16


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -0.1, True, "1.0"])
def test_invalid_error_rejected_even_when_failure_would_be_capped(error):
    for clean in (False, True):
        with pytest.raises(ValueError):
            penalty(error, clean)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(error_weight_per_m=0),
        dict(error_weight_per_m=21),
        dict(error_weight_per_m=True),
        dict(missing_or_failed_error_m=0),
        dict(missing_or_failed_error_m=float("nan")),
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ShotPrecisionRewardConfig(**kwargs)


def test_wrong_flags_and_config_rejected():
    with pytest.raises(ValueError):
        penalty(1, 1)
    with pytest.raises(ValueError):
        penalty(1, config=False)


def test_potential_telescopes_but_terminal_error_does_not():
    gamma = 0.995
    trajectories = ([0, 0.2, 3, 2], [0, 2, 0.1, 2])
    shaped_returns = []
    for phi in trajectories:
        rewards = [
            gamma * following - current
            for current, following in zip(phi[:-1], phi[1:], strict=True)
        ]
        shaped_returns.append(sum(gamma**t * r for t, r in enumerate(rewards)))
    assert shaped_returns[0] == pytest.approx(shaped_returns[1])
    assert shaped_returns[0] == pytest.approx(gamma**3 * 2)
    # Same final potential, but now a measured closer crossing is preferable.
    assert shaped_returns[0] + gamma**2 * penalty(0.2) > shaped_returns[1] + gamma**2 * penalty(1.6)
