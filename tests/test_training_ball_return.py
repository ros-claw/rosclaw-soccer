from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.world.training_ball_return import (
    TrainingBallReturnConfig,
    TrainingBallReturnReferee,
)


def referee(**kwargs):
    return TrainingBallReturnReferee(
        TrainingBallReturnConfig(**kwargs),
        left_x=-1.5,
        right_x=4.5,
        radius=0.115,
        goal_width=3.0,
        goal_height=2.0,
    )


def observe(r, t, ball=(0.0, 3.2, 0.115), players=((0.0, 0.0),)):
    return r.observe(time_sec=t, ball_position_m=ball, player_positions_xy_m=players)


def test_exit_wait_release_reentry_and_budget():
    r = referee(maximum_returns=1)
    assert observe(r, 0.0, (0.0, 3.1, 0.115)) is None  # Not whole-ball exit.
    assert observe(r, 0.1).code == 1
    assert observe(r, 1.0) is None
    event = observe(r, 1.2)
    assert event.code == 2 and event.return_count == 1
    assert event.release_position_m[1] > 3.115
    assert event.release_velocity_mps == (0.0, -2.0, 1.8)
    assert observe(r, 1.3, event.release_position_m) is None
    assert observe(r, 1.5, (0.0, 2.8, 0.7)).code == 3
    assert observe(r, 2.0).code == 1
    assert observe(r, 4.0) is None
    assert r.return_count == 1


@pytest.mark.parametrize("ball", [(0.0, -3.2, 0.115), (-1.7, 0.0, 0.8), (4.7, 2.0, 0.8)])
def test_return_is_outside_and_inward_for_every_exit(ball):
    r = referee()
    assert observe(r, 0.0, ball).code == 1
    e = observe(r, 1.0, ball)
    assert e.code == 2
    assert -0.75 <= e.release_position_m[0] <= 3.75
    assert abs(e.release_position_m[1]) > 3.115
    assert e.release_position_m[1] * e.release_velocity_mps[1] < 0


def test_no_spawn_inside_player_and_no_consumed_budget():
    r = referee()
    observe(r, 0.0)
    blocked = tuple((x, y) for x in (0.0, 1.0, -0.75, 2.0) for y in (-3.365, 3.365))
    assert observe(r, 1.0, players=blocked) is None
    assert r.return_count == 0
    assert observe(r, 2.0).code == 2


def test_failed_throw_does_not_loop_or_count_as_reentry():
    r = referee(maximum_returns=2)
    observe(r, 0.0)
    e = observe(r, 1.0)
    assert observe(r, 3.0, e.release_position_m) is None
    assert observe(r, 4.1, e.release_position_m).code == 1
    assert observe(r, 5.2, e.release_position_m).code == 2
    assert r.return_count == 2


@pytest.mark.parametrize("t", [-1.0, float("nan"), float("inf"), True])
def test_invalid_clock_does_not_change_state(t):
    r = referee()
    with pytest.raises(ValueError):
        observe(r, t)
    assert observe(r, 0.0).code == 1


def test_nonfinite_or_stale_observation_fails_closed():
    r = referee()
    with pytest.raises(ValueError):
        observe(r, 0.0, (float("nan"), 0.0, 0.115))
    assert observe(r, 0.0).code == 1
    with pytest.raises(ValueError):
        observe(r, 0.0)
    assert r.return_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(maximum_returns=True),
        dict(maximum_returns=0),
        dict(maximum_returns=11),
        dict(delay_sec=0),
        dict(inward_speed_mps=4),
        dict(upward_speed_mps=-1),
        dict(release_height_m=0.1),
        dict(player_clearance_m=0.1),
        dict(delay_sec=float("nan")),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        referee(**kwargs)


def test_default_identity_retained_and_assistance_explicit():
    default = IndependentTeamWorldConfig()
    assert (
        default.config_hash
        == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    new = replace(
        default,
        bilateral_goals=True,
        training_ball_return=TrainingBallReturnConfig(),
        simulation_duration_sec=45.0,
    )
    assert new.config_hash != default.config_hash
    for kwargs in (
        dict(bilateral_goals=False),
        dict(stop_on_ball_exit=True),
        dict(training_ball_return=True),
        dict(simulation_duration_sec=61.0),
    ):
        with pytest.raises(ValueError):
            replace(new, **kwargs)
    with pytest.raises(ValueError):
        replace(default, simulation_duration_sec=45.0)


def test_replay_and_private_referee_state():
    a, b = referee(), referee()
    for t, ball in ((0.0, (0.0, 3.2, 0.1)), (1.0, (0.0, 3.4, 0.1)), (2.0, (0.0, 2.0, 0.2))):
        assert observe(a, t, ball) == observe(b, t, ball)
    assert referee().return_count == 0
