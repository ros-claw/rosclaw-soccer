from dataclasses import replace

import pytest

from rosclaw_soccer.training.native_team_courses import NativeTeamBallCourse
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.stadium_team_courses import (
    StadiumTeamBallCourse,
    preflight_stadium_team_courses,
)


def course() -> StadiumTeamBallCourse:
    return StadiumTeamBallCourse(
        course_id="case14-x-plus",
        agent_id="blue.playmaker",
        ball_xy=(4.0028465609229055, 1.759927110724756),
        ball_velocity_xy=(0.0759961033042938, -0.08061739444238553),
        seed=1928,
        left_x=-1.5,
        right_x=7.5,
        half_width=3.0,
        ball_radius_m=0.11,
        ball_mass_kg=0.43,
    )


def test_stadium_domain_is_explicit_and_routes_through_shared_preflight() -> None:
    bank = (course(), replace(course(), course_id="case14-x-minus", ball_xy=(3.90, 1.76)))
    assert preflight_receiving_courses(bank) == preflight_stadium_team_courses(bank)
    assert preflight_receiving_courses(bank).startswith("sha256:")
    assert preflight_receiving_courses(bank) != preflight_receiving_courses(bank[::-1])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ball_xy", (7.391, 0.0)),
        ("ball_xy", (-1.391, 0.0)),
        ("ball_xy", (0.0, 2.891)),
        ("ball_xy", (float("nan"), 0.0)),
        ("ball_xy", (10**1000, 0.0)),
        ("ball_velocity_xy", (float("inf"), 0.0)),
        ("ball_velocity_xy", (5.01, 0.0)),
        ("ball_radius_m", 0.0),
        ("half_width", 4.0),
        ("ball_mass_kg", True),
        ("seed", True),
        ("course_id", "../outside"),
        ("agent_id", ""),
    ],
)
def test_rejects_invalid_course_without_clipping(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        preflight_receiving_courses((replace(course(), **{field: value}),))


def test_rejects_duplicate_physical_state_even_with_new_id_or_seed() -> None:
    with pytest.raises(ValueError, match="unique"):
        preflight_receiving_courses((course(), replace(course(), course_id="other", seed=7)))


def test_rejects_profile_drift_for_entire_bank() -> None:
    second = replace(course(), course_id="other", ball_xy=(4.1, 1.7), ball_mass_kg=0.4)
    with pytest.raises(ValueError, match="profile"):
        preflight_receiving_courses((course(), second))


def test_rejects_mixed_native_and_stadium_domains() -> None:
    native = NativeTeamBallCourse("legacy", (0.0, 0.0), (0.1, 0.0))
    with pytest.raises(ValueError):
        preflight_receiving_courses((course(), native))  # type: ignore[arg-type]


def test_accepts_stationary_ball_and_exact_whole_ball_boundaries() -> None:
    border = replace(course(), ball_xy=(7.39, 2.89), ball_velocity_xy=(0.0, 0.0))
    assert preflight_receiving_courses((border,)).startswith("sha256:")
