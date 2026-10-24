"""Separate8-body moving-ball launch domain; never coerce legacy receiving exams."""

import math
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class NativeTeamBallCourse:
    course_id: str
    ball_xy: tuple[float, float]
    ball_velocity_xy: tuple[float, float]


def native_team_course_bank() -> tuple[NativeTeamBallCourse, ...]:
    return tuple(
        NativeTeamBallCourse(name, start, velocity)
        for name, start, velocity in (
            ("outbound-north", (0.0, 0.0), (0.0, 0.8)),
            ("incoming-north", (0.0, 2.0), (0.0, -0.8)),
            ("incoming-south", (0.0, -2.0), (0.0, 0.8)),
            ("incoming-east", (2.0, 0.0), (-0.8, 0.0)),
            ("incoming-west", (-2.0, 0.0), (0.8, 0.0)),
        )
    )


def preflight_native_team_courses(courses: tuple[NativeTeamBallCourse, ...]) -> str:
    if type(courses) is not tuple or not 1 <= len(courses) <= 256:
        raise ValueError("bounded immutable whole native-team course bank required")
    names, states = set(), set()
    for course in courses:
        if (
            type(course) is not NativeTeamBallCourse
            or type(course.course_id) is not str
            or not course.course_id
            or len(course.course_id) > 64
        ):
            raise ValueError("typed native-team ball course required")
        for vector in (course.ball_xy, course.ball_velocity_xy):
            if (
                type(vector) is not tuple
                or len(vector) != 2
                or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector)
            ):
                raise ValueError("finite typed planar launch required")
        if (
            math.hypot(*course.ball_xy) > 2.01
            or not 0.05 <= math.hypot(*course.ball_velocity_xy) < 0.95
            or course.course_id in names
            or (course.ball_xy, course.ball_velocity_xy) in states
        ):
            raise ValueError("unique bounded actual slow-ball course required; no clipping")
        names.add(course.course_id)
        states.add((course.ball_xy, course.ball_velocity_xy))
    return str(
        hash_json(
            dict(
                schema="soccer.native_team_ball_preflight.v1",
                courses=[asdict(x) for x in courses],
                body_count=8,
                ball_radius_m=0.11,
                ball_mass_kg=0.43,
                physics_qualified=False,
                private_fresh_accessed=False,
            )
        )
    )
