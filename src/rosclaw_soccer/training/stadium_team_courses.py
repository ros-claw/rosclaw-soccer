"""Explicit small-sided stadium launch courses for SIM receiving research.

The legacy native-team bank has a radius-two-metre launch domain. Stadium
coordinates are a different domain and must never be silently reinterpreted as
that bank. The present match boundary uses a fixed 3m half-width. This module
validates declarations, not compiled geometry, physical skill or safety.
"""

import math
import re
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.world.match_boundary import require_in_play_ball_launch


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class StadiumTeamBallCourse:
    """One world-coordinate ball state, role and explicit stadium/ball profile."""

    course_id: str
    agent_id: str
    ball_xy: tuple[float, float]
    ball_velocity_xy: tuple[float, float]
    seed: int
    left_x: float
    right_x: float
    half_width: float
    ball_radius_m: float
    ball_mass_kg: float


def preflight_stadium_team_courses(courses: tuple[StadiumTeamBallCourse, ...]) -> str:
    """Bind every valid course; never clip or discard a failed bank member."""
    if type(courses) is not tuple or not 1 <= len(courses) <= 256:
        raise ValueError("bounded immutable stadium course bank required")
    ids: set[str] = set()
    states: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    profile: tuple[float, ...] | None = None
    for course in courses:
        if (
            type(course) is not StadiumTeamBallCourse
            or type(course.course_id) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}", course.course_id) is None
            or type(course.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", course.agent_id) is None
            or type(course.seed) is not int
            or not 0 <= course.seed < 2**32
        ):
            raise ValueError("typed stadium course, player and seed required")
        for vector in (course.ball_xy, course.ball_velocity_xy):
            if (
                type(vector) is not tuple
                or len(vector) != 2
                or any(not _finite_number(value) for value in vector)
            ):
                raise ValueError("finite ordinary planar launch vectors required")
        geometry = (
            course.left_x,
            course.right_x,
            course.half_width,
            course.ball_radius_m,
            course.ball_mass_kg,
        )
        if (
            any(not _finite_number(value) for value in geometry)
            or not 0.01 <= course.ball_radius_m <= 0.5
            or not 0.01 <= course.ball_mass_kg <= 2.0
            or course.half_width != 3.0
            or max(abs(course.left_x), abs(course.right_x), abs(course.half_width)) > 1000
            or math.hypot(*course.ball_velocity_xy) > 5.0
        ):
            raise ValueError("finite bounded stadium, ball and launch velocity required")
        if profile is None:
            profile = tuple(float(value) for value in geometry)
        elif profile != tuple(float(value) for value in geometry):
            raise ValueError("one explicit stadium and ball profile per bank required")
        require_in_play_ball_launch(
            (course.ball_xy[0], course.ball_xy[1], course.ball_radius_m),
            left_x=course.left_x,
            right_x=course.right_x,
            half_width=course.half_width,
            radius=course.ball_radius_m,
        )
        state = (course.ball_xy, course.ball_velocity_xy)
        if course.course_id in ids or state in states:
            raise ValueError("unique stadium course IDs and physical launches required")
        ids.add(course.course_id)
        states.add(state)
    return str(
        hash_json(
            {
                "schema": "soccer.stadium_team_course_preflight.v1",
                "courses": [asdict(course) for course in courses],
                "simulation_only": True,
                "physics_qualified": False,
                "hardware_authorized": False,
                "promotion_eligible": False,
            }
        )
    )
