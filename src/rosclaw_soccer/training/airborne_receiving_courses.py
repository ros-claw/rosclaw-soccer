"""Separate, explicitly unqualified eight-player airborne receiving domain."""

import math
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.role_receiving_courses import ROSTER


@dataclass(frozen=True)
class AirborneReceivingCourse:
    """Initial ball centre and linear velocity in team-forward coordinates."""

    agent_id: str
    seed: int
    position_local_m: tuple[float, float, float]
    velocity_local_mps: tuple[float, float, float]


def airborne_receiving_ball_launch(
    course: AirborneReceivingCourse, *, origin: tuple[float, float, float], radius_m: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Initialize once, never clip, mutate the flight, or grant action authority."""
    if (
        type(course) is not AirborneReceivingCourse
        or course.agent_id not in ROSTER
        or type(course.seed) is not int
        or not 0 <= course.seed < 2**32
        or type(origin) is not tuple
        or len(origin) != 3
        or type(radius_m) not in (float, int)
        or not math.isfinite(radius_m)
        or not 0.03 <= radius_m <= 0.15
    ):
        raise ValueError("typed airborne course, focal player and physical radius required")
    for vector in (origin, course.position_local_m, course.velocity_local_mps):
        if (
            type(vector) is not tuple
            or len(vector) != 3
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector)
        ):
            raise ValueError("finite explicit three-dimensional launch vectors required")
    x, y, z = course.position_local_m
    vx, vy, vz = course.velocity_local_mps
    if (
        not 0.5 <= x <= 2.5
        or abs(y) > 1.5
        or not max(0.20, radius_m) <= z <= 1.5
        or not 0.20 <= math.hypot(vx, vy) <= 3.0
        or not -4.0 <= vz <= 1.0
    ):
        raise ValueError("airborne receiving launch outside fixed SIM domain")
    sign = 1.0 if course.agent_id.startswith("red.") else -1.0
    return ((origin[0] + sign * x, origin[1] + sign * y, z), (sign * vx, sign * vy, vz))


def airborne_receiving_course_bank() -> tuple[AirborneReceivingCourse, ...]:
    """Fixed development cases; not a private or unseen test bank."""
    cases: list[AirborneReceivingCourse] = []
    for agent in ("red.finisher", "blue.playmaker"):
        for sign in (-1.0, 1.0):
            for position, velocity in (
                ((1.6, sign * 0.7, 0.8), (-0.5, -sign * 1.3, -0.8)),
                ((1.2, sign * 0.8, 0.6), (-0.7, -sign * 1.6, -1.1)),
            ):
                cases.append(AirborneReceivingCourse(agent, 96300 + len(cases), position, velocity))
    return tuple(cases)


def preflight_airborne_receiving_courses(courses: tuple[AirborneReceivingCourse, ...]) -> str:
    if type(courses) is not tuple or not 1 <= len(courses) <= 4096:
        raise ValueError("bounded immutable airborne receiving bank required")
    identities, states = set(), set()
    for course in courses:
        airborne_receiving_ball_launch(course, origin=(0.0, 0.0, 0.0), radius_m=0.115)
        identity = (course.agent_id, course.seed)
        state = (course.agent_id, course.position_local_m, course.velocity_local_mps)
        if identity in identities or state in states:
            raise ValueError("duplicate airborne identity or physical launch state")
        identities.add(identity)
        states.add(state)
    return str(
        hash_json(
            dict(
                schema="soccer.airborne_receiving_preflight.v1",
                courses=[asdict(c) for c in courses],
                physics_qualified=False,
                hardware_authorized=False,
            )
        )
    )
