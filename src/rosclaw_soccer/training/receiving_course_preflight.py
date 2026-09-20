"""Validate the complete receiving course declaration before allocating workers.

This is input validation, not physical qualification, split isolation, or
evidence of independent samples. Replays belong to execution, not this bank.
"""

from dataclasses import asdict

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse, receiving_ball_launch


def preflight_receiving_courses(courses: tuple[ReceivingCourse, ...]) -> str:
    """Return a content binding only after every course passes the launch domain.

    Using a normalized origin/radius here validates course parameters only.
    The actual scene, dimensions, ownership and replay budget still require
    their own checks. No course is clipped, replaced, dropped, or made easier.
    """
    if type(courses) is not tuple or not 1 <= len(courses) <= 4096:
        raise ValueError("bounded immutable receiving course bank required")
    identities = set()
    for course in courses:
        if (
            type(course) is not ReceivingCourse
            or type(course.agent_id) is not str
            or type(course.seed) is not int
            or not 0 <= course.seed < 2**32
            or type(course.speed_mps) not in (int, float)
            or type(course.lateral_m) not in (int, float)
        ):
            raise ValueError("typed receiving course and uint32 seed required")
        receiving_ball_launch(course, origin=(0.0, 0.0, 0.0), radius_m=0.11)
        identity = (course.agent_id, course.seed)
        if identity in identities:
            raise ValueError("duplicate player/seed; paired executions are not new courses")
        identities.add(identity)
    return str(
        hash_json(
            {
                "schema": "soccer.receiving_course_preflight.v1",
                "courses": [asdict(course) for course in courses],
                "physics_qualified": False,
            }
        )
    )
