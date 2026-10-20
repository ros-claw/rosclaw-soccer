"""Validate the complete receiving course declaration before allocating workers.

This is input validation, not physical qualification, split isolation, or
evidence of independent samples. Replays belong to execution, not this bank.
"""

import re
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.close_contact_courses import (
    CloseContactCourse,
    preflight_close_contact_bank,
)
from rosclaw_soccer.training.expanded_entry_courses import (
    ExpandedEntryCourse,
    preflight_expanded_entry_bank,
)
from rosclaw_soccer.training.motion_portfolio_holdout import (
    MotionPortfolioHoldoutCourse,
    preflight_motion_portfolio_holdout,
)
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse, receiving_ball_launch


@dataclass(frozen=True)
class FirstTouchCourse:
    """Explicit RSI launch domain, not the legacy0.9m receiving classroom."""

    agent_id: str
    seed: int
    course: tuple[float, float, float]


def preflight_receiving_courses(
    courses: tuple[ReceivingCourse, ...]
    | tuple[FirstTouchCourse, ...]
    | tuple[CloseContactCourse, ...]
    | tuple[ExpandedEntryCourse, ...]
    | tuple[MotionPortfolioHoldoutCourse, ...],
    # Separately named near holdout never coerces the original RSI/legacy domains.
) -> str:
    """Return a content binding only after every course passes the launch domain.

    Using a normalized origin/radius here validates course parameters only.
    The actual scene, dimensions, ownership and replay budget still require
    their own checks. No course is clipped, replaced, dropped, or made easier.
    The explicit FirstTouchCourse path checks the RSI x/y/v launch domain;
    it never coerces it into the legacy0.9m ReceivingCourse classroom. Legacy
    receiving inputs keep their original launch rules and content-hash schema.
    """
    if type(courses) is not tuple or not 1 <= len(courses) <= 4096:
        raise ValueError("bounded immutable receiving course bank required")
    if all(type(course) is MotionPortfolioHoldoutCourse for course in courses):
        return preflight_motion_portfolio_holdout(courses)  # type: ignore[arg-type]
    if all(type(course) is CloseContactCourse for course in courses):
        return preflight_close_contact_bank(courses)  # type: ignore[arg-type]
    if all(type(course) is ExpandedEntryCourse for course in courses):
        return preflight_expanded_entry_bank(courses)  # type: ignore[arg-type]
    if all(type(course) is FirstTouchCourse for course in courses):
        from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course

        identities, states = set(), set()
        for item in courses:
            assert isinstance(item, FirstTouchCourse)
            if (
                type(item.agent_id) is not str
                or not re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", item.agent_id)
                or type(item.seed) is not int
                or not 0 <= item.seed < 2**32
            ):
                raise ValueError("typed public first-touch identity and uint32 seed required")
            validate_public_course(item.course)
            identity, state = (item.agent_id, item.seed), (item.agent_id, item.course)
            if identity in identities or state in states:
                raise ValueError("duplicate first-touch identity/state; arms are not new courses")
            identities.add(identity)
            states.add(state)
        return str(
            hash_json(
                dict(
                    schema="soccer.public_first_touch_course_preflight.v1",
                    courses=[asdict(item) for item in courses],
                    public_only=True,
                    private_fresh_accessed=False,
                    physics_qualified=False,
                )
            )
        )
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
