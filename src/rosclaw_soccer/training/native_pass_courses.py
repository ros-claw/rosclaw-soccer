"""Preflight actual-pass courses, not a coerced synthetic ball launcher."""

import math
import re
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class NativePassCourse:
    context_index: int
    receiver_forward_m: float
    receiver_lateral_m: float
    initial_world_state_hash: str


def preflight_native_pass_courses(courses: tuple[NativePassCourse, ...]) -> str:
    if type(courses) is not tuple or not 1 <= len(courses) <= 4096:
        raise ValueError("bounded immutable actual-pass bank required")
    states, identities = set(), set()
    for c in courses:
        if (
            type(c) is not NativePassCourse
            or type(c.context_index) is not int
            or not 0 <= c.context_index < 64
            or any(
                type(x) is not float or not math.isfinite(x)
                for x in (c.receiver_forward_m, c.receiver_lateral_m)
            )
            or not 2 <= c.receiver_forward_m <= 8
            or not -1.5 <= c.receiver_lateral_m <= 1.5
            or type(c.initial_world_state_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", c.initial_world_state_hash) is None
        ):
            raise ValueError(
                "typed native-pass source, receiver domain and initial state binding required"
            )
        identity = (c.context_index, c.receiver_forward_m, c.receiver_lateral_m)
        if identity in identities or c.initial_world_state_hash in states:
            raise ValueError("duplicate actual-pass course/state; arms are not additional courses")
        identities.add(identity)
        states.add(c.initial_world_state_hash)
    return str(
        hash_json(
            dict(
                schema="soccer.native_pass_course_preflight.v1",
                courses=[asdict(c) for c in courses],
                physics_qualified=False,
                synthetic_ball_launcher=False,
                evidence_domain="SIM_ONLY",
            )
        )
    )
