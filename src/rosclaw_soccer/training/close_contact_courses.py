"""Separate public close-ball diagnostic bank; never the original RSI exam."""

import math
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json

BASES = ((0.35, -0.10, 0.0), (0.50, -0.10, 0.0), (0.35, -0.22, 0.0), (0.50, -0.22, 0.0))
OFFSETS = ((-0.01, 0.0), (0.01, 0.0), (0.0, -0.01), (0.0, 0.01))


@dataclass(frozen=True)
class CloseContactCourse:
    agent_id: str
    seed: int
    parent_index: int
    variant_index: int
    course: tuple[float, float, float]


def close_contact_perturbations() -> tuple[CloseContactCourse, ...]:
    """Exactly16 declared variants, four already-consumed parent clusters."""
    return tuple(
        CloseContactCourse("tracker", 202610200 + 4 * p + v, p, v, (x + dx, y + dy, 0.0))
        for p, (x, y, _) in enumerate(BASES)
        for v, (dx, dy) in enumerate(OFFSETS)
    )


def preflight_close_contact_bank(courses: tuple[CloseContactCourse, ...]) -> str:
    expected = close_contact_perturbations()
    if type(courses) is not tuple or len(courses) != 16:
        raise ValueError("entire16-condition close-contact diagnostic bank required")
    for item, gold in zip(courses, expected, strict=True):
        if (
            type(item) is not CloseContactCourse
            or type(item.agent_id) is not str
            or any(type(x) is not int for x in (item.seed, item.parent_index, item.variant_index))
            or type(item.course) is not tuple
            or len(item.course) != 3
            or any(type(x) is not float or not math.isfinite(x) for x in item.course)
            or item != gold
        ):
            raise ValueError(
                "exact declared static close-contact variant and unique ordered identity required"
            )
    return hash_json(
        dict(
            schema="soccer.public_close_contact_course_preflight.v1",
            courses=[asdict(x) for x in courses],
            independent_parent_clusters=4,
            perturbations_per_parent=4,
            seeds_are_identity_only_not_randomized_motion=True,
            public_diagnostic_only=True,
            original_rsi_exam=False,
            private_fresh_accessed=False,
            physics_qualified=False,
            hardware_authorized=False,
        )
    )
