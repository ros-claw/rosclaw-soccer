"""Public diagnostic40, with distinct ball positions and explicit rolling cases.

Not the original far-ball RSI exam, a blind holdout, or forty independent
randomized environments. Seeds identify deterministic launch states only.
"""

from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json

X_POSITIONS = (0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58)
Y_POSITIONS = (-0.24, -0.17, -0.10, -0.03, 0.04)


@dataclass(frozen=True)
class ExpandedEntryCourse:
    agent_id: str
    seed: int
    course_index: int
    course: tuple[float, float, float]


def expanded_entry_courses() -> tuple[ExpandedEntryCourse, ...]:
    rows: list[ExpandedEntryCourse] = []
    moving = 0
    for i, x in enumerate(X_POSITIONS):
        for j, y in enumerate(Y_POSITIONS):
            speed = 0.0
            if (i + j) % 2:
                speed = -0.15 if moving % 2 == 0 else 0.15
                moving += 1
            k = len(rows)
            rows.append(ExpandedEntryCourse("tracker", 202610240 + k, k, (x, y, speed)))
    return tuple(rows)


def preflight_expanded_entry_bank(courses: tuple[ExpandedEntryCourse, ...]) -> str:
    expected = expanded_entry_courses()
    if type(courses) is not tuple or len(courses) != 40:
        raise ValueError("entire40 expanded entry bank required")
    for item, gold in zip(courses, expected, strict=True):
        if (
            type(item) is not ExpandedEntryCourse
            or type(item.agent_id) is not str
            or type(item.seed) is not int
            or type(item.course_index) is not int
            or type(item.course) is not tuple
            or any(type(x) is not float for x in item.course)
            or item != gold
        ):
            raise ValueError("exact ordered public40 course required; no clipping or replacement")
    return hash_json(
        dict(
            schema="soccer.public_expanded_entry_course_preflight.v1",
            courses=[asdict(c) for c in courses],
            distinct_ball_positions=40,
            stationary=20,
            approaching=10,
            receding=10,
            seeds_are_identity_only=True,
            original_rsi_exam=False,
            private_fresh_accessed=False,
            physics_qualified=False,
            hardware_authorized=False,
        )
    )
