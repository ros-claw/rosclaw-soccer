"""Role-balanced fresh rehearsal and harder receiving classrooms.

Retain the old geometries, not their old actions: callers must collect new
on-policy trajectories from the selected research parent. These public courses
are not an unseen holdout and do not grant match qualification or activation.
"""

import math
from collections import Counter
from dataclasses import replace

from rosclaw_soccer.training.role_receiving_courses import ROSTER, ReceivingCourse


def mixed_receiving_courses(
    retention_courses: tuple[ReceivingCourse, ...],
    *,
    first_seed: int,
    hard_first_seed: int,
    hard_speeds_mps: tuple[float, ...] = (0.85, 1.0, 1.15, 1.25),
    hard_lateral_m: tuple[float, ...] = (-0.08, 0.08),
) -> tuple[ReceivingCourse, ...]:
    """Copy every retained geometry, followed by an eight-role hard grid.

    Original order is preserved. Fresh uint32 streams cannot overlap one
    another or reuse any supplied reference seed. Equal role counts alone are
    insufficient: each role must rehearse the same multiset of geometries.
    """
    if (
        type(retention_courses) is not tuple
        or not 8 <= len(retention_courses) <= 4096
        or type(first_seed) is not int
        or type(hard_first_seed) is not int
    ):
        raise ValueError("bounded explicit rehearsal courses and integer seeds required")
    geometries: dict[str, Counter[tuple[float, float]]] = {a: Counter() for a in ROSTER}
    old_seeds: set[int] = set()
    for course in retention_courses:
        if (
            not isinstance(course, ReceivingCourse)
            or not isinstance(course.agent_id, str)
            or course.agent_id not in ROSTER
            or type(course.seed) is not int
            or not 0 <= course.seed < 2**32
            or course.seed in old_seeds
            or type(course.speed_mps) not in (int, float)
            or not math.isfinite(course.speed_mps)
            or not 0.5 <= course.speed_mps <= 3.0
            or type(course.lateral_m) not in (int, float)
            or not math.isfinite(course.lateral_m)
            or abs(course.lateral_m) > 0.3
        ):
            raise ValueError("unique typed finite receiving reference courses required")
        old_seeds.add(course.seed)
        geometries[course.agent_id][course.speed_mps, course.lateral_m] += 1
    if not geometries[ROSTER[0]] or any(
        value != geometries[ROSTER[0]] for value in geometries.values()
    ):
        raise ValueError("all eight roles must rehearse identical geometry coverage")
    for values, low, high in ((hard_speeds_mps, 0.5, 3.0), (hard_lateral_m, -0.3, 0.3)):
        if (
            type(values) is not tuple
            or not 1 <= len(values) <= 8
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high
                for v in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError("finite bounded distinct hard-course axes required")
    hard_count = 8 * len(hard_speeds_mps) * len(hard_lateral_m)
    if not (
        0 <= first_seed <= 2**32 - len(retention_courses)
        and 0 <= hard_first_seed <= 2**32 - hard_count
    ):
        raise ValueError("fresh course streams must fit uint32")
    retained_seeds = set(range(first_seed, first_seed + len(retention_courses)))
    hard_seeds = set(range(hard_first_seed, hard_first_seed + hard_count))
    if retained_seeds & hard_seeds or (retained_seeds | hard_seeds) & old_seeds:
        raise ValueError("rehearsal, hard and reference seed streams must not overlap")
    retained = tuple(replace(c, seed=first_seed + i) for i, c in enumerate(retention_courses))
    harder = tuple(
        ReceivingCourse(agent, hard_first_seed + i, speed, lateral)
        for i, (speed, lateral, agent) in enumerate(
            (s, side, a) for s in hard_speeds_mps for side in hard_lateral_m for a in ROSTER
        )
    )
    return retained + harder
