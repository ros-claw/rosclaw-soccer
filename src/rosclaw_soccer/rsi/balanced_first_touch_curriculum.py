"""Public, balanced course declarations; not rollout admission or skill evidence.

Keep the original generator intact. Each seed contributes one course in each
of the four x-side/velocity-direction strata. Rotate the quartet by seed order
to cover all sixteen lanes, rather than repeating only incoming lower-x lanes.
No private pool, outcomes, model selection, physics or optimizer is accessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses


@dataclass(frozen=True, slots=True)
class PublicFirstTouchJob:
    """One declared draw. Identity and course are immutable public inputs."""

    group: int
    seed: int
    lane: int
    repeat: int
    course: tuple[float, float, float]


def balanced_public_first_touch_jobs(
    seeds: tuple[int, ...], *, repeats: int = 4
) -> tuple[PublicFirstTouchJob, ...]:
    """Declare a complete context-major bank with equal four-stratum counts.

    Seed order is part of the declaration: seed i uses lanes
    (2*(i%4), 2*(i%4)+1, 8+2*(i%4), 9+2*(i%4)). All draws, including future
    failures, must be retained by a separately admitted collector. This pure
    function grants neither collection permission nor global Fresh separation.
    """
    if (
        type(seeds) is not tuple
        or not 1 <= len(seeds) <= 64
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or type(repeats) is not int
        or not 1 <= repeats <= 16
    ):
        raise ValueError("unique bounded public seeds and bounded repeat count required")
    jobs: list[PublicFirstTouchJob] = []
    for index, seed in enumerate(seeds):
        courses = sample_training_courses(seed)
        first = 2 * (index % 4)
        for lane in (first, first + 1, first + 8, first + 9):
            course = courses[lane]
            for repeat in range(repeats):
                jobs.append(PublicFirstTouchJob(len(jobs), seed, lane, repeat, course))
    return tuple(jobs)
