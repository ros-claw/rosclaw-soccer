"""Preregister balanced parallel capture courses and their storage budget."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureCourse:
    index: int
    seed: int
    inward_speed_mps: float

    @property
    def name(self) -> str:
        return f"train-{self.index:03d}"


def capture_campaign_courses(
    *, count: int, first_seed: int, speeds: tuple[float, ...]
) -> tuple[CaptureCourse, ...]:
    """Interleave speed strata before observing outcomes; never select successes."""
    if (
        type(count) is not int
        or not 1 <= count <= 1024
        or type(first_seed) is not int
        or not 0 <= first_seed < 2**32
        or first_seed + count > 2**32
        or type(speeds) is not tuple
        or not speeds
        or any(
            type(speed) not in (int, float) or not math.isfinite(speed) or not 0.5 <= speed <= 3.0
            for speed in speeds
        )
        or len(set(speeds)) != len(speeds)
        or count % len(speeds)
    ):
        raise ValueError("bounded unique seeds and balanced declared return speeds required")
    return tuple(
        CaptureCourse(i, first_seed + i, float(speeds[i % len(speeds)])) for i in range(count)
    )


def require_capture_storage(
    *, free_bytes: int, pending_courses: int, course_budget_bytes: int, reserve_bytes: int
) -> None:
    """Admission budget, not a reservation; workers must recheck actual free space."""
    values = (free_bytes, pending_courses, course_budget_bytes, reserve_bytes)
    if (
        any(type(value) is not int or value < 0 for value in values)
        or course_budget_bytes == 0
        or reserve_bytes == 0
    ):
        raise ValueError("explicit nonnegative storage budget and positive reserve required")
    if free_bytes < pending_courses * course_budget_bytes + reserve_bytes:
        raise ValueError("insufficient persistent storage for the declared pending campaign")
