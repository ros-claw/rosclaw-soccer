"""Outcome-independent diagnostic subset of the now-consumed near96 population."""


def teacher_coverage_indices() -> tuple[int, ...]:
    # Four fixed ranks per stratum, not selected by success/failure or force.
    return tuple(16 * group + rank for group in range(6) for rank in (0, 5, 10, 15))
