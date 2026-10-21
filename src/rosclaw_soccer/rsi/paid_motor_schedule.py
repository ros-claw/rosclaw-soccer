"""Explicit simulation experiment options; never infer policy choice from case IDs."""


def lifecycle_motor_schedule(
    variant: int, skills: int, override: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    if (
        type(variant) is not int
        or variant not in (0, 1, 2)
        or type(skills) is not int
        or skills not in (1, 3)
    ):
        raise ValueError("declared lifecycle variant and horizon required")
    if override is None:
        return ("base",) + (("student",) if variant == 2 else ("base",)) * (skills - 1)
    if (
        variant != 2
        or type(override) is not tuple
        or len(override) != skills
        or any(type(x) is not str or x not in ("base", "student") for x in override)
    ):
        raise ValueError(
            "explicit immutable base/student motor schedule for neural variant required"
        )
    return override
