import pytest

from rosclaw_soccer.rsi.first_touch_mechanism_expansion import (
    declare_expansion,
    validate_public_course,
)
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan


def plan():
    return PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)


def test_balanced_fixed_perturbations_and_reuse_budget():
    result = declare_expansion(plan(), seed=20261015)
    assert result == declare_expansion(plan(), seed=20261015)
    assert result["strata_counts"] == [10] * 4
    assert sum(r["pilot_adopted"] for r in result["records"]) == 16
    assert result["planned_new_native_executions"] == 105
    assert result["private_fresh_accessed"] is False
    assert result["privileged_teacher_proven"] is False
    for row in result["records"]:
        assert validate_public_course(row["course"]) == validate_public_course(
            row["perturbed_course"]
        )
        assert all(
            abs(a - b) <= bound
            for a, b, bound in zip(
                row["course"], row["perturbed_course"], (0.005, 0.01, 0.025), strict=True
            )
        )


@pytest.mark.parametrize(
    "course",
    [
        (2.5, 0.0, 0.4),
        (2.3, 0.17, 0.4),
        (2.3, 0.0, 0.8),
        (2.3, 0.0, float("nan")),
        (2, 0.0, 0.4),
        [2.3, 0.0, 0.4],
    ],
)
def test_invalid_course_rejects_before_allocation(course):
    with pytest.raises(ValueError):
        validate_public_course(course)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_invalid_generator_identity(seed):
    with pytest.raises(ValueError):
        declare_expansion(plan(), seed=seed)
