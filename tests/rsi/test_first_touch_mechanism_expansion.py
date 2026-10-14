import pytest

from rosclaw_soccer.rsi.first_touch_mechanism_expansion import (
    declare_expansion,
    select_pilot_mechanism,
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


def pilot():
    rows = []
    for arm in ("lateral", "direction", "combined"):
        for group in range(16):
            rows.append(
                dict(
                    group=group,
                    arm=arm,
                    baseline_outcome=dict(task_pass=group < 4, safe=True),
                    outcome=dict(
                        task_pass=group < (6 if arm == "combined" else 5),
                        safe=not (arm != "direction" and group in (8, 9, 10)),
                    ),
                )
            )
    return rows


def test_safety_veto_overrides_larger_task_gain():
    result = select_pilot_mechanism(pilot())
    assert result["chosen_mechanism"] == "direction"
    assert result["summaries"]["combined"]["net_rescue"] == 2
    assert result["summaries"]["combined"]["safety_regression"] == 3
    assert result["summaries"]["combined"]["eligible_for_expansion"] is False
    assert result["training_authorized"] is False


def test_no_eligible_arm_means_pivot_not_automatic_repeat():
    rows = pilot()
    for row in rows:
        row["outcome"]["safe"] = False
    assert select_pilot_mechanism(rows)["chosen_mechanism"] is None


def test_incomplete_or_mismatched_baseline_cannot_select():
    with pytest.raises(ValueError):
        select_pilot_mechanism(pilot()[:-1])
    rows = pilot()
    rows[-1]["baseline_outcome"]["safe"] = False
    with pytest.raises(ValueError):
        select_pilot_mechanism(rows)
