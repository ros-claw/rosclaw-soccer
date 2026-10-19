"""No physics or teacher claim from a local course declaration."""

import copy

import pytest

from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.sim.contracts import hash_json


def courses():
    return tuple(
        c
        for c in ((2.68, 0.117, 0.34), (2.645, 0.104, 0.277), (2.646, 0.133, 0.516))
        for _ in range(2)
    )


def test_full_declaration_before_workers_is_deterministic_and_not_new_sample_inflation():
    original = courses()
    p = declare_local_impact_perturbations(original)
    assert p == declare_local_impact_perturbations(copy.deepcopy(original))
    assert p["new_rollouts"] == 24
    assert p["new_unique_physical_courses"] == 6
    assert p["original_source_clusters"] == 3
    assert p["paired_policy_course_contexts"] == 12
    assert p["baseline_source_rollouts_reused"] == 12
    assert p["combined_step_budget"] == 93600
    assert len(p["jobs"]) == 12
    assert p["report_hash"] == hash_json({k: v for k, v in p.items() if k != "report_hash"})
    for key in (
        "private_fresh_accessed",
        "teacher_qualified",
        "global_four_strata_generalization_claimed",
        "training_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert p[key] is False
    assert p["optimizer_updates"] == 0
    assert original == courses()


@pytest.mark.parametrize(
    "course",
    [
        (2.795, 0.1, 0.4),
        (2.605, 0.1, 0.4),
        (2.65, 0.15, 0.4),
        (2.65, -0.15, 0.4),
        (2.65, 0.1, 0.26),
        (2.65, 0.1, 0.69),
    ],
)
def test_single_bad_perturbation_rejects_whole_declaration_not_clip_drop_or_replace(course):
    c = list(courses())
    c[:2] = [course, course]
    with pytest.raises(ValueError):
        declare_local_impact_perturbations(tuple(c))


@pytest.mark.parametrize(
    "mutate", ["missing", "duplicate", "pair", "negative", "lowx", "nan", "bool", "list"]
)
def test_wrong_cohort_or_nonordinary_domain_is_not_admitted(mutate):
    c = list(courses())
    if mutate == "missing":
        c.pop()
    elif mutate == "duplicate":
        c[2:4] = c[:2]
    elif mutate == "pair":
        c[1] = (2.67, 0.117, 0.34)
    elif mutate == "negative":
        c[:2] = [(2.68, 0.117, -0.34)] * 2
    elif mutate == "lowx":
        c[:2] = [(2.28, 0.117, 0.34)] * 2
    elif mutate == "nan":
        c[0] = (2.68, float("nan"), 0.34)
    elif mutate == "bool":
        c[0] = (True, 0.117, 0.34)
    with pytest.raises(ValueError):
        declare_local_impact_perturbations(c if mutate == "list" else tuple(c))
