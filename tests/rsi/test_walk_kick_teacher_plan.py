import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.rsi.walk_kick_teacher_plan import (
    BOUNDS,
    candidate_parameters,
    elite_candidates,
    outcome_rank,
    search_bounds,
    teacher_courses,
    uniform_candidates,
)
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_all16_preflight_before_search_balanced_and_distinct():
    courses = teacher_courses()
    assert preflight_receiving_courses(courses).startswith("sha256:")
    assert len({c.course for c in courses}) == 16
    counts = [0] * 4
    for c in courses:
        counts[validate_public_course(c.course)] += 1
    assert counts == [4] * 4


def test_bounded_two_generation_sampler_reproducible_and_no_reference_aliasing():
    first = uniform_candidates(17)
    np.testing.assert_array_equal(first, uniform_candidates(17))
    before = first.copy()
    ranks = tuple((i,) for i in range(8))
    second = elite_candidates(first, ranks, seed=18)
    np.testing.assert_array_equal(first, before)
    np.testing.assert_array_equal(second, elite_candidates(first, ranks, seed=18))
    assert second.shape == (8, 4)
    assert (second >= BOUNDS[:, 0]).all() and (second <= BOUNDS[:, 1]).all()
    for x in np.concatenate((first, second)):
        p = candidate_parameters(x)
        assert 180 <= p["reference_frame"] <= 250
        assert 0.5 <= p["speed_mps"] <= 1


@pytest.mark.parametrize("defect", ["nan", "outside", "short", "missing-rank"])
def test_invalid_search_declaration_rejected(defect):
    first = uniform_candidates(17)
    ranks = ((1,),) * 8
    if defect == "nan":
        first[0, 0] = np.nan
    elif defect == "outside":
        first[0, 0] = 2
    elif defect == "short":
        first = first[:7]
    else:
        ranks = ranks[:7]
    with pytest.raises(ValueError):
        elite_candidates(first, ranks, seed=18)


def test_true_safety_and_retention_dominate_scalar_reward():
    def row(healthy, safe, passed, reward):
        return dict(
            healthy_full_horizon=healthy,
            outcome=dict(
                safe=safe,
                task_pass=passed,
                training_reward=reward,
                first_foot_contact_frame=100,
            ),
        )

    healthy_failure = row(True, True, False, -3)
    assert outcome_rank(healthy_failure) > outcome_rank(row(False, True, True, 100))
    assert outcome_rank(healthy_failure) > outcome_rank(row(True, False, True, 100))
    assert outcome_rank(row(True, True, True, 1)) > outcome_rank(row(True, True, False, 100))


def test_experimental_domain_is_explicit_and_does_not_mutate_old_search():
    original = uniform_candidates(42)
    fast = uniform_candidates(42, experimental_fast=True)
    assert (fast[:, 0] >= 1).all() and (fast[:, 0] <= 2).all()
    np.testing.assert_array_equal(original, uniform_candidates(42))
    assert search_bounds()[0].tolist() == [0.5, 1.0]
    with pytest.raises(ValueError):
        candidate_parameters(fast[0])
    assert candidate_parameters(fast[0], experimental_fast=True)["speed_mps"] > 1
    second = elite_candidates(fast, tuple((i,) for i in range(8)), seed=43, experimental_fast=True)
    assert (second[:, 0] >= 1).all() and (second[:, 0] <= 2).all()
