from collections import Counter

import pytest

from rosclaw_soccer.training.capture_campaign_plan import (
    capture_campaign_courses,
    require_capture_storage,
)


def test_large_batch_balanced_unique_and_repeatable():
    kwargs = dict(count=48, first_seed=268600, speeds=(1.25, 2.0, 2.75))
    jobs = capture_campaign_courses(**kwargs)
    assert jobs == capture_campaign_courses(**kwargs)
    assert len({j.seed for j in jobs}) == len({j.name for j in jobs}) == 48
    assert Counter(j.inward_speed_mps for j in jobs) == {1.25: 16, 2.0: 16, 2.75: 16}
    assert [j.inward_speed_mps for j in jobs[:3]] == [1.25, 2.0, 2.75]


@pytest.mark.parametrize(
    "change",
    [
        {"count": 0},
        {"count": True},
        {"count": 47},
        {"first_seed": 2**32 - 1},
        {"speeds": (float("nan"),)},
        {"speeds": (0.0,)},
        {"speeds": (2.0, 2.0)},
        {"speeds": ()},
        {"speeds": [2.0]},
    ],
)
def test_invalid_plan_rejected(change):
    kwargs = dict(count=48, first_seed=268600, speeds=(1.25, 2.0, 2.75))
    kwargs.update(change)
    with pytest.raises(ValueError):
        capture_campaign_courses(**kwargs)


def test_storage_includes_all_pending_workers_not_one_job():
    require_capture_storage(
        free_bytes=148, pending_courses=48, course_budget_bytes=1, reserve_bytes=100
    )
    with pytest.raises(ValueError, match="insufficient"):
        require_capture_storage(
            free_bytes=147, pending_courses=48, course_budget_bytes=1, reserve_bytes=100
        )


def test_resume_budget_counts_only_pending_but_keeps_reserve():
    require_capture_storage(
        free_bytes=110, pending_courses=10, course_budget_bytes=1, reserve_bytes=100
    )
    with pytest.raises(ValueError):
        require_capture_storage(
            free_bytes=99, pending_courses=0, course_budget_bytes=1, reserve_bytes=100
        )
