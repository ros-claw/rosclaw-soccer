"""Complete coverage and independent exploration, not native physics evidence."""

import copy

import pytest

from rosclaw_soccer.rsi.recurrent_bank_plan import declare_jobs
from scripts.rsi_collect_failed_step_courses import sampling_seed


def source_records():
    return [
        dict(
            seed=20261177 + course,
            lane=course % 16,
            baseline_course_index=course,
            sampling_seed=sampling_seed(course, draw, stream=8),
            sample_index=draw,
            outcome={"high_quality": draw == 0},
        )
        for course in range(40)
        for draw in range(4)
    ]


def test_all_contexts_and_draws_declared_without_reading_outcomes():
    records = source_records()
    original = copy.deepcopy(records)
    jobs = declare_jobs(records, samples_per_context=8, stream=13)
    assert len(jobs) == 320
    assert [job["group"] for job in jobs] == list(range(320))
    assert len({(job["seed"], job["lane"]) for job in jobs}) == 40
    assert len({job["sampling_seed"] for job in jobs}) == 320
    for job in jobs:
        assert job["sampling_seed"] == sampling_seed(
            job["baseline_course_index"], job["sample_index"], stream=13
        )
    assert records == original
    for row in records:
        row.pop("outcome")
    assert declare_jobs(records, samples_per_context=8, stream=13) == jobs
    jobs[0]["seed"] = 0
    assert records[0]["seed"] == original[0]["seed"]


@pytest.mark.parametrize("change", ["missing", "identity", "draw", "seed", "course", "bool"])
def test_incomplete_duplicate_or_misaligned_source_rejected(change):
    records = source_records()
    if change == "missing":
        records.pop()
    elif change == "identity":
        records[1]["lane"] += 1
    elif change == "draw":
        records[1]["sample_index"] = 0
    elif change == "seed":
        records[1]["sampling_seed"] = records[0]["sampling_seed"]
    elif change == "course":
        for row in records[4:8]:
            row["baseline_course_index"] = 0
    else:
        records[0]["seed"] = True
    with pytest.raises(ValueError):
        declare_jobs(records, samples_per_context=4, stream=13)


@pytest.mark.parametrize(
    "count,stream", [(True, 13), (3, 13), (17, 13), (4, True), (4, 32), (4, 8)]
)
def test_bounds_and_reused_behavior_draws_rejected(count, stream):
    with pytest.raises(ValueError):
        declare_jobs(source_records(), samples_per_context=count, stream=stream)
