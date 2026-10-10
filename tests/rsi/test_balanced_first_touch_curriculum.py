"""Synthetic declarations only: no physical success or admission claims."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rosclaw_soccer.rsi.balanced_first_touch_curriculum import balanced_public_first_touch_jobs
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses


def test_complete160_has40_contexts_and_equal_four_strata() -> None:
    seeds = tuple(range(20261010, 20261020))
    jobs = balanced_public_first_touch_jobs(seeds)
    assert jobs == balanced_public_first_touch_jobs(seeds)
    assert len(jobs) == 160
    assert [job.group for job in jobs] == list(range(160))
    assert len({job.course for job in jobs}) == 40
    assert len({(job.seed, job.lane) for job in jobs}) == 40
    for seed in seeds:
        group = [job for job in jobs if job.seed == seed]
        assert len(group) == 16
        for upper in (False, True):
            for outgoing in (False, True):
                assert (
                    sum(
                        (job.course[0] >= 2.6) == upper and (job.course[2] > 0) == outgoing
                        for job in group
                    )
                    == 4
                )
        for lane in {job.lane for job in group}:
            selected = [job for job in group if job.lane == lane]
            assert [job.repeat for job in selected] == list(range(4))
            assert all(job.course == sample_training_courses(seed)[lane] for job in selected)


def test_rotating_quartets_cover_all16_original_lanes_and_preserve_gap() -> None:
    jobs = balanced_public_first_touch_jobs((10, 20, 30, 40), repeats=1)
    assert {job.lane for job in jobs} == set(range(16))
    assert all(x <= 2.4 or x >= 2.6 for x, _, _ in (job.course for job in jobs))
    assert all(
        -0.16 <= y <= 0.16 and 0.25 <= abs(v) <= 0.7 for _, y, v in (job.course for job in jobs)
    )
    assert jobs != balanced_public_first_touch_jobs((40, 30, 20, 10), repeats=1)


@pytest.mark.parametrize(
    "seeds", [(), [1], (True,), (-1,), (2**32,), (1, 1), (1.0,), ("1",), (None,), tuple(range(65))]
)
def test_reject_bad_seed_declarations(seeds: object) -> None:
    with pytest.raises(ValueError):
        balanced_public_first_touch_jobs(seeds)  # type: ignore[arg-type]


@pytest.mark.parametrize("repeats", [True, False, 0, -1, 17, 4.0, "4", None])
def test_reject_bad_repeat_declarations(repeats: object) -> None:
    with pytest.raises(ValueError):
        balanced_public_first_touch_jobs((1,), repeats=repeats)  # type: ignore[arg-type]


def test_boundary_seeds_and_job_immutability() -> None:
    jobs = balanced_public_first_touch_jobs((0, 2**32 - 1), repeats=16)
    assert len(jobs) == 128
    with pytest.raises(FrozenInstanceError):
        jobs[0].seed = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        jobs[0].course[0] = 0.0  # type: ignore[index]


def test_does_not_touch_global_numpy_rng() -> None:
    before = np.random.get_state()
    balanced_public_first_touch_jobs((11, 12, 13, 14))
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
