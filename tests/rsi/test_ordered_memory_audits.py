import time

import pytest

from scripts.rsi_audit_memory_learning_rollouts import ordered_audits


def delayed(job):
    time.sleep(job["delay"])
    if job.get("reject"):
        raise ValueError("full audit rejected the trajectory")
    return job["index"]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_completion_order_cannot_relabel_trajectories(workers):
    jobs = [dict(index=i, delay=(3 - i) * 0.01) for i in range(4)]
    assert list(ordered_audits(delayed, jobs, workers)) == [0, 1, 2, 3]


@pytest.mark.parametrize("workers", [1, 2])
def test_worker_audit_failure_is_not_silently_skipped(workers):
    jobs = [dict(index=i, delay=0, reject=i == 1) for i in range(3)]
    with pytest.raises(ValueError, match="full audit rejected"):
        list(ordered_audits(delayed, jobs, workers))


@pytest.mark.parametrize("workers", [True, False, 0, 5, -1, 1.5])
def test_unbounded_or_boolean_worker_count_is_rejected(workers):
    with pytest.raises(ValueError, match="audit workers"):
        list(ordered_audits(delayed, [dict(index=0, delay=0)], workers))


def test_empty_bank_is_not_learning_evidence():
    with pytest.raises(ValueError, match="nonempty"):
        list(ordered_audits(delayed, [], 1))
