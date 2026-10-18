from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.rsi_mesh_option4_teacher_screen import declared_jobs


def contexts():
    return [SimpleNamespace(context_index=i, source_group=64 + i) for i in range(8)]


def test_all_contexts_are_paired_and_old_successes_also_tested():
    jobs = declared_jobs(contexts())
    assert len(jobs) == len({j["job"] for j in jobs}) == 18
    assert Counter(j["role"] for j in jobs) == {"control": 8, "teacher": 10}
    for i in range(8):
        pair = [j for j in jobs if j["pair_id"] == f"context-{i}"]
        assert [j["option_index"] for j in pair] == [0, 4]
        assert len({j["group"] for j in pair}) == 1
    assert [j["group"] for j in jobs[-2:]] == [4, 32]


def test_reused_hard_draws_are_not_charged_again_or_selected_after_results():
    jobs = declared_jobs(contexts())
    assert not any(j["context_index"] is None and j["group"] in (8, 56) for j in jobs)
    jobs[0]["group"] = 1000
    assert declared_jobs(contexts())[0]["group"] == 64


def test_partial_context_list_is_rejected_before_allocation():
    with pytest.raises(ValueError):
        declared_jobs(contexts()[:7])
