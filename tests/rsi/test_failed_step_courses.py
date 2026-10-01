import copy

import pytest

from scripts.rsi_collect_failed_step_courses import failure_rows, sampling_seed


def bank():
    summary = dict(
        schema="soccer.rsi.protected_phase_bank_physics.v1",
        commitment=dict(partition="TRAIN_CONSUMED"),
        physical_executions=156,
        independent_contexts=52,
        report_hash="sealed-summary",
        warm_high_quality=32,
        promotion_authorized=False,
        hardware_authorized=False,
        rows=[
            dict(index=i, seed=1000 + i, lane=0, warm=dict(high_quality=i < 32)) for i in range(52)
        ],
    )
    review = dict(
        schema="soccer.rsi.protected_phase_bank_review.v1",
        source_summary_hash=summary["report_hash"],
        physical_reports_reviewed=156,
        motor_frames_reconstructed=31200,
        warm_high_quality=32,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return summary, review


def test_all_twenty_failures_are_selected_without_mutating_review():
    summary, review = bank()
    original = copy.deepcopy(summary)
    rows = failure_rows(summary, review)
    assert [r["index"] for r in rows] == list(range(32, 52))
    assert summary == original


def test_later_parent_failure_selection_uses_candidate_not_initial_warm_results():
    summary, review = bank()
    for row in summary["rows"]:
        row["candidate"] = dict(high_quality=row["index"] < 35)
    summary["candidate_high_quality"] = review["candidate_high_quality"] = 35
    assert [r["index"] for r in failure_rows(summary, review, arm="candidate")] == list(
        range(35, 52)
    )
    assert len(failure_rows(summary, review)) == 20
    with pytest.raises(ValueError):
        failure_rows(summary, review, arm="undeclared")
    review["candidate_high_quality"] = 32
    with pytest.raises(ValueError):
        failure_rows(summary, review, arm="candidate")


def test_all_declared_sampling_seeds_fit_actual_sampler_range_and_are_unique():
    seeds = [sampling_seed(i, s) for i in range(52) for s in range(16)]
    assert len(set(seeds)) == 52 * 16
    assert all(0 <= seed <= 2**31 - 1 for seed in seeds)
    with pytest.raises(ValueError):
        sampling_seed(52, 0)


@pytest.mark.parametrize(
    "mutation", ["fresh", "truncated", "duplicate", "unreviewed", "authority", "nonboolean"]
)
def test_incomplete_or_wrong_partition_cannot_allocate_workers(mutation):
    summary, review = bank()
    if mutation == "fresh":
        summary["commitment"]["partition"] = "FRESH"
    elif mutation == "truncated":
        summary["rows"].pop()
    elif mutation == "duplicate":
        summary["rows"][51]["seed"] = summary["rows"][50]["seed"]
    elif mutation == "unreviewed":
        review["source_summary_hash"] = "another-summary"
    elif mutation == "authority":
        review["hardware_authorized"] = True
    else:
        summary["rows"][0]["warm"]["high_quality"] = 1
    with pytest.raises(ValueError):
        failure_rows(summary, review)
