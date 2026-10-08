"""Structural fixtures only; no native physics or learned-policy evidence."""

from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.current_behavior_bank import validate_current_bank_rows

MEAN = "sha256:" + "a" * 64


def bank():
    jobs, rows = [], []
    for i in range(160):
        job = dict(
            group=i,
            context_id=i // 4,
            sample_index=i % 4,
            seed=100 + i // 4,
            lane=i // 4 % 2,
            baseline_course_index=i // 4,
            sampling_seed=1000 + i,
            actual_behavior_model_hash=MEAN,
            view_hash="sha256:" + f"{i:064x}",
        )
        jobs.append(job)
        rows.append(
            dict(
                job,
                frame_sample_count=270,
                physical_replay_performed_here=True,
                execution_origin="NEW",
                private_fresh_accessed=False,
                promotion_authorized=False,
                hardware_authorized=False,
                outcome=dict(high_quality=i == 0, safety_passed=i != 159),
            )
        )
    return jobs, rows


def test_all_failures_and_unsafe_draws_are_retained_without_authorization():
    jobs, rows = bank()
    original = deepcopy(rows)
    report = validate_current_bank_rows(jobs, rows, behavior_model_hash=MEAN)
    assert (report["episodes"], report["contexts"], report["frame_rows"]) == (160, 40, 43200)
    assert report["high_quality_episodes"] == 1
    assert report["safe_episodes"] == 159
    assert not report["training_authorized"]
    assert not report["physics_validated_here"]
    assert rows == original


@pytest.mark.parametrize("mode", ["partial", "reordered", "stale", "duplicate", "course"])
def test_incomplete_or_different_behavior_bank_is_rejected(mode):
    jobs, rows = bank()
    if mode == "partial":
        rows.pop()
    elif mode == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mode == "stale":
        rows[0]["actual_behavior_model_hash"] = "sha256:" + "b" * 64
    elif mode == "duplicate":
        jobs[1]["sampling_seed"] = rows[1]["sampling_seed"] = 1000
    else:
        jobs[1]["seed"] = rows[1]["seed"] = 999
    with pytest.raises(ValueError):
        validate_current_bank_rows(jobs, rows, behavior_model_hash=MEAN)


@pytest.mark.parametrize(
    "key,value",
    [
        ("group", False),
        ("frame_sample_count", 270.0),
        ("hardware_authorized", 0),
        ("private_fresh_accessed", True),
        ("execution_origin", "CACHED"),
    ],
)
def test_ambiguous_identity_or_execution_authority_is_rejected(key, value):
    jobs, rows = bank()
    rows[0][key] = value
    with pytest.raises(ValueError):
        validate_current_bank_rows(jobs, rows, behavior_model_hash=MEAN)


def test_repeated_course_and_non_boolean_labels_rejected():
    jobs, rows = bank()
    for i in range(4, 8):
        for key in ("seed", "lane", "baseline_course_index"):
            jobs[i][key] = rows[i][key] = jobs[0][key]
    with pytest.raises(ValueError, match="distinct"):
        validate_current_bank_rows(jobs, rows, behavior_model_hash=MEAN)
    jobs, rows = bank()
    rows[0]["outcome"]["high_quality"] = 1
    with pytest.raises(ValueError, match="labels"):
        validate_current_bank_rows(jobs, rows, behavior_model_hash=MEAN)


@pytest.mark.parametrize("value", [None, "sha256:" + "g" * 64, "sha256:" + "a" * 63])
def test_missing_or_malformed_sampling_view_is_rejected(value):
    jobs, rows = bank()
    jobs[0]["view_hash"] = rows[0]["view_hash"] = value
    with pytest.raises(ValueError):
        validate_current_bank_rows(jobs, rows, behavior_model_hash=MEAN)
