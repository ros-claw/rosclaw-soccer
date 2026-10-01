import copy

import pytest

from scripts import rsi_fit_smooth_memory_motor as fitter


def fixture(monkeypatch):
    failures = [dict(seed=i, lane=0) for i in range(1, 6)]
    monkeypatch.setattr(fitter, "failure_rows", lambda bank, review, *, arm: failures)
    model = dict(
        model_hash="current",
        parent_model_hash="memory-parent",
        frozen_parent=dict(frozen_parent=dict(model_hash="physical-parent")),
    )
    bank = dict(report_hash="bank", commitment=dict(model_hash="physical-parent"))
    review = dict(report_hash="review")
    commitment = dict(
        schema="soccer.rsi.smooth_memory_exploration_commitment.v1",
        behavior_kind="OUTPUT_MEMORY_CURRENT_PARENT_AR1",
        partition="TRAIN_CONSUMED",
        course_selection="FIRST_FOUR_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER",
        courses=[[i, 0] for i in range(1, 5)],
        samples_per_course=4,
        bank_hash="bank",
        bank_review_hash="review",
        base_model_hash="current",
        warm_model_hash="current",
        frozen_parent_model_hash="memory-parent",
        failure_reference_model_hash="physical-parent",
        failure_reference_total_courses=5,
        std_raw=0.1,
        sampling_rho=0.9,
        sampling_view_hashes=[str(i) for i in range(16)],
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary = dict(
        schema="soccer.rsi.smooth_memory_failure_exploration.v1",
        commitment=commitment,
        rows=[{}, {}, {}, {}],
        exploration_executions=16,
        physical_executions=24,
        independent_contexts=4,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return model, summary, bank, review, copy.deepcopy(commitment)


def test_fixed_prefix_cannot_be_selected_by_favorable_sampling_results(monkeypatch):
    args = fixture(monkeypatch)
    assert fitter.checked_curriculum(*args) == [[1, 0], [2, 0], [3, 0], [4, 0]]
    args[4]["courses"] = [[2, 0], [3, 0], [4, 0], [5, 0]]
    args[1]["commitment"] = copy.deepcopy(args[4])
    with pytest.raises(ValueError):
        fitter.checked_curriculum(*args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("course_selection", "SUCCESSFUL_SAMPLES_ONLY"),
        ("partition", "FRESH_HOLDOUT"),
        ("samples_per_course", True),
        ("warm_model_hash", "stale"),
        ("sampling_rho", 0),
        ("std_raw", 0.05),
        ("failure_reference_model_hash", "another-policy"),
        ("failure_reference_total_courses", 4),
        ("hardware_authorized", True),
        ("sampling_view_hashes", []),
    ],
)
def test_wrong_conditioning_or_reference_is_rejected(monkeypatch, key, value):
    args = fixture(monkeypatch)
    args[4][key] = value
    args[1]["commitment"] = copy.deepcopy(args[4])
    with pytest.raises(ValueError):
        fitter.checked_curriculum(*args)


def test_next_smooth_generation_cannot_reuse_a_stale_physical_parent(monkeypatch):
    args = fixture(monkeypatch)
    args[0]["generation"] = 1
    with pytest.raises(ValueError):
        fitter.checked_curriculum(*args)
    args[2]["commitment"]["model_hash"] = "current"
    args[4]["failure_reference_model_hash"] = "current"
    args[1]["commitment"] = copy.deepcopy(args[4])
    assert fitter.checked_curriculum(*args) == [[1, 0], [2, 0], [3, 0], [4, 0]]
