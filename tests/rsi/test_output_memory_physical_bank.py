import copy

import pytest

from scripts import rsi_fit_output_memory_motor as fitter


def fixture(monkeypatch):
    courses = [[11, 0], [12, 4]]
    monkeypatch.setattr(
        fitter,
        "failure_rows",
        lambda bank, review, *, arm: [dict(seed=seed, lane=lane) for seed, lane in courses],
    )
    model = dict(model_hash="current", parent_model_hash="frozen")
    bank = dict(report_hash="bank", commitment=dict(model_hash="frozen"))
    review = dict(report_hash="review")
    commitment = dict(
        schema="soccer.rsi.output_memory_exploration_commitment.v1",
        behavior_kind="OUTPUT_MEMORY_CURRENT_PARENT",
        partition="TRAIN_CONSUMED",
        courses=courses,
        samples_per_course=4,
        bank_hash="bank",
        bank_review_hash="review",
        base_model_hash="current",
        warm_model_hash="current",
        frozen_parent_model_hash="frozen",
        std_raw=0.1,
        sampling_view_hashes=[str(i) for i in range(8)],
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary = dict(
        schema="soccer.rsi.output_memory_failure_exploration.v1",
        commitment=commitment,
        rows=[{}, {}],
        exploration_executions=8,
        physical_executions=12,
        independent_contexts=2,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return model, summary, bank, review, copy.deepcopy(commitment)


def test_complete_current_parent_declaration_is_required(monkeypatch):
    args = fixture(monkeypatch)
    assert fitter.checked_curriculum(*args) == [[11, 0], [12, 4]]


@pytest.mark.parametrize(
    "target,key,value",
    [
        (1, "physical_executions", 8),
        (1, "independent_contexts", 8),
        (1, "promotion_authorized", True),
        (4, "partition", "FRESH_HOLDOUT"),
        (4, "samples_per_course", True),
        (4, "warm_model_hash", "old"),
        (4, "std_raw", 0.2),
        (4, "hardware_authorized", True),
        (4, "sampling_view_hashes", []),
    ],
)
def test_incomplete_stale_or_authorized_bank_is_rejected(monkeypatch, target, key, value):
    args = fixture(monkeypatch)
    args[target][key] = value
    if target == 4:
        args[1]["commitment"] = copy.deepcopy(args[4])
    with pytest.raises(ValueError):
        fitter.checked_curriculum(*args)


def test_wrong_frozen_parent_is_rejected(monkeypatch):
    args = fixture(monkeypatch)
    args[2]["commitment"]["model_hash"] = "another-parent"
    with pytest.raises(ValueError):
        fitter.checked_curriculum(*args)
