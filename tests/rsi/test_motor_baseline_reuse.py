import copy

import pytest

from scripts.rsi_collect_protected_phase_bank_validation import verify_baseline_reuse


def source():
    commitment = dict(
        runner_hash="runner",
        asset_hash="asset",
        core_commit="core",
        learning_bank_hash="bank",
        warm_model_hash="warm",
        partition="TRAIN_CONSUMED",
    )
    courses = [dict(seed=1000 + i, lane=0) for i in range(52)]
    previous = dict(
        schema="soccer.rsi.protected_phase_bank_physics.v1",
        physical_executions=156,
        independent_contexts=52,
        promotion_authorized=False,
        hardware_authorized=False,
        commitment=commitment,
        rows=[dict(index=i, **c) for i, c in enumerate(courses)],
    )
    return previous, dict(courses=courses), copy.deepcopy(commitment)


def test_only_complete_same_physical_controls_are_reusable():
    previous, bank, commitment = source()
    verify_baseline_reuse(previous, bank, commitment)
    for field in commitment:
        changed = copy.deepcopy(commitment)
        changed[field] = "changed"
        with pytest.raises(ValueError):
            verify_baseline_reuse(previous, bank, changed)


def test_partial_reordered_or_authorized_evidence_is_not_reusable():
    for mutation in ("partial", "reordered", "authority"):
        previous, bank, commitment = source()
        if mutation == "partial":
            previous["rows"].pop()
        elif mutation == "reordered":
            previous["rows"].reverse()
        else:
            previous["promotion_authorized"] = True
        with pytest.raises(ValueError):
            verify_baseline_reuse(previous, bank, commitment)
