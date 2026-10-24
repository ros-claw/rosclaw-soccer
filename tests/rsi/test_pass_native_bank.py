import pytest

from rosclaw_soccer.rsi.pass_native_bank import original_pass_bank


def test_new_learning_bank_resolves_unchanged_original_physical_exam():
    original = dict(status="TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED", records=[{}] * 177)
    retained = dict(
        status="TEAM_PASS_ACTUAL_STUDENT_CORRECTION_BANK274_READY_NOT_TRAINED",
        records=[{}] * 274,
        base_data_hash="original",
        parent_lineage_hash="parent",
    )
    new = dict(
        status="QUALIFIED_RESCUE_REFINEMENT_BANK_READY_PENDING_OWN_FEATURE_AUDIT",
        records=[{}] * 62,
        fit_plan=dict(fit_histories=62),
        retained_bank_hash="retained",
        parent_lineage_hash="parent",
    )
    resolve = dict(original=original, retained=retained).__getitem__
    assert original_pass_bank(new, "consolidated-refinement-v1", resolve) is original
    new["parent_lineage_hash"] = "different"
    with pytest.raises(ValueError, match="lineage"):
        original_pass_bank(new, "consolidated-refinement-v1", resolve)
    with pytest.raises(ValueError, match="explicit"):
        original_pass_bank(original, "unknown", resolve)
