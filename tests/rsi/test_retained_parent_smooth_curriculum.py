import pytest

from scripts.rsi_prepare_retained_parent_smooth_curriculum import qualify_parent_bank


def fixture():
    parent = dict(model_hash="retained-parent", generation=1)
    model = dict(generation=0, frozen_parent=parent)
    bank = dict(
        schema="soccer.rsi.protected_phase_bank_physics.v1",
        commitment=dict(partition="TRAIN_CONSUMED", model_hash="retained-parent"),
        physical_executions=156,
        independent_contexts=52,
        report_hash="bank",
        warm_high_quality=35,
        candidate_high_quality=38,
        promotion_authorized=False,
        hardware_authorized=False,
        rows=[
            dict(index=i, seed=1000 + i, lane=0, candidate=dict(high_quality=i < 38))
            for i in range(52)
        ],
    )
    review = dict(
        schema="soccer.rsi.protected_phase_bank_review.v1",
        source_summary_hash="bank",
        physical_reports_reviewed=156,
        motor_frames_reconstructed=31200,
        warm_high_quality=35,
        candidate_high_quality=38,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    for report in (bank, review):
        report.update(
            safe_pelvis=True, old_high_quality_loss=0, old_clean_foot_loss=0, new_out_of_play=0
        )
    return model, parent, bank, review


def test_all_current_failures_are_used_not_earlier_parent_failures():
    assert len(qualify_parent_bank(*fixture())) == 14


def test_gain_cannot_override_two_new_out_of_play_failures():
    model, parent, bank, review = fixture()
    review["new_out_of_play"] = 2
    with pytest.raises(ValueError, match="retained safe"):
        qualify_parent_bank(model, parent, bank, review)


@pytest.mark.parametrize(
    "key,value", [("old_clean_foot_loss", 1), ("old_high_quality_loss", 1), ("safe_pelvis", False)]
)
def test_nonretained_parent_is_never_advanced(key, value):
    model, parent, bank, review = fixture()
    bank[key] = value
    with pytest.raises(ValueError):
        qualify_parent_bank(model, parent, bank, review)


def test_unrelated_parent_or_nonzero_child_cannot_use_a_good_bank():
    model, parent, bank, review = fixture()
    parent = dict(model_hash="different", generation=1)
    with pytest.raises(ValueError):
        qualify_parent_bank(model, parent, bank, review)
    model, parent, bank, review = fixture()
    model["generation"] = 1
    with pytest.raises(ValueError):
        qualify_parent_bank(model, parent, bank, review)
