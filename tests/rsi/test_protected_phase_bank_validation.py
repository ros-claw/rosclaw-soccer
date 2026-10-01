from scripts.rsi_collect_protected_phase_bank_validation import score


def test_quality_gain_does_not_hide_retention_loss_or_safety_failure():
    def outcome(hq, clean=True, pelvis=0.70, lateral=1.0):
        return dict(
            high_quality=hq,
            clean_foot_only=clean,
            minimum_pelvis_z_m=pelvis,
            maximum_lateral_excursion_m=lateral,
        )

    rows = [
        dict(warm=outcome(True), candidate=outcome(False, clean=False)),
        dict(warm=outcome(False), candidate=outcome(True)),
        dict(warm=outcome(False), candidate=outcome(True, pelvis=0.60, lateral=4.1)),
    ]
    result = score(rows)
    assert result["candidate_high_quality"] > result["warm_high_quality"]
    assert result["old_high_quality_loss"] == 1
    assert result["old_clean_foot_loss"] == 1
    assert result["new_out_of_play"] == 1
    assert result["safe_pelvis"] is False
