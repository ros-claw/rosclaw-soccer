from scripts.rsi_diagnose_lateral_bank_handoff import summarize


def test_summary_counts_simultaneous_groups_without_attributing_cause():
    rows = [
        dict(
            first_receiver_contact_force_groups=[],
            actual_passer_foot_over_10n=True,
            actual_receiver_contact_over_1n=False,
            excluded_contact_ever=False,
            minimum_root_ball_xy_m=0.6,
        ),
        dict(
            first_receiver_contact_force_groups=["left_foot", "nonfoot"],
            actual_passer_foot_over_10n=True,
            actual_receiver_contact_over_1n=True,
            excluded_contact_ever=True,
            minimum_root_ball_xy_m=0.2,
        ),
    ]
    result = summarize(rows)
    assert result["first_contact_groups"] == {"left_foot+nonfoot": 1, "none": 1}
    assert result["no_handoff"] == 2
    assert result["receiver_contact_over_1n"] == 1
    assert result["within_0_65m_after_pass"] == 2
