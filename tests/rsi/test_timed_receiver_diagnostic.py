from scripts.rsi_diagnose_timed_receiver_pairs import summary


def test_summary_counts_contact_and_full_failures_without_assuming_causality():
    rows = [
        dict(
            contact=dict(
                observed_receiver_contact=True,
                unmatched_recorded_categories=[2],
                reconstructed_contacts=[
                    dict(receiver_body="b/thigh", category="other_body", relative_speed_mps=1.5)
                ],
            ),
            failure_control=20,
            failed_bodies=["A"],
            handoff_control=10,
        ),
        dict(
            contact=dict(observed_receiver_contact=False, reconstructed_contacts=[]),
            failure_control=None,
            failed_bodies=[],
            handoff_control=None,
        ),
    ]
    out = summary(rows)
    assert out["trajectories"] == 2 and out["contact_observed"] == 1
    assert out["contact_unmatched"] == 1 and out["reconstructed_bodies"] == {"b/thigh": 1}
    assert out["A_failure"] == 1 and out["B_failure"] == 0
    assert out["first_failure_after_handoff"] == 1
