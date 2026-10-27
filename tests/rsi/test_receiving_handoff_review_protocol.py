from scripts.rsi_review_receiving_handoff_windows import STUDY_STATUSES


def test_sweep_reports_have_distinct_exact_own_audit_statuses():
    assert len(STUDY_STATUSES) == 5
    assert len(set(STUDY_STATUSES.values())) == len(STUDY_STATUSES)
    for source, audit in STUDY_STATUSES.items():
        assert source.endswith("_COLLECTED_PENDING_OWN_REPLAY")
        assert audit.endswith("_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS")
        assert source.replace("_COLLECTED_PENDING_OWN_REPLAY", "") == audit.replace(
            "_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS", ""
        ) or source.startswith("PREALIGNED_FOOT_CAPTURE_TEACHER32_")
