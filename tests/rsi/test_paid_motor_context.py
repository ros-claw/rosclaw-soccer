import pytest

from scripts.rsi_exam_paid_live_distilled import select_entry_actor


def test_motor_context_routes_only_declared_skill_not_case_or_observed_outcome():
    prior, candidate = object(), object()
    assert select_entry_actor("candidate", candidate, prior) is candidate
    assert select_entry_actor("qualified-prior", candidate, prior) is prior
    with pytest.raises(ValueError, match="context"):
        select_entry_actor("best-for-case5", candidate, prior)
