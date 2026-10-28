import pytest

from scripts.rsi_review_incoming_pilot import paired_changes


def test_equal_totals_can_hide_gains_and_losses():
    before = [{"bank_index": 3, "healthy": True}, {"bank_index": 4, "healthy": False}]
    after = [{"bank_index": 3, "healthy": False}, {"bank_index": 4, "healthy": True}]
    assert paired_changes(before, after, "healthy") == {"gained": [4], "lost": [3]}


def test_mismatched_worlds_fail_closed():
    before = [{"bank_index": 3, "healthy": True}]
    with pytest.raises(ValueError, match="identity"):
        paired_changes(before, [{"bank_index": 4, "healthy": True}], "healthy")
    with pytest.raises(ValueError, match="count"):
        paired_changes(before, [], "healthy")
