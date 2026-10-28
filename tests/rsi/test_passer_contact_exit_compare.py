import copy

import pytest

from rosclaw_soccer.rsi.passer_contact_exit_compare import compare_passer_reviews


def review():
    outcome = {
        "strict_receiver_clean_handoff": False,
        "strict_both_clean_handoff": False,
        "original_handoff": False,
        "both_bodies_healthy": True,
        "original": {"return_0p1m_and_healthy": False},
    }
    return {
        "status": "PASSER_RESIDUAL_PAIR_REPLAY_REVIEWED",
        "conditions": 24,
        "hardware_authorized": False,
        "promotion_authorized": False,
        "goal_complete": False,
        "records": [
            {
                "block": i // 8,
                "context": i // 2,
                "side": (-0.55, 0.55)[i % 2],
                "historical_baseline_identical": True,
                "baseline": copy.deepcopy(outcome),
                "teacher": copy.deepcopy(outcome),
            }
            for i in range(24)
        ],
    }


def test_compare_counts_matched_rescues_and_rejects_changed_baseline():
    old = review()
    new = copy.deepcopy(old)
    new["records"][0]["teacher"]["strict_both_clean_handoff"] = True
    old["records"][1]["teacher"]["original_handoff"] = True
    result = compare_passer_reviews(old, new)
    assert result["summary"]["strict_both_clean_handoff"]["rescues"] == 1
    assert result["summary"]["original_handoff"]["losses"] == 1
    assert result["promotion_authorized"] is False
    new["records"][0]["baseline"]["original_handoff"] = True
    with pytest.raises(ValueError, match="baselines"):
        compare_passer_reviews(old, new)


def test_compare_rejects_duplicate_or_unverified_review():
    old = review()
    new = copy.deepcopy(old)
    new["records"][1] = copy.deepcopy(new["records"][0])
    with pytest.raises(ValueError, match="unique"):
        compare_passer_reviews(old, new)
    old["hardware_authorized"] = True
    with pytest.raises(ValueError, match="unpromoted"):
        compare_passer_reviews(old, review())
