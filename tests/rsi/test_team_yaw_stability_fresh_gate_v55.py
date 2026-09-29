"""A strong pass count cannot conceal missing or unsafe G1 teammates."""

import pytest

from scripts.rsi_team_yaw_stability_fresh_gate_v55 import evaluate_fresh_gate

GATE = {
    "primary_arm": "gate22_cap10",
    "minimum_useful_gain_over_baseline": 6,
    "minimum_safe_contact_gain_over_baseline": 10,
    "maximum_unsafe_excess_over_baseline": 0,
    "minimum_primary_useful": 16,
}


def _scores() -> dict[str, dict[str, int]]:
    return {
        "parent": {"complete": 64, "incomplete": 0, "safe": 50, "safe_contact": 0, "useful": 0},
        "baseline": {
            "complete": 64,
            "incomplete": 0,
            "safe": 52,
            "safe_contact": 10,
            "useful": 5,
        },
        "gate22_cap10": {
            "complete": 64,
            "incomplete": 0,
            "safe": 53,
            "safe_contact": 23,
            "useful": 17,
        },
    }


def test_gate_requires_full_team_completion_and_safety() -> None:
    scores = _scores()
    assert evaluate_fresh_gate(scores, GATE, 64) == (12, 13, -1, True)
    scores["gate22_cap10"].update(complete=63, incomplete=1)
    assert evaluate_fresh_gate(scores, GATE, 64) == (12, 13, -1, False)
    scores["gate22_cap10"].update(complete=64, incomplete=0, safe=51)
    assert evaluate_fresh_gate(scores, GATE, 64) == (12, 13, 1, False)


def test_invalid_success_accounting_is_rejected() -> None:
    scores = _scores()
    scores["gate22_cap10"]["safe_contact"] = 16
    with pytest.raises(ValueError, match="outcome accounting"):
        evaluate_fresh_gate(scores, GATE, 64)
