"""Fresh skill gates cannot hide incomplete team safety episodes."""

import pytest

from scripts.rsi_team_lateral_phase_fresh_gate_v53 import evaluate_primary_gate

GATE = {
    "primary_arm": "gate22_cap10",
    "minimum_useful_gain_over_baseline": 3,
    "minimum_safe_contact_gain_over_baseline": 5,
    "maximum_unsafe_excess_over_baseline": 0,
}


def test_incomplete_primary_cannot_pass_despite_large_gain() -> None:
    metrics = {
        "parent": {"complete": 32, "incomplete": 0, "safe": 30},
        "baseline": {
            "complete": 31,
            "incomplete": 1,
            "safe": 29,
            "safe_contact": 4,
            "useful": 1,
        },
        "gate22_cap10": {
            "complete": 31,
            "incomplete": 1,
            "safe": 30,
            "safe_contact": 15,
            "useful": 12,
        },
    }
    assert evaluate_primary_gate(metrics, GATE) == (11, 11, -1, False)
    metrics["gate22_cap10"]["complete"] = 32
    metrics["gate22_cap10"]["incomplete"] = 0
    assert evaluate_primary_gate(metrics, GATE) == (11, 11, -1, True)


def test_invalid_episode_accounting_rejected() -> None:
    metrics = {
        "parent": {"complete": 32, "incomplete": 0, "safe": 32},
        "baseline": {
            "complete": 32,
            "incomplete": 0,
            "safe": 31,
            "safe_contact": 1,
            "useful": 0,
        },
        "gate22_cap10": {
            "complete": 31,
            "incomplete": 0,
            "safe": 31,
            "safe_contact": 10,
            "useful": 6,
        },
    }
    with pytest.raises(ValueError, match="episode counts"):
        evaluate_primary_gate(metrics, GATE)
