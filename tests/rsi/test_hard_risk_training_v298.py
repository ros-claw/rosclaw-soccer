"""The training bank selection is fixed before physics or outcome inspection."""

from scripts.rsi_collect_hard_risk_training_v298 import courses


def test_outcome_blind_training_catalog() -> None:
    selected = courses()
    assert len(selected) == len(set(selected)) == 40
    assert selected[0] == (20261161, 6)
    assert selected[-1] == (20261446, 0)
