import pytest

from scripts.rsi_motion_ppo_pilot import learner_receipt_artifacts


def test_consolidation_uses_its_actual_receipts_not_ppo_only_file():
    assert learner_receipt_artifacts("consolidate") == ("complete.json", "declaration.json")
    assert "experience-commitment.json" in learner_receipt_artifacts("learn")


def test_unknown_learner_protocol_rejected():
    with pytest.raises(ValueError):
        learner_receipt_artifacts("unknown")
