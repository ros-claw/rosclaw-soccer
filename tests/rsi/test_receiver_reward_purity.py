import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_reward_purity import (
    receiver_contact_prefix,
    summarize_reward_purity,
)


def test_prefix_contact_is_not_confused_with_later_handoff():
    force = np.zeros((4, 10, 2, 3), np.float64)
    force[3, 0, 1, 2] = 10
    assert receiver_contact_prefix(force, 3)["nonfoot_contact_over_1n"] is False
    force[1, 4, 1, 2] = 1.1
    result = receiver_contact_prefix(force, 3)
    assert result["nonfoot_contact_over_1n"] is True and result["first_nonfoot_substep"] == 14
    with pytest.raises(ValueError, match="prefix"):
        receiver_contact_prefix(force, 5)


def test_purity_counts_keep_positive_failure_and_original_labels_separate():
    rows = [
        dict(handoff=True, precise_healthy=True, reward=3.25, nonfoot_contact_over_1n=True),
        dict(handoff=True, precise_healthy=False, reward=-1.0, nonfoot_contact_over_1n=False),
        dict(handoff=False, precise_healthy=False, reward=0.0, nonfoot_contact_over_1n=False),
    ]
    result = summarize_reward_purity(rows)
    assert result["handoff"] == dict(total=2, nonfoot_in_prefix=1, no_nonfoot_observed_in_prefix=1)
    assert result["precise_healthy"]["nonfoot_in_prefix"] == 1
    assert result["positive_reward"]["total"] == 1
