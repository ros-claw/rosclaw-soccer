"""The hindsight oracle is diagnostic, not an executable policy."""

import numpy as np
import pytest

from scripts.rsi_taskspace_family_diagnostic import oracle_support


def test_oracle_support_counts_distinct_rescues_and_losses() -> None:
    result = oracle_support(np.array([[1, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]]))
    assert result["oracle_new_rescues"] == [1, 2]
    assert result["family_new_rescues"] == [2]
    assert result["oracle_unselected_rescues"] == [1]
    assert result["up_regressed_parent"] == [0]
    with pytest.raises(ValueError):
        oracle_support(np.zeros((3, 3)))
