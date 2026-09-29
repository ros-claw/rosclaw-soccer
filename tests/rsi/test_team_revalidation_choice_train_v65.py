"""Paired revalidation chooser preserves per-scene safety accounting."""

import numpy as np
import pytest

from scripts.rsi_team_revalidation_choice_train_v65 import _newly_unsafe


def test_newly_unsafe_counts_only_old_safe_new_unsafe() -> None:
    labels = np.zeros((3, 3, 3), dtype=np.bool_)
    labels[0, 1, 0] = True
    labels[1, 1, 0] = True
    labels[1, 2, 0] = True
    labels[2, 2, 0] = True
    assert _newly_unsafe(np.asarray([2, 2, 1]), labels) == 1
    assert _newly_unsafe(np.asarray([1, 1, 1]), labels) == 0
    with pytest.raises(ValueError):
        _newly_unsafe(np.asarray([1, 2]), labels)
