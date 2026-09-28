"""Out-of-support body states must abstain independently of the learned gate."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.team_memory_support import (
    leave_one_out_support_radius,
    query_support_distance,
)


def test_support_radius_is_frozen_training_only() -> None:
    memories = np.zeros((3, len(ACTION_FEATURE_NAMES)))
    memories[:, 1] = (0.0, 1.0, 2.0)
    radius = leave_one_out_support_radius(memories)
    assert radius == pytest.approx(1.0)
    assert query_support_distance(memories, memories[1]) == 0.0
    assert query_support_distance(memories, np.ones(len(ACTION_FEATURE_NAMES)) * 10) > radius


def test_support_rejects_nonfinite() -> None:
    memories = np.zeros((3, len(ACTION_FEATURE_NAMES)))
    memories[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        leave_one_out_support_radius(memories)
