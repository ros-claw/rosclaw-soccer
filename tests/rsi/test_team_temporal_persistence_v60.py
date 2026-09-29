"""Sustained alerts cannot accumulate evidence across different episodes."""

import numpy as np
import pytest

from scripts.rsi_team_temporal_persistence_v60 import _sustained


def test_sustained_alert_resets_at_episode_boundary() -> None:
    alert = _sustained(
        np.asarray([0.9, 0.9, 0.9, 0.9, 0.9]),
        np.asarray([0, 0, 1, 1, 1]),
        0.5,
        3,
    )
    assert alert.tolist() == [False, False, False, False, True]


def test_sustained_alert_rejects_nonfinite_probability() -> None:
    with pytest.raises(ValueError):
        _sustained(np.asarray([np.nan]), np.asarray([0]), 0.5, 2)
