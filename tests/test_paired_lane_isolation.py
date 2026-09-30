"""Untreated physical lanes must be checked before assigning causal credit."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.paired_lane_isolation import first_untreated_precontact_drift


def _trajectories() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((5, 2, 7), dtype=np.float32),
        np.zeros((5, 2, 7), dtype=np.float32),
        np.zeros((5, 2, 29), dtype=np.float32),
        np.zeros((5, 2, 29), dtype=np.float32),
    )


def test_untreated_precontact_spillover_blocks_lane_attribution() -> None:
    root_a, root_b, target_a, target_b = _trajectories()
    target_b[2, 0, 4] = 0.01
    root_b[3, 0, 0] = 0.02
    assert first_untreated_precontact_drift(
        root_a,
        root_b,
        target_a,
        target_b,
        np.array([True, False]),
        np.array([4, 4]),
    ) == {0: 2}


def test_postcontact_change_is_not_precontact_spillover() -> None:
    root_a, root_b, target_a, target_b = _trajectories()
    target_b[3, 0, 4] = 0.01
    assert not first_untreated_precontact_drift(
        root_a,
        root_b,
        target_a,
        target_b,
        np.array([True, False]),
        np.array([3, 5]),
    )


def test_invalid_trajectory_fails_closed() -> None:
    root_a, root_b, target_a, target_b = _trajectories()
    root_b[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite paired"):
        first_untreated_precontact_drift(
            root_a,
            root_b,
            target_a,
            target_b,
            np.array([True, False]),
            np.array([4, 4]),
        )
