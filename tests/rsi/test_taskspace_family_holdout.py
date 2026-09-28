"""A fresh paired verdict cannot hide lost parent successes."""

import numpy as np
import pytest

from scripts.rsi_taskspace_family_holdout import score_pair


def test_score_pair_reports_rescue_and_regression_separately() -> None:
    assert score_pair(np.array([1, 0, 1]), np.array([0, 1, 1])) == {
        "parent_clean": 2,
        "candidate_clean": 2,
        "rescued_lane_indices": [1],
        "regressed_lane_indices": [0],
    }
    with pytest.raises(ValueError):
        score_pair(np.array([1, 0]), np.array([1]))
