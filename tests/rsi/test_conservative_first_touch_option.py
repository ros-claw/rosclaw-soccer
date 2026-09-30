"""Conservative learned first-touch option stays causal and out-of-support safe."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.conservative_first_touch_option import fit_conservative_option_value
from scripts.rsi_train_independent_option_curriculum import ARMS, SEEDS, cross_validate


def test_learned_option_changes_only_supported_decisions() -> None:
    features = np.zeros((32, 9))
    features[:, 1] = np.linspace(-1.0, 1.0, 32)
    rewards = np.column_stack((np.zeros(32), 1.0 + features[:, 1], np.full(32, 0.5)))
    model = fit_conservative_option_value(features, rewards)
    assert model.fixed == 1
    assert model.choose(features[0]) == 2
    assert model.choose(features[-1]) == 1
    far = features[0].copy()
    far[2] = 100.0
    assert model.choose(far) == model.fixed
    with pytest.raises(ValueError):
        model.choose(np.full(9, np.nan))


def test_option_fit_rejects_incomplete_or_nonfinite_courses() -> None:
    features = np.zeros((32, 9))
    rewards = np.zeros((32, 3))
    with pytest.raises(ValueError):
        fit_conservative_option_value(features[:23], rewards[:23])
    with pytest.raises(ValueError):
        fit_conservative_option_value(features, rewards[:, :2])
    rewards[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_conservative_option_value(features, rewards)


def test_cross_validation_does_not_promote_equal_options() -> None:
    episodes = []
    for seed in SEEDS:
        for lane in range(0, 16, 2):
            episodes.append(
                {
                    "seed": seed,
                    "lane": lane,
                    "arms": {
                        arm: {"clean_foot_only": True, "maximum_lateral_excursion_m": 0.0}
                        for arm in ARMS
                    },
                }
            )
    features = np.zeros((64, 9))
    rewards = np.ones((64, 3))
    result = cross_validate({"report_hash": "synthetic", "episodes": episodes}, features, rewards)
    assert result["winning_seed_folds"] == 0
    assert result["all_folds_safe"] is True
    assert result["development_gate_passed"] is False
    assert result["fresh_seed_20260966_authorized"] is False
