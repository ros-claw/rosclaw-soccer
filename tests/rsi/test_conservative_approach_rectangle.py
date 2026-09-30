"""Causal approach learning abstains on unsupported or unsafe comparisons."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.conservative_approach_rectangle import fit_approach_rectangle
from scripts.rsi_train_conservative_approach_rectangle_v288 import cross_validate


def test_rectangle_learns_safe_negative_side_context() -> None:
    context = np.asarray([[2.2, -0.04]] * 8 + [[2.8, -0.10]] * 8)
    gain = np.asarray([2.0] * 8 + [-5.0] * 8)
    clean_loss = np.asarray([False] * 8 + [True] * 8)
    new_out = np.zeros(16, dtype=bool)
    model = fit_approach_rectangle(context, gain, clean_loss, new_out)
    assert model.choose(2.2, -0.04)
    assert not model.choose(2.8, -0.10)
    assert not model.choose(2.2, 0.04)
    with pytest.raises(ValueError):
        model.choose(float("nan"), -0.04)


def test_equal_actions_cannot_pass_grouped_gate() -> None:
    courses = [
        {
            "seed": 100 + seed,
            "lane": lane,
            "frame_zero_context_m": [2.2 + 0.01 * lane, -0.04],
            "baseline_reward": 3.0,
            "candidate_reward": 3.0,
            "baseline_clean": True,
            "candidate_clean": True,
            "baseline_out": False,
            "candidate_out": False,
            "baseline_high_quality": True,
            "candidate_high_quality": True,
        }
        for seed in range(4)
        for lane in range(8)
    ]
    result = cross_validate(courses)
    assert result["selected_count"] == 0
    assert result["development_gate_passed"] is False
