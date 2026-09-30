"""Causal approach learning abstains on unsupported or unsafe comparisons."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.conservative_approach_rectangle import (
    GUARDED_CV_HASH,
    fit_approach_rectangle,
    load_guarded_approach_policy,
)
from rosclaw_soccer.sim.contracts import hash_json
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


def test_guarded_policy_loader_is_json_only_and_hash_bound(tmp_path) -> None:
    import json

    policy = {
        "schema": "rsi_isaac_guarded_approach_policy_v1",
        "activation_ceiling": "SIM_ONLY",
        "cv_report_hash": GUARDED_CV_HASH,
        "x_max_m": 2.5524194955825807,
        "y_min_m": -0.09402785405516624,
        "navigation_lateral_ball_gain": 0.8,
        "promotion_authorized": False,
    }
    policy["policy_hash"] = hash_json(policy)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    model, digest = load_guarded_approach_policy(path)
    assert digest == policy["policy_hash"]
    assert model.choose(2.3, -0.04)
    assert not model.choose(2.7, -0.04)
    policy["x_max_m"] = 2.7
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError):
        load_guarded_approach_policy(path)
