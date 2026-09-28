"""Multi-action memory cannot infer its own outcomes or ignore safety regressions."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.taskspace_family_memory import (
    load_taskspace_family_actor,
    select_taskspace_family,
)
from rosclaw_soccer.sim.contracts import hash_json


def test_family_selects_supported_rescue_and_abstains_on_regression() -> None:
    query = np.zeros((1, 9))
    memory = np.zeros((5, 9))
    clean = np.zeros((5, 3))
    clean[:3, 1] = 1
    reward = clean.copy()
    groups = np.array([0, 1, 2, 3, 4])
    assert select_taskspace_family(
        query, memory, clean, reward, groups, neighbors=3, confidence=0.5
    ).tolist() == [1]
    clean[0] = [1, 0, 0]
    assert select_taskspace_family(
        query, memory, clean, reward, groups, neighbors=3, confidence=0.5
    ).tolist() == [0]
    with pytest.raises(ValueError):
        select_taskspace_family(query, memory, clean, reward, groups, neighbors=2, confidence=0.5)


def test_family_actor_rejects_tampered_memory(tmp_path: Path) -> None:
    path = tmp_path / "actor.json"
    actor = {
        "schema": "rsi_taskspace_family_actor_v9b",
        "activation_ceiling": "SIM_ONLY",
        "action_names": ["parent", "up", "wide"],
        "feature_names": [
            "bias",
            "gap_x_scaled",
            "gap_y_scaled",
            "ball_vx_scaled",
            "root_vx_scaled",
            "sin_predicted_gait_phase",
            "cos_predicted_gait_phase",
            "lateral_times_sin_phase",
            "lateral_times_cos_phase",
        ],
        "neighbors": 3,
        "confidence": 0.5,
        "holdout_open_authorized": True,
        "promotion_authorized": False,
        "memory_features": np.zeros((64, 9)).tolist(),
        "memory_clean": np.zeros((64, 3)).tolist(),
        "memory_reward": np.zeros((64, 3)).tolist(),
        "memory_groups": np.repeat(np.arange(8), 8).tolist(),
        "contact_time_weights": np.zeros(6).tolist(),
    }
    actor["actor_hash"] = hash_json(actor)
    path.write_text(json.dumps(actor), encoding="utf-8")
    assert load_taskspace_family_actor(path)["actor_hash"] == actor["actor_hash"]
    actor["memory_clean"][0][1] = 1.0
    path.write_text(json.dumps(actor), encoding="utf-8")
    with pytest.raises(ValueError):
        load_taskspace_family_actor(path)
