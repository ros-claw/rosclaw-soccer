"""Guarded phase selection preserves a locally reliable parent action."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.baseline_retention_phase import (
    load_guarded_phase_actor,
    select_guarded_phase,
)
from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.sim.contracts import hash_json


def test_guard_abstains_when_parent_is_often_clean() -> None:
    memory = np.zeros((10, len(ACTION_FEATURE_NAMES)))
    groups = np.repeat(np.arange(5), 2)
    clean = np.zeros((10, 3))
    clean[:5, 0] = 1
    clean[:, 2] = 1
    reward = np.zeros((10, 3))
    reward[:, 2] = 1
    assert (
        select_guarded_phase(
            memory[:1],
            memory,
            clean,
            reward,
            groups,
            neighbors=10,
            confidence=1.0,
            baseline_clean_ceiling=0.4,
        )[0]
        == 0
    )
    clean[:, 0] = 0
    assert (
        select_guarded_phase(
            memory[:1],
            memory,
            clean,
            reward,
            groups,
            neighbors=10,
            confidence=1.0,
            baseline_clean_ceiling=0.4,
        )[0]
        == 6
    )


def test_guarded_actor_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    manifest = {
        "schema": "rsi_baseline_retention_phase_actor_v5",
        "activation_ceiling": "SIM_ONLY",
        "phase_actions_frames": [0.0, 3.0, 6.0],
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "neighbors": 10,
        "confidence": 1.0,
        "baseline_clean_ceiling": 0.5,
        "memory_features": np.zeros((35, len(ACTION_FEATURE_NAMES))).tolist(),
        "memory_clean": np.zeros((35, 3)).tolist(),
        "memory_reward": np.zeros((35, 3)).tolist(),
        "memory_groups": np.repeat(np.arange(5), 7).tolist(),
        "contact_time_weights": np.zeros(6).tolist(),
        "promotion_authorized": False,
    }
    manifest["actor_hash"] = hash_json(manifest)
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_guarded_phase_actor(path)["actor_hash"] == manifest["actor_hash"]
    manifest["baseline_clean_ceiling"] = 0.6
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        load_guarded_phase_actor(path)
