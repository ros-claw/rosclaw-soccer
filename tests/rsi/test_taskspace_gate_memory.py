"""Paired task-space memory abstains without independent net-benefit evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.taskspace_gate_memory import (
    load_taskspace_gate_actor,
    select_taskspace_gate,
)
from rosclaw_soccer.sim.contracts import hash_json


def test_taskspace_memory_requires_clean_gain_and_preserves_good_parent() -> None:
    memory = np.zeros((6, len(ACTION_FEATURE_NAMES)))
    groups = np.repeat(np.arange(3), 2)
    clean = np.zeros((6, 2))
    clean[:, 1] = 1
    reward = np.zeros((6, 2))
    reward[:, 1] = 1
    assert select_taskspace_gate(
        memory[:1],
        memory,
        clean,
        reward,
        groups,
        neighbors=5,
        confidence=1.0,
        baseline_clean_ceiling=0.4,
    )[0]
    clean[:, 0] = 1
    assert not select_taskspace_gate(
        memory[:1],
        memory,
        clean,
        reward,
        groups,
        neighbors=5,
        confidence=1.0,
        baseline_clean_ceiling=0.4,
    )[0]


def test_taskspace_actor_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    manifest = {
        "schema": "rsi_taskspace_gate_actor_v8",
        "activation_ceiling": "SIM_ONLY",
        "frozen_taskspace_action_forward_m": 0.08,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "neighbors": 5,
        "confidence": 1.0,
        "baseline_clean_ceiling": 0.4,
        "memory_features": np.zeros((32, len(ACTION_FEATURE_NAMES))).tolist(),
        "memory_clean": np.zeros((32, 2)).tolist(),
        "memory_reward": np.zeros((32, 2)).tolist(),
        "memory_groups": np.repeat(np.arange(4), 8).tolist(),
        "contact_time_weights": np.zeros(6).tolist(),
        "promotion_authorized": False,
    }
    manifest["actor_hash"] = hash_json(manifest)
    path = tmp_path / "taskspace.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_taskspace_gate_actor(path)["actor_hash"] == manifest["actor_hash"]
    manifest["memory_clean"][0][1] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        load_taskspace_gate_actor(path)
