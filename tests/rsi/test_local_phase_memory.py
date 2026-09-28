"""Paired-contact muscle memory must fall back when evidence is weak."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.local_phase_memory import load_local_phase_actor, select_local_phase
from rosclaw_soccer.sim.contracts import hash_json


def test_local_memory_selects_only_supported_paired_gain() -> None:
    memory = np.zeros((10, len(ACTION_FEATURE_NAMES)))
    memory[:, 0] = 1
    groups = np.repeat(np.arange(5), 2)
    clean = np.zeros((10, 3))
    clean[:, 2] = 1
    reward = np.zeros((10, 3))
    reward[:, 2] = 1
    action = select_local_phase(
        memory[:1], memory, clean, reward, groups, neighbors=5, confidence=1.0
    )
    np.testing.assert_array_equal(action, [6])
    clean[:, 2] = 0
    assert (
        select_local_phase(memory[:1], memory, clean, reward, groups, neighbors=5, confidence=1.0)[
            0
        ]
        == 0
    )
    with pytest.raises(ValueError):
        select_local_phase(memory[:1], memory, clean, reward, groups, neighbors=2, confidence=1)


def test_local_actor_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    manifest = {
        "schema": "rsi_local_contact_phase_actor_v4",
        "activation_ceiling": "SIM_ONLY",
        "phase_actions_frames": [0.0, 3.0, 6.0],
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "neighbors": 5,
        "confidence": 1.0,
        "memory_features": np.zeros((35, len(ACTION_FEATURE_NAMES))).tolist(),
        "memory_clean": np.zeros((35, 3)).tolist(),
        "memory_reward": np.zeros((35, 3)).tolist(),
        "memory_groups": np.repeat(np.arange(5), 7).tolist(),
        "contact_time_weights": np.zeros(6).tolist(),
        "promotion_authorized": False,
    }
    manifest["actor_hash"] = hash_json(manifest)
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_local_phase_actor(path)["actor_hash"] == manifest["actor_hash"]
    manifest["memory_clean"][0][2] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        load_local_phase_actor(path)
