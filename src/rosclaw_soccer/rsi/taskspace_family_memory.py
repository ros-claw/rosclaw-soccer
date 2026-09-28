"""Conservative multi-action selection from paired SIM_ONLY physical episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.sim.contracts import hash_json

ACTION_NAMES = ("parent", "up", "wide")
NEIGHBORS = (3, 5, 7)
CONFIDENCE = (0.0, 0.5, 1.0)
DISTANCE_COLUMNS = (1, 2, 3, 4, 5, 6)


def select_taskspace_family(
    query: np.ndarray[Any, Any],
    memory: np.ndarray[Any, Any],
    clean: np.ndarray[Any, Any],
    reward: np.ndarray[Any, Any],
    groups: np.ndarray[Any, Any],
    *,
    neighbors: int,
    confidence: float,
) -> np.ndarray[Any, Any]:
    """Abstain unless independent local memories support a clean, safe action."""
    if (
        query.ndim != 2
        or query.shape[1] != len(ACTION_FEATURE_NAMES)
        or memory.shape != (len(memory), len(ACTION_FEATURE_NAMES))
        or clean.shape != (len(memory), len(ACTION_NAMES))
        or reward.shape != clean.shape
        or groups.shape != (len(memory),)
        or len(memory) < neighbors
        or neighbors not in NEIGHBORS
        or confidence not in CONFIDENCE
        or not all(np.isfinite(value).all() for value in (query, memory, clean, reward))
        or not np.isin(clean, (0, 1)).all()
    ):
        raise ValueError("invalid paired task-space action memory")
    action = np.zeros(len(query), dtype=np.int64)
    for index, row in enumerate(query):
        distances = np.linalg.norm(
            memory[:, DISTANCE_COLUMNS] - row[list(DISTANCE_COLUMNS)], axis=1
        )
        selected: list[int] = []
        per_group: dict[int, int] = {}
        for position in np.argsort(distances):
            group = int(groups[position])
            if per_group.get(group, 0) >= 2:
                continue
            selected.append(int(position))
            per_group[group] = per_group.get(group, 0) + 1
            if len(selected) == neighbors:
                break
        if len(selected) < neighbors or len(per_group) < 2:
            continue
        candidates = []
        for candidate in (1, 2):
            losses = (clean[selected, 0] == 1) & (clean[selected, candidate] == 0)
            if np.any(losses):
                continue
            delta = clean[selected, candidate] - clean[selected, 0]
            lower = float(np.mean(delta) - confidence * np.std(delta, ddof=1) / np.sqrt(neighbors))
            reward_gain = float(np.mean(reward[selected, candidate] - reward[selected, 0]))
            if lower > 0 and reward_gain > 0:
                candidates.append((lower, reward_gain, candidate))
        if candidates:
            action[index] = max(candidates)[2]
    return action


def load_taskspace_family_actor(path: Path) -> dict[str, Any]:
    """Reject tampered or non-development-qualified SIM_ONLY action memories."""
    actor = json.loads(path.read_text(encoding="utf-8"))
    if (
        actor.get("schema") != "rsi_taskspace_family_actor_v9b"
        or actor.get("activation_ceiling") != "SIM_ONLY"
        or actor.get("action_names") != list(ACTION_NAMES)
        or actor.get("feature_names") != list(ACTION_FEATURE_NAMES)
        or actor.get("neighbors") not in NEIGHBORS
        or actor.get("confidence") not in CONFIDENCE
        or actor.get("holdout_open_authorized") is not True
        or actor.get("promotion_authorized") is not False
        or actor.get("actor_hash")
        != hash_json({key: value for key, value in actor.items() if key != "actor_hash"})
    ):
        raise ValueError("unauthenticated SIM_ONLY task-space family actor")
    memory = np.asarray(actor.get("memory_features"), dtype=np.float64)
    clean = np.asarray(actor.get("memory_clean"), dtype=np.float64)
    reward = np.asarray(actor.get("memory_reward"), dtype=np.float64)
    groups = np.asarray(actor.get("memory_groups"), dtype=np.int64)
    time_weights = np.asarray(actor.get("contact_time_weights"), dtype=np.float64)
    if (
        memory.shape != (64, len(ACTION_FEATURE_NAMES))
        or clean.shape != (64, len(ACTION_NAMES))
        or reward.shape != clean.shape
        or groups.shape != (64,)
        or len(np.unique(groups)) != 8
        or time_weights.shape != (6,)
        or not all(np.isfinite(value).all() for value in (memory, clean, reward, time_weights))
        or not np.isin(clean, (0, 1)).all()
    ):
        raise ValueError("invalid task-space family physical memory")
    authenticated: dict[str, Any] = actor
    return authenticated
