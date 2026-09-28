"""Paired-physics SIM_ONLY memory gate for bounded G1 foot correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.local_phase_memory import DISTANCE_COLUMNS
from rosclaw_soccer.sim.contracts import hash_json

NEIGHBOR_COUNTS = (3, 5, 7)
CONFIDENCE_MULTIPLIERS = (0.0, 0.5, 1.0)
BASELINE_CLEAN_CEILINGS = (0.25, 0.4, 0.6)


def select_taskspace_gate(
    query: np.ndarray[Any, Any],
    memories: np.ndarray[Any, Any],
    clean: np.ndarray[Any, Any],
    reward: np.ndarray[Any, Any],
    groups: np.ndarray[Any, Any],
    *,
    neighbors: int,
    confidence: float,
    baseline_clean_ceiling: float,
) -> np.ndarray[Any, Any]:
    if (
        query.ndim != 2
        or query.shape[1] != len(ACTION_FEATURE_NAMES)
        or memories.ndim != 2
        or memories.shape[1] != len(ACTION_FEATURE_NAMES)
        or len(memories) < neighbors
        or clean.shape != (len(memories), 2)
        or reward.shape != (len(memories), 2)
        or groups.shape != (len(memories),)
        or neighbors not in NEIGHBOR_COUNTS
        or confidence not in CONFIDENCE_MULTIPLIERS
        or baseline_clean_ceiling not in BASELINE_CLEAN_CEILINGS
        or not np.isfinite(query).all()
        or not np.isfinite(memories).all()
        or not np.isfinite(reward).all()
        or not np.isin(clean, (0, 1)).all()
    ):
        raise ValueError("invalid paired task-space memory or query")
    choose = np.zeros(len(query), dtype=np.bool_)
    columns = np.asarray(DISTANCE_COLUMNS)
    for index, row in enumerate(query):
        distances = np.linalg.norm(memories[:, columns] - row[columns], axis=1)
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
        if len(selected) != neighbors or len(per_group) < 2:
            continue
        local_clean = clean[selected]
        if float(np.mean(local_clean[:, 0])) > baseline_clean_ceiling:
            continue
        delta = local_clean[:, 1] - local_clean[:, 0]
        lower = float(np.mean(delta) - confidence * np.std(delta, ddof=1) / np.sqrt(neighbors))
        reward_gain = float(np.mean(reward[selected, 1] - reward[selected, 0]))
        choose[index] = lower > 0 and reward_gain > 0
    return choose


def load_taskspace_gate_actor(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rsi_taskspace_gate_actor_v8"
        or manifest.get("activation_ceiling") != "SIM_ONLY"
        or manifest.get("frozen_taskspace_action_forward_m") != 0.08
        or manifest.get("action_feature_names") != list(ACTION_FEATURE_NAMES)
        or manifest.get("neighbors") not in NEIGHBOR_COUNTS
        or manifest.get("confidence") not in CONFIDENCE_MULTIPLIERS
        or manifest.get("baseline_clean_ceiling") not in BASELINE_CLEAN_CEILINGS
        or manifest.get("promotion_authorized") is not False
        or manifest.get("actor_hash")
        != hash_json({key: value for key, value in manifest.items() if key != "actor_hash"})
    ):
        raise ValueError("unauthenticated SIM_ONLY task-space gate actor")
    memories = np.asarray(manifest.get("memory_features"), dtype=np.float64)
    clean = np.asarray(manifest.get("memory_clean"), dtype=np.float64)
    reward = np.asarray(manifest.get("memory_reward"), dtype=np.float64)
    groups = np.asarray(manifest.get("memory_groups"), dtype=np.int64)
    time_weights = np.asarray(manifest.get("contact_time_weights"), dtype=np.float64)
    if (
        memories.ndim != 2
        or memories.shape[1] != len(ACTION_FEATURE_NAMES)
        or not 32 <= len(memories) <= 512
        or clean.shape != (len(memories), 2)
        or reward.shape != (len(memories), 2)
        or groups.shape != (len(memories),)
        or time_weights.shape != (6,)
        or not np.isfinite(memories).all()
        or not np.isfinite(clean).all()
        or not np.isfinite(reward).all()
        or not np.isfinite(time_weights).all()
        or not np.isin(clean, (0, 1)).all()
    ):
        raise ValueError("invalid task-space gate memories")
    authenticated: dict[str, Any] = manifest
    return authenticated
