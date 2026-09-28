"""Authenticated SIM_ONLY memory actor for delayed swing-foot acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.taskspace_gate_memory import (
    BASELINE_CLEAN_CEILINGS,
    CONFIDENCE_MULTIPLIERS,
    NEIGHBOR_COUNTS,
)
from rosclaw_soccer.sim.contracts import hash_json


def load_late_swing_actor(path: Path) -> dict[str, Any]:
    actor = json.loads(path.read_text(encoding="utf-8"))
    if (
        actor.get("schema") != "rsi_late_swing_memory_actor_v10b"
        or actor.get("activation_ceiling") != "SIM_ONLY"
        or actor.get("action_forward_m") != 0.08
        or actor.get("action_vertical_offset_m") != 0.04
        or actor.get("action_lateral_cap_m") != 0.05
        or actor.get("action_acquisition_max_gap_m") != 0.55
        or actor.get("feature_names") != list(ACTION_FEATURE_NAMES)
        or actor.get("neighbors") not in NEIGHBOR_COUNTS
        or actor.get("confidence") not in CONFIDENCE_MULTIPLIERS
        or actor.get("baseline_clean_ceiling") not in BASELINE_CLEAN_CEILINGS
        or actor.get("holdout_open_authorized") is not True
        or actor.get("promotion_authorized") is not False
        or actor.get("actor_hash")
        != hash_json({key: value for key, value in actor.items() if key != "actor_hash"})
    ):
        raise ValueError("unauthenticated late-swing SIM_ONLY actor")
    memory = np.asarray(actor.get("memory_features"), dtype=np.float64)
    clean = np.asarray(actor.get("memory_clean"), dtype=np.float64)
    reward = np.asarray(actor.get("memory_reward"), dtype=np.float64)
    groups = np.asarray(actor.get("memory_groups"), dtype=np.int64)
    time_weights = np.asarray(actor.get("contact_time_weights"), dtype=np.float64)
    if (
        memory.shape != (78, len(ACTION_FEATURE_NAMES))
        or clean.shape != (78, 2)
        or reward.shape != clean.shape
        or groups.shape != (78,)
        or len(np.unique(groups)) != 10
        or time_weights.shape != (6,)
        or not all(np.isfinite(value).all() for value in (memory, clean, reward, time_weights))
        or not np.isin(clean, (0, 1)).all()
    ):
        raise ValueError("invalid ten-source late-swing memory")
    authenticated: dict[str, Any] = actor
    return authenticated
