"""Late-swing memory actors are SIM_ONLY and content authenticated."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.sim.contracts import hash_json


def test_late_swing_actor_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "late.json"
    actor = {
        "schema": "rsi_late_swing_memory_actor_v10b",
        "activation_ceiling": "SIM_ONLY",
        "action_forward_m": 0.08,
        "action_vertical_offset_m": 0.04,
        "action_lateral_cap_m": 0.05,
        "action_acquisition_max_gap_m": 0.55,
        "feature_names": list(ACTION_FEATURE_NAMES),
        "neighbors": 3,
        "confidence": 0.5,
        "baseline_clean_ceiling": 0.6,
        "holdout_open_authorized": True,
        "promotion_authorized": False,
        "memory_features": np.zeros((78, len(ACTION_FEATURE_NAMES))).tolist(),
        "memory_clean": np.zeros((78, 2)).tolist(),
        "memory_reward": np.zeros((78, 2)).tolist(),
        "memory_groups": np.repeat(np.arange(10), [8] * 8 + [6, 8]).tolist(),
        "contact_time_weights": np.zeros(6).tolist(),
    }
    actor["actor_hash"] = hash_json(actor)
    path.write_text(json.dumps(actor), encoding="utf-8")
    assert load_late_swing_actor(path)["actor_hash"] == actor["actor_hash"]
    actor["action_acquisition_max_gap_m"] = 0.35
    path.write_text(json.dumps(actor), encoding="utf-8")
    with pytest.raises(ValueError):
        load_late_swing_actor(path)
