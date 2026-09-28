"""Immutable shared proprioceptive actor for SIM_ONLY first-contact search.

One 6x3 weight matrix is shared across robot lanes and ball courses. The
existing bounded temporal policy and physical joint-limit shield remain in
charge of the actual target. This is a candidate, not a deployed skill.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.temporal_first_touch_policy import (
    FEATURE_NAMES,
    JOINT_NAMES,
    MAX_WEIGHT,
)
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "rsi_isaac_snapshot_shared_temporal_candidate_v1"


def candidate_manifest(
    weights: np.ndarray, *, bank_manifest_hash: str, seed: int
) -> dict[str, Any]:
    if (
        weights.shape != (len(FEATURE_NAMES), len(JOINT_NAMES))
        or not np.isfinite(weights).all()
        or float(np.max(np.abs(weights))) > MAX_WEIGHT
        or type(seed) is not int
        or not 0 <= seed < 2**31
        or not isinstance(bank_manifest_hash, str)
        or not bank_manifest_hash.startswith("sha256:")
        or len(bank_manifest_hash) != 71
    ):
        raise ValueError("invalid bounded shared first-contact candidate")
    result = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "learning_authorized": False,
        "promotion_authorized": False,
        "bank_manifest_hash": bank_manifest_hash,
        "seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "joint_names": list(JOINT_NAMES),
        "weights": weights.tolist(),
    }
    result["candidate_hash"] = hash_json(result)
    return result


def load_candidate(path: Path, *, bank_manifest_hash: str) -> tuple[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "schema",
        "activation_ceiling",
        "partition",
        "learning_authorized",
        "promotion_authorized",
        "bank_manifest_hash",
        "seed",
        "feature_names",
        "joint_names",
        "weights",
        "candidate_hash",
    }:
        raise ValueError("shared first-contact candidate fields changed")
    try:
        weights = np.asarray(data["weights"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid shared first-contact candidate weights") from exc
    expected = candidate_manifest(weights, bank_manifest_hash=bank_manifest_hash, seed=data["seed"])
    if data != expected:
        raise ValueError("shared first-contact candidate commitment changed")
    return expected["candidate_hash"], weights
