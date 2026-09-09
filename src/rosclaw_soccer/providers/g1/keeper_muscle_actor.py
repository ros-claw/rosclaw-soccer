"""Numeric-only SIM_ONLY behavioral-cloning actor for G1 goalkeeper arms.

Inputs are causal intercept and measured proprioception. Outputs are fourteen
arm joint targets, never ball/root state or hardware authority. A physics
controller must still project targets and limit actuator torques.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

OBSERVATION_NAMES = (
    "causal_intercept_time",
    "causal_intercept_y",
    "causal_intercept_z",
    "pelvis_height",
    *(f"gravity.{a}" for a in "xyz"),
    *(f"joint_position.{n}" for n in G1_DDS_JOINT_NAMES),
    *(f"joint_velocity.{n}" for n in G1_DDS_JOINT_NAMES),
)
CONTRACT_HASH = str(hash_json({"names": OBSERVATION_NAMES, "schema": "keeper-muscle-input.v1"}))


def muscle_observation(
    intercept: np.ndarray, height: float, gravity: np.ndarray, q: np.ndarray, dq: np.ndarray
) -> np.ndarray:
    raw = np.r_[intercept, height, gravity, q, dq].astype(np.float64)
    if raw.shape != (65,) or not np.isfinite(raw).all():
        raise ValueError("keeper muscle input requires 65 finite causal/proprioceptive values")
    scales = np.r_[2.0, 2.0, 1.0, 1.0, np.ones(3), np.ones(29) / 3, np.ones(29) * 0.05]
    return np.asarray(np.clip(raw * scales, -5, 5), dtype=np.float64)


class KeeperMuscleActor:
    def __init__(self, path: Path) -> None:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("keeper muscle artifact is missing or oversized")
        raw = path.read_bytes()
        payload: dict[str, Any] = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("keeper muscle artifact must be an object")
        if (
            payload.get("schema") != "keeper-muscle-bc.v1"
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("promotion_authorized") is not False
            or payload.get("observation_contract_hash") != CONTRACT_HASH
            or payload.get("joint_names") != list(G1_DDS_JOINT_NAMES[15:])
        ):
            raise ValueError("keeper muscle artifact contract/authority mismatch")
        layers = payload.get("layers", [])
        shapes = ((64, 65), (64, 64), (14, 64))
        if not isinstance(layers, list) or len(layers) != 3:
            raise ValueError("keeper muscle actor requires the audited three-layer topology")
        self.layers = []
        for layer, shape in zip(layers, shapes, strict=True):
            if not isinstance(layer, dict) or not {"weight", "bias"} <= layer.keys():
                raise ValueError("keeper muscle layer requires weight and bias")
            w, b = np.asarray(layer["weight"], dtype=float), np.asarray(layer["bias"], dtype=float)
            if (
                w.shape != shape
                or b.shape != (shape[0],)
                or not np.isfinite(w).all()
                or not np.isfinite(b).all()
            ):
                raise ValueError("invalid keeper muscle tensor")
            if np.max(np.abs(w)) > 20 or np.max(np.abs(b)) > 20:
                raise ValueError("keeper muscle weights outside bounded numeric envelope")
            self.layers.append((w, b))
        reference_only = payload.get("task_reference_only", False)
        if type(reference_only) is not bool or (
            reference_only and np.any(self.layers[0][0][:, 4:] != 0)
        ):
            raise ValueError("task reference artifact must exclude body feedback weights")
        self.policy_hash = str(hash_bytes(raw))
        self.metadata = payload

    def target(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (65,) or not np.isfinite(x).all() or np.max(np.abs(x)) > 5:
            raise ValueError("invalid keeper muscle observation")
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        return np.asarray(3.0 * x, dtype=np.float64)
