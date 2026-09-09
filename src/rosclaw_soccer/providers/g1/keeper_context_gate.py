"""SIM_ONLY learned selection between a reference and its reach correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def gate_config_hash(config: dict[str, Any]) -> str:
    base = dict(config)
    for key in ("muscle_gate_path", "muscle_reach_correction", "muscle_actor_path"):
        base.pop(key, None)
    return str(hash_json(base))


class KeeperContextGate:
    def __init__(self, path: Path, *, parent_policy_hash: str, config_hash: str) -> None:
        if not path.is_file() or path.stat().st_size > 100_000:
            raise ValueError("missing or oversized keeper context gate")
        raw = path.read_bytes()
        p = json.loads(raw)
        if (
            not isinstance(p, dict)
            or p.get("schema") != "keeper-context-gate.v1"
            or p.get("activation_ceiling") != "SIM_ONLY"
            or p.get("promotion_authorized") is not False
            or p.get("parent_policy_hash") != parent_policy_hash
            or p.get("config_hash") != config_hash
        ):
            raise ValueError("keeper gate authority or parent mismatch")
        try:
            self.center, self.scale, self.weight, self.bias = (
                float(p[k]) for k in ("center", "scale", "weight", "bias")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid keeper gate numeric fields") from exc
        if (
            not np.isfinite((self.center, self.scale, self.weight, self.bias)).all()
            or not 0.5 <= self.center <= 2
            or not 0.01 <= self.scale <= 1
            or abs(self.weight) > 100
            or abs(self.bias) > 100
        ):
            raise ValueError("invalid bounded keeper gate weights")
        self.policy_hash = str(hash_bytes(raw))

    def select_reach(self, causal_height_m: float) -> bool:
        if not np.isfinite(causal_height_m) or not 0.5 <= causal_height_m <= 2:
            raise ValueError("keeper gate requires bounded causal context")
        return bool(self.weight * (causal_height_m - self.center) / self.scale + self.bias >= 0)
