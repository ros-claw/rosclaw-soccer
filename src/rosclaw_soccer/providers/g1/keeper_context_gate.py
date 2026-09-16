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
    def __init__(
        self,
        path: Path,
        *,
        parent_policy_hash: str,
        config_hash: str,
        explicit_config: dict[str, Any] | None = None,
    ) -> None:
        # Keep literal current identities unchanged. An explicitly supplied
        # configuration may also match the two historical serialized shapes:
        # reach floor was added after intercept floor. Only exact historical
        # defaults may be omitted; changed thresholds never inherit an old gate.
        hashes = {config_hash}
        if explicit_config is not None:
            if (
                not isinstance(explicit_config, dict)
                or gate_config_hash(explicit_config) != config_hash
            ):
                raise ValueError("explicit keeper configuration differs from its declared hash")
            legacy = dict(explicit_config)
            if (
                type(legacy.get("minimum_reach_height_m")) in (int, float)
                and legacy["minimum_reach_height_m"] == 0.72
            ):
                legacy.pop("minimum_reach_height_m")
                hashes.add(gate_config_hash(legacy))
                if (
                    type(legacy.get("minimum_intercept_height_m")) in (int, float)
                    and legacy["minimum_intercept_height_m"] == 0.65
                ):
                    legacy.pop("minimum_intercept_height_m")
                    hashes.add(gate_config_hash(legacy))
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
            or not isinstance(p.get("config_hash"), str)
            or p["config_hash"] not in hashes
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
        self.bound_config_hash = p["config_hash"]
        self.config_binding = (
            "EXACT" if self.bound_config_hash == config_hash else "LEGACY_DEFAULT_HEIGHTS"
        )

    def select_reach(self, causal_height_m: float) -> bool:
        if not np.isfinite(causal_height_m) or not 0.5 <= causal_height_m <= 2:
            raise ValueError("keeper gate requires bounded causal context")
        return bool(self.weight * (causal_height_m - self.center) / self.scale + self.bias >= 0)
