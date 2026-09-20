"""Empirical receiving evidence for a future tactical skill model.

Counts on a public development bank are not calibrated deployment probabilities.
Unknown geometry or unmeasured readiness stays unknown, never a guessed success.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class ReceivingObservation:
    agent_id: str
    speed_mps: float
    lateral_m: float
    policy_hash: str
    evidence_hash: str
    captured: bool
    safe: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or re.fullmatch(
                r"(?:red|blue)\.(?:defender|finisher|goalkeeper|playmaker)", self.agent_id
            )
            is None
            or any(
                not isinstance(h, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
                for h in (self.policy_hash, self.evidence_hash)
            )
            or type(self.captured) is not bool
            or type(self.safe) is not bool
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (self.speed_mps, self.lateral_m)
            )
            or not 0.5 <= self.speed_mps <= 3.0
            or abs(self.lateral_m) > 0.3
        ):
            raise ValueError("typed measured receiving course required")


def receiving_capability_table(observations: tuple[ReceivingObservation, ...]) -> dict[str, Any]:
    """Build exact-context empirical rows from externally authenticated courses."""
    if type(observations) is not tuple or not 1 <= len(observations) <= 100000:
        raise ValueError("bounded explicit observations required")
    if any(not isinstance(r, ReceivingObservation) for r in observations):
        raise ValueError("typed observations required")
    if len({r.policy_hash for r in observations}) != 1:
        raise ValueError("different team policies require separate capability versions")
    if len({r.evidence_hash for r in observations}) != len(observations):
        raise ValueError("replays and duplicate evidence cannot inflate sample count")
    groups: dict[tuple[str, float, float], list[ReceivingObservation]] = {}
    for row in observations:
        groups.setdefault((row.agent_id, row.speed_mps, row.lateral_m), []).append(row)
    rows = []
    for (agent, speed, side), values in sorted(groups.items()):
        successes = sum(r.captured and r.safe for r in values)
        rows.append(
            {
                "agent_id": agent,
                "speed_mps": speed,
                "lateral_m": side,
                "trials": len(values),
                "safe_captures": successes,
                "observed_fraction": successes / len(values),
                "duration_sec": None,
                "readiness_after": None,
                "evidence_hashes": sorted(r.evidence_hash for r in values),
            }
        )
    result = {
        "schema": "soccer.receiving_capability_observations.v1",
        "policy_hash": observations[0].policy_hash,
        "rows": rows,
        "scope": "public_coached_development_exact_context_only",
        "calibrated_probability_model": False,
        "unseen_context": "UNKNOWN",
        "evidence_authenticated_by_this_function": False,
        "promotion_authorized": False,
    }
    result["manifest_hash"] = hash_json(result)
    return result
