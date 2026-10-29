"""Empirical receiving evidence for a future tactical skill model.

Counts on a public development bank are not calibrated deployment probabilities.
Unknown geometry or unmeasured readiness stays unknown, never a guessed success.
"""

import math
import re
from dataclasses import dataclass
from enum import StrEnum
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


class ReceivingAttemptStatus(StrEnum):
    """A single observed terminal result, or an incomplete observation window."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNSAFE = "unsafe"
    CENSORED = "censored"


@dataclass(frozen=True)
class ReceivingAttempt:
    """External source must authenticate physical evidence and episode identity.

    An episode hash is not itself proof of independence or safety. Replays of
    the same episode must share the same episode hash and cannot be counted twice.
    The caller supplies the context commitment, including relevant world and
    policy conditions; this function cannot authenticate that commitment. For a
    censored attempt, `safe=True` means no safety veto was observed in the
    recorded window, not that the unobserved future was safe.
    """

    observation: ReceivingObservation
    episode_hash: str
    context_hash: str
    status: ReceivingAttemptStatus

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observation, ReceivingObservation)
            or not isinstance(self.episode_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.episode_hash) is None
            or not isinstance(self.context_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.context_hash) is None
            or not isinstance(self.status, ReceivingAttemptStatus)
        ):
            raise ValueError("typed observation, episode hash and status required")
        obs = self.observation
        if self.status == ReceivingAttemptStatus.SUCCESS and not (obs.captured and obs.safe):
            raise ValueError("success requires a safe observed capture")
        if self.status == ReceivingAttemptStatus.FAILURE and (obs.captured or not obs.safe):
            raise ValueError("ordinary failure must be observed, safe and not captured")
        if self.status == ReceivingAttemptStatus.UNSAFE and obs.safe:
            raise ValueError("unsafe result requires recorded safety failure")
        if self.status == ReceivingAttemptStatus.CENSORED and (obs.captured or not obs.safe):
            raise ValueError("censored result cannot hide a capture or known safety failure")


def receiving_capability_censored_table(attempts: tuple[ReceivingAttempt, ...]) -> dict[str, Any]:
    """Exact-context descriptive counts with explicit censoring, never a permission."""
    if type(attempts) is not tuple or not 1 <= len(attempts) <= 100000:
        raise ValueError("bounded explicit attempts required")
    if any(not isinstance(a, ReceivingAttempt) for a in attempts):
        raise ValueError("typed attempts required")
    if len({a.observation.policy_hash for a in attempts}) != 1:
        raise ValueError("different policies require separate capability versions")
    if len({a.observation.evidence_hash for a in attempts}) != len(attempts):
        raise ValueError("duplicate evidence cannot be counted twice")
    if len({a.episode_hash for a in attempts}) != len(attempts):
        raise ValueError("one attempt per episode; do not count replays or windows as independent")
    groups: dict[tuple[str, float, float, str], list[ReceivingAttempt]] = {}
    for attempt in attempts:
        obs = attempt.observation
        groups.setdefault(
            (obs.agent_id, obs.speed_mps, obs.lateral_m, attempt.context_hash), []
        ).append(attempt)
    rows = []
    for (agent, speed, side, context), values in sorted(groups.items()):
        counts = {s.value: sum(a.status == s for a in values) for s in ReceivingAttemptStatus}
        resolved = len(values) - counts[ReceivingAttemptStatus.CENSORED.value]
        success = counts[ReceivingAttemptStatus.SUCCESS.value]
        rows.append(
            {
                "agent_id": agent,
                "speed_mps": speed,
                "lateral_m": side,
                "context_hash": context,
                "attempts": len(values),
                "status_counts": counts,
                "resolved_attempts": resolved,
                "resolved_success_fraction": success / resolved if resolved else None,
                "completion_lower_bound": success / len(values),
                "completion_upper_bound": (success + counts["censored"]) / len(values),
                "duration_sec": None,
                "readiness_after": None,
                "episode_hashes": sorted(a.episode_hash for a in values),
                "evidence_hashes": sorted(a.observation.evidence_hash for a in values),
            }
        )
    result = {
        "schema": "soccer.receiving_capability_censored.v1",
        "policy_hash": attempts[0].observation.policy_hash,
        "rows": rows,
        "scope": "public_coached_development_exact_context_only",
        "bounds_scope": "observed_attempts_only_not_population_confidence_intervals",
        "calibrated_probability_model": False,
        "unseen_context": "UNKNOWN",
        "episode_independence_authenticated": False,
        "context_coverage_authenticated": False,
        "evidence_authenticated_by_this_function": False,
        "hardware_authorized": False,
        "promotion_authorized": False,
    }
    result["manifest_hash"] = hash_json(result)
    return result
