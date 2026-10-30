"""Exact-context physical-skill evidence for tactical research, not control authority."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}\Z")


@dataclass(frozen=True)
class PhysicalSkillTrial:
    """One caller-authenticated physical case, not a replay or imputed outcome."""

    group_id: str
    skill_id: str
    context_hash: str
    body_hash: str
    policy_hash: str
    evidence_hash: str
    outcome: Literal["SUCCESS", "FAILURE", "UNKNOWN"]
    safe: bool | None
    duration_sec: float | None = None
    successor_ready: bool | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.group_id, str)
            or not _ID.fullmatch(self.group_id)
            or not isinstance(self.skill_id, str)
            or not _ID.fullmatch(self.skill_id)
            or any(
                not isinstance(v, str) or not _HASH.fullmatch(v)
                for v in (self.context_hash, self.body_hash, self.policy_hash, self.evidence_hash)
            )
            or self.outcome not in ("SUCCESS", "FAILURE", "UNKNOWN")
            or (self.safe is not None and type(self.safe) is not bool)
            or (self.successor_ready is not None and type(self.successor_ready) is not bool)
            or (self.outcome == "SUCCESS" and self.safe is not True)
            or (
                self.duration_sec is not None
                and (
                    isinstance(self.duration_sec, bool)
                    or not isinstance(self.duration_sec, (float, int))
                    or not math.isfinite(self.duration_sec)
                    or self.duration_sec < 0
                )
            )
        ):
            raise ValueError("typed complete physical evidence and explicit unknowns required")


@dataclass(frozen=True)
class CapabilityResearchGate:
    """Declared screening parameters, not a real-robot or promotion standard."""

    minimum_groups: int
    minimum_wilson_lower: float

    def __post_init__(self) -> None:
        if (
            type(self.minimum_groups) is not int
            or not 1 <= self.minimum_groups <= 100000
            or isinstance(self.minimum_wilson_lower, bool)
            or not isinstance(self.minimum_wilson_lower, (float, int))
            or not math.isfinite(self.minimum_wilson_lower)
            or not 0 <= self.minimum_wilson_lower <= 1
        ):
            raise ValueError("bounded explicit research gate required")


def _wilson_95(successes: int, trials: int) -> tuple[float, float]:
    """Nominal binomial interval; correlated physical cases invalidate coverage."""
    z = 1.959963984540054
    fraction = successes / trials
    denom = 1 + z * z / trials
    centre = (fraction + z * z / (2 * trials)) / denom
    radius = (
        z * math.sqrt(fraction * (1 - fraction) / trials + z * z / (4 * trials * trials)) / denom
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def summarize_exact_capability(
    trials: tuple[PhysicalSkillTrial, ...], gate: CapabilityResearchGate
) -> dict[str, Any]:
    """Summarize one exact body/policy/context; preserve all unknown evidence.

    Caller must authenticate source receipts and provide honest physical group
    identities. Repeated replays are not new trials. The Wilson interval is
    descriptive, assumes independent Bernoulli cases, and is not calibrated
    when the physical groups are correlated. No authorization follows.
    """
    if type(trials) is not tuple or not trials or type(gate) is not CapabilityResearchGate:
        raise ValueError("nonempty typed physical trials and fixed research gate required")
    if any(type(t) is not PhysicalSkillTrial for t in trials):
        raise ValueError("typed physical trials required")
    first = trials[0]
    identity = (first.skill_id, first.context_hash, first.body_hash, first.policy_hash)
    if any((t.skill_id, t.context_hash, t.body_hash, t.policy_hash) != identity for t in trials):
        raise ValueError("mixed skills, bodies, policies or contexts must not be pooled")
    if len({t.group_id for t in trials}) != len(trials) or len(
        {t.evidence_hash for t in trials}
    ) != len(trials):
        raise ValueError("duplicate physical group or evidence cannot count twice")
    observed = [t for t in trials if t.outcome != "UNKNOWN"]
    successes = sum(t.outcome == "SUCCESS" for t in observed)
    unknown = len(trials) - len(observed)
    safety_unknown = sum(t.safe is None for t in trials)
    unsafe = sum(t.safe is False for t in trials)
    interval = _wilson_95(successes, len(observed)) if observed else None
    sufficient = bool(
        len(observed) >= gate.minimum_groups
        and not unknown
        and not safety_unknown
        and not unsafe
        and interval is not None
        and interval[0] >= gate.minimum_wilson_lower
    )
    durations = [t.duration_sec for t in observed if t.duration_sec is not None]
    readiness = [t.successor_ready for t in observed if t.successor_ready is not None]
    result = {
        "schema": "soccer.physical_skill_capability.exact_context.v1",
        "skill_id": first.skill_id,
        "context_hash": first.context_hash,
        "body_hash": first.body_hash,
        "policy_hash": first.policy_hash,
        "trial_groups": len(trials),
        "observed": len(observed),
        "successes": successes,
        "unknown_outcomes": unknown,
        "unknown_safety": safety_unknown,
        "unsafe": unsafe,
        "empirical_fraction": successes / len(observed) if observed else None,
        "nominal_wilson_95": list(interval) if interval else None,
        "duration_mean_sec": sum(durations) / len(durations) if durations else None,
        "duration_missing": len(observed) - len(durations),
        "successor_ready_true": sum(readiness),
        "successor_ready_observed": len(readiness),
        "gate": asdict(gate),
        "research_gate_passed": sufficient,
        "sample_independence_verified": False,
        "evidence_authenticated_by_this_function": False,
        "unseen_context": "UNKNOWN",
        "physics_generalization_proven": False,
        "hardware_authorized": False,
        "promotion_authorized": False,
    }
    result["manifest_hash"] = str(hash_json(result))
    return result


def choose_tactical_option(
    scores: dict[str, float],
    evidence: dict[str, dict[str, Any]],
    identities: dict[str, tuple[str, str, str, str]],
) -> str | None:
    """Filter named tactical proposals using exact capability reports; no action is executed.

    Reports are ordinary data and MUST be authenticated/identity-bound by the
    caller. A returned name is an experimental ranking, never permission to
    operate even a simulated or physical actuator.
    """
    if (
        type(scores) is not dict
        or type(evidence) is not dict
        or type(identities) is not dict
        or not scores
        or set(identities) != set(scores)
    ):
        raise ValueError("explicit tactical options and capability reports required")
    if any(
        not isinstance(k, str)
        or not _ID.fullmatch(k)
        or isinstance(v, bool)
        or not isinstance(v, (float, int))
        or not math.isfinite(v)
        for k, v in scores.items()
    ):
        raise ValueError("finite tactical option scores required")
    eligible = []
    for name in scores:
        report = evidence.get(name)
        if report is None:
            continue
        if (
            type(report) is not dict
            or report.get("schema") != "soccer.physical_skill_capability.exact_context.v1"
            or type(report.get("research_gate_passed")) is not bool
            or report.get("hardware_authorized") is not False
            or report.get("promotion_authorized") is not False
        ):
            raise ValueError("valid non-authorizing capability report required")
        copy = dict(report)
        claimed = copy.pop("manifest_hash", None)
        if claimed != str(hash_json(copy)):
            raise ValueError("capability report integrity mismatch")
        if (
            tuple(report.get(k) for k in ("skill_id", "context_hash", "body_hash", "policy_hash"))
            != identities[name]
        ):
            raise ValueError("capability report does not match requested skill/body/context/policy")
        if report["research_gate_passed"]:
            eligible.append(name)
    return max(eligible, key=lambda k: (scores[k], k)) if eligible else None
