"""Paired receiving-oracle comparison, not learner or promotion authority.

Callers must authenticate CPU MuJoCo evidence before constructing outcomes.
An unsuccessful bounded search does not prove physical impossibility.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

SUBSTRATES = ("A0_leg12", "A1_body29", "A2_sonic", "A3_sonic_residual")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class ReceivingOracleOutcome:
    case_id: str
    substrate: str
    initial_state_hash: str
    physics_hash: str
    evaluation_hash: str
    teacher_hash: str
    evidence_hash: str
    optimizer_seed: int
    rollout_budget: int
    rollouts_used: int
    captured: bool
    safe: bool
    strict_replay: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or re.fullmatch(r"[a-z0-9_.-]{1,80}", self.case_id) is None
            or self.substrate not in SUBSTRATES
            or any(
                not isinstance(h, str) or _HASH.fullmatch(h) is None
                for h in (
                    self.initial_state_hash,
                    self.physics_hash,
                    self.evaluation_hash,
                    self.teacher_hash,
                    self.evidence_hash,
                )
            )
            or type(self.optimizer_seed) is not int
            or not 0 <= self.optimizer_seed < 2**32
            or type(self.rollout_budget) is not int
            or not 1 <= self.rollout_budget <= 10000
            or type(self.rollouts_used) is not int
            or not 1 <= self.rollouts_used <= self.rollout_budget
            or any(type(v) is not bool for v in (self.captured, self.safe, self.strict_replay))
        ):
            raise ValueError("explicit bounded, typed, content-bound oracle outcome required")


def compare_receiving_oracles(
    outcomes: tuple[ReceivingOracleOutcome, ...], *, case_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Require all four substrates on 64 identical, distinct initial states.

    No winner is inferred from missing adapters; no automatic training activation.
    All trials, including failed ones, count against the declared search budget.
    """
    if type(case_ids) is not tuple or len(case_ids) != 64 or len(set(case_ids)) != 64:
        raise ValueError("exactly 64 predeclared representative case identities required")
    if type(outcomes) is not tuple or len(outcomes) != 256:
        raise ValueError("all 64 x 4 outcomes required; missing is not failure or success")
    grouped: dict[tuple[str, str], ReceivingOracleOutcome] = {}
    evidence: set[str] = set()
    for row in outcomes:
        if not isinstance(row, ReceivingOracleOutcome):
            raise ValueError("typed oracle outcome required")
        key = (row.case_id, row.substrate)
        if row.case_id not in case_ids or key in grouped or row.evidence_hash in evidence:
            raise ValueError("unknown, duplicate or evidence-reused trial")
        if not row.strict_replay:
            raise ValueError("CPU physical replay required before comparison")
        grouped[key] = row
        evidence.add(row.evidence_hash)
    states = set()
    for case in case_ids:
        ref = grouped[case, SUBSTRATES[0]]
        states.add(ref.initial_state_hash)
        for substrate in SUBSTRATES[1:]:
            other = grouped[case, substrate]
            if any(
                getattr(ref, key) != getattr(other, key)
                for key in (
                    "initial_state_hash",
                    "physics_hash",
                    "evaluation_hash",
                    "optimizer_seed",
                    "rollout_budget",
                )
            ):
                raise ValueError("paired state, physics, scoring, seed and budget must match")
    if len(states) != 64:
        raise ValueError("duplicated physical initial states cannot inflate coverage")
    if len({r.evaluation_hash for r in outcomes}) != 1:
        raise ValueError("one frozen evaluation contract required")
    result = {
        "schema": "soccer.receiving_reachability_comparison.v1",
        "counts": {
            s: {
                "cases": 64,
                "safe_captures": sum(r.captured and r.safe for r in outcomes if r.substrate == s),
                "unsafe_cases": sum(not r.safe for r in outcomes if r.substrate == s),
                "rollouts_used": sum(r.rollouts_used for r in outcomes if r.substrate == s),
            }
            for s in SUBSTRATES
        },
        "outcome_set_hash": hash_json(
            [asdict(grouped[c, s]) for c in case_ids for s in SUBSTRATES]
        ),
        "interpretation": "bounded_search_evidence_not_proof_of_physical_upper_bound",
        "evidence_authenticated_by_this_function": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["manifest_hash"] = hash_json(result)
    return result
