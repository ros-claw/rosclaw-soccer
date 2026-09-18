"""Choose the next receiving research parent, never activate a match policy.

The caller authenticates trajectories and recomputes scoring. A complete public
receiving exam may retain a safe nonregressing research candidate (including a
tie); an unsafe or regressing one falls back to the fixed reference. Missing or
malformed evidence raises instead of being interpreted as a failed skill exam.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.role_receiving_courses import ROSTER


def receiving_research_branch(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_course_ids: tuple[str, ...],
    reference_policy_hash: str,
    candidate_policy_hash: str,
    evidence_hash: str,
) -> dict[str, Any]:
    """Return a bounded SIM_ONLY routing decision from a complete paired exam."""
    hashes = (reference_policy_hash, candidate_policy_hash, evidence_hash)
    if any(
        not isinstance(h, str) or re.fullmatch(r"sha256:[a-f0-9]{64}", h) is None for h in hashes
    ):
        raise ValueError("content-bound policies and authenticated exam reference required")
    if (
        not isinstance(expected_course_ids, tuple)
        or not expected_course_ids
        or any(not isinstance(c, str) or not c for c in expected_course_ids)
        or len(set(expected_course_ids)) != len(expected_course_ids)
    ):
        raise ValueError("unique predeclared course identifiers required")
    seen: set[str] = set()
    agents: set[str] = set()
    unsafe: list[str] = []
    lost: list[str] = []
    gained: list[str] = []
    remediation: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("explicit paired course evidence required")
        course, agent = row.get("course_id"), row.get("agent_id")
        if (
            not isinstance(course, str)
            or course not in expected_course_ids
            or course in seen
            or not isinstance(agent, str)
            or agent not in ROSTER
            or row.get("reference_policy_hash") != reference_policy_hash
            or row.get("candidate_policy_hash") != candidate_policy_hash
            or any(
                type(row.get(k)) is not bool
                for k in (
                    "reference_safe",
                    "candidate_safe",
                    "reference_capture",
                    "candidate_capture",
                )
            )
            or not row["reference_safe"]
        ):
            raise ValueError("complete unique paired outcomes and a safe reference required")
        seen.add(course)
        agents.add(agent)
        if not row["candidate_safe"]:
            unsafe.append(course)
            remediation.add(agent)
        if row["reference_capture"] and not row["candidate_capture"]:
            lost.append(course)
            remediation.add(agent)
        if not row["reference_capture"] and row["candidate_capture"]:
            gained.append(course)
    if seen != set(expected_course_ids) or agents != set(ROSTER):
        raise ValueError("full declared exam and all eight roles required")
    retain = not unsafe and not lost
    report = dict(
        schema="soccer.receiving_research_branch.v1",
        action="RETAIN_RESEARCH_CANDIDATE" if retain else "ROLLBACK_AND_RETRAIN",
        next_parent_hash=candidate_policy_hash if retain else reference_policy_hash,
        reference_policy_hash=reference_policy_hash,
        candidate_policy_hash=candidate_policy_hash,
        evidence_hash=evidence_hash,
        expected_course_ids=list(expected_course_ids),
        unsafe_courses=sorted(unsafe),
        lost_courses=sorted(lost),
        gained_courses=sorted(gained),
        remediation_course_roles=sorted(remediation),
        remediation_roles_are_not_causal_attribution=True,
        skill_improved_on_public_exam=retain and bool(gained),
        evidence_authenticated_by_this_function=False,
        activation_ceiling="SIM_ONLY",
        promoted=False,
        match_qualified=False,
    )
    report["manifest_hash"] = hash_json(report)
    return report
