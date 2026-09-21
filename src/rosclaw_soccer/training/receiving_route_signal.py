"""Historical 17→25 development milestone, never teacher or learner authority.

The caller authenticates the frozen 128-course bank, execution/replay files and
unchanged scoring. This arithmetic gate cannot authenticate supplied booleans.
An execution contract binds the entire intervention, not merely neural weights.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_research_branch import receiving_research_branch
from rosclaw_soccer.training.role_receiving_courses import ROSTER


def receiving_development_route_signal(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_course_ids: tuple[str, ...],
    reference_execution_hash: str,
    candidate_execution_hash: str,
    evidence_hash: str,
) -> dict[str, Any]:
    """Require the complete balanced seen bank and retention, not a selected subset.

    Repeated executions belong to one source-course row. `intervention_applied`
    explicitly distinguishes unchanged fallback roles from new teacher coverage.
    This signal does not select a checkpoint, open a sealed bank or start training.
    """
    if type(expected_course_ids) is not tuple or len(expected_course_ids) != 128:
        raise ValueError("the complete frozen 128-course development bank is required")
    if len(rows) != 128:
        raise ValueError("one audited paired outcome per source course is required")
    paired: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or any(
            type(row.get(key)) is not bool for key in ("intervention_applied", "strict_replay")
        ):
            raise ValueError("explicit intervention scope and physical replay required")
        if not row["strict_replay"]:
            raise ValueError("independent physical replay required for every course")
        if (
            row.get("reference_execution_hash") != reference_execution_hash
            or row.get("candidate_execution_hash") != candidate_execution_hash
        ):
            raise ValueError("mixed execution lineage cannot establish a route signal")
        if not row["intervention_applied"] and any(
            row.get(f"reference_{name}") != row.get(f"candidate_{name}")
            for name in ("safe", "capture")
        ):
            raise ValueError("unchanged fallback outcomes must match their reference")
        paired.append(
            dict(
                row,
                reference_policy_hash=reference_execution_hash,
                candidate_policy_hash=candidate_execution_hash,
            )
        )
    # Reuse the complete-exam, explicit safety, identity and role validation.
    branch = receiving_research_branch(
        paired,
        expected_course_ids=expected_course_ids,
        reference_policy_hash=reference_execution_hash,
        candidate_policy_hash=candidate_execution_hash,
        evidence_hash=evidence_hash,
    )
    counts = Counter(row["agent_id"] for row in paired)
    if counts != Counter({agent: 16 for agent in ROSTER}):
        raise ValueError("all eight role-agents require their sixteen declared courses")
    baseline = sum(row["reference_capture"] for row in paired)
    if baseline != 17:
        raise ValueError("this milestone is bound to the original 17-success reference")
    successes = sum(row["candidate_safe"] and row["candidate_capture"] for row in paired)
    passed = successes >= 25 and not branch["unsafe_courses"] and not branch["lost_courses"]
    report = dict(
        schema="soccer.receiving_development_route_signal.v1",
        status="ROUTE_VALIDATED" if passed else "ROUTE_NOT_VALIDATED",
        evidence_domain="SEEN_DEVELOPMENT_ONLY",
        course_count=128,
        reference_successes=baseline,
        candidate_safe_successes=successes,
        intervention_courses=sum(row["intervention_applied"] for row in paired),
        unchanged_fallback_courses=sum(not row["intervention_applied"] for row in paired),
        new_safe_successes=sum(
            not row["reference_capture"] and row["candidate_capture"] and row["candidate_safe"]
            for row in paired
        ),
        unsafe_courses=branch["unsafe_courses"],
        lost_courses=branch["lost_courses"],
        reference_execution_hash=reference_execution_hash,
        candidate_execution_hash=candidate_execution_hash,
        evidence_hash=evidence_hash,
        paired_routing_hash=branch["manifest_hash"],
        evidence_authenticated_by_this_function=False,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        blind_exam_evaluated=False,
        activation_ceiling="SIM_ONLY",
    )
    report["manifest_hash"] = hash_json(report)
    return report
