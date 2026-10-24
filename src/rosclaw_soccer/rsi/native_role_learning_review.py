"""All-seed physical task readback, never a promotion or a best-seed selector."""

from __future__ import annotations

import math
from typing import Any

from rosclaw_soccer.training.native_team_courses import native_team_course_bank


def review_native_role_learning(reports: list[dict[str, Any]]) -> dict[str, Any]:
    courses = [c.course_id for c in native_team_course_bank()]
    if len(reports) != 3 or sorted(r["seed"] for r in reports) != [926061, 926062, 926063]:
        raise ValueError("all three independently fitted declared seeds required")
    rows = []
    for report in sorted(reports, key=lambda r: r["seed"]):
        if (
            report["status"] != "NATIVE_ROLE_TD3_ONE_GENERATION_PHYSICALLY_MEASURED_NOT_PROMOTED"
            or report["all5_cases_measured"] is not True
            or report["goal_complete"] is not False
            or report["promotion_authorized"] is not False
            or report["hardware_authorized"] is not False
            or [r["course"] for r in report["results"]] != courses
        ):
            raise ValueError("complete fixed courses and SIM-only evidence required")
        for case in report["results"]:
            deltas = {}
            for name in (
                "role_return",
                "actual_foot_contact_steps",
                "close_recent_comotion_controls",
            ):
                parent = case["parent_metrics"][name]
                candidate = case["candidate_metrics"][name]
                if (
                    len(parent) != 8
                    or len(candidate) != 8
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in parent + candidate
                    )
                    or (
                        name != "role_return"
                        and any(v < 0 or int(v) != v for v in parent + candidate)
                    )
                ):
                    raise ValueError("all eight finite role measurements required")
                deltas[name] = [b - a for a, b in zip(parent, candidate, strict=True)]
            rows.append(
                dict(
                    seed=report["seed"],
                    course=case["course"],
                    candidate_healthy=case["candidate_healthy"],
                    parent=case["parent_metrics"],
                    candidate=case["candidate_metrics"],
                    deltas=deltas,
                )
            )
    healthy = all(r["candidate_healthy"] is True for r in rows)
    contact_lost = [
        (r["seed"], r["course"], i)
        for r in rows
        for i, (a, b) in enumerate(
            zip(
                r["parent"]["actual_foot_contact_steps"],
                r["candidate"]["actual_foot_contact_steps"],
                strict=True,
            )
        )
        if a > 0 and b == 0
    ]
    # This is a negative diagnosis, not an acceptance threshold. A positive
    # proxy delta would still need independent possession/task qualification.
    close_increased = {
        str(seed): sum(
            sum(r["deltas"]["close_recent_comotion_controls"]) for r in rows if r["seed"] == seed
        )
        for seed in (926061, 926062, 926063)
    }
    return dict(
        schema="soccer.native_role_learning_review.v1",
        all_seeds_and_cases_retained=True,
        rows=rows,
        all_candidates_healthy=healthy,
        lost_parent_contact_cases=contact_lost,
        close_comotion_delta_by_seed=close_increased,
        every_seed_has_positive_comotion_proxy=all(v > 0 for v in close_increased.values()),
        physical_possession_qualified=False,
        promotion_authorized=False,
        goal_complete=False,
        hardware_authorized=False,
        next_agent_action="ANALYZE_ALL_CASE_COUNTEREXAMPLES_AND_DECLARE_NEXT_HYPOTHESIS",
        observer_or_review_is_new_training=False,
    )
