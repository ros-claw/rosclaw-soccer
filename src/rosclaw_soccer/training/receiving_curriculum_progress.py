"""Per-player curriculum progress; never substitute aggregate wins for coverage."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rosclaw_soccer.training.role_receiving_courses import ROSTER


def receiving_curriculum_progress(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_hash: str,
    speeds_mps: tuple[float, ...],
    minimum_courses: int = 4,
    target_fraction: float = 0.8,
) -> dict[str, Any]:
    """Summarize already audited course outcomes for one frozen policy.

    Caller must verify physical files and scoring before supplying rows. This
    coverage diagnostic neither authenticates evidence nor authorizes promotion.
    Duplicated courses, mixed policy lineage and missing roles cannot graduate.
    """
    import re

    if (
        not isinstance(policy_hash, str)
        or re.fullmatch(r"sha256:[a-f0-9]{64}", policy_hash) is None
        or not speeds_mps
        or any(
            type(s) not in (int, float) or not math.isfinite(s) or not 0.5 <= s <= 3
            for s in speeds_mps
        )
        or len(set(speeds_mps)) != len(speeds_mps)
        or type(minimum_courses) is not int
        or minimum_courses < 2
        or type(target_fraction) not in (int, float)
        or not math.isfinite(target_fraction)
        or not 0.8 <= target_fraction <= 1
    ):
        raise ValueError("explicit bounded receiving curriculum and policy required")
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {
        (agent, speed): [] for agent in ROSTER for speed in speeds_mps
    }
    seen: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("agent_id"), str)
            or type(row.get("speed_mps")) not in (int, float)
        ):
            raise ValueError("typed course identity and numeric speed required")
        key = (row.get("agent_id"), row.get("speed_mps"))
        course = row.get("course_id")
        if (
            not isinstance(course, str)
            or not course
            or course in seen
            or key not in groups
            or row.get("policy_hash") != policy_hash
            or type(row.get("safe")) is not bool
            or type(row.get("controlled_reception")) is not bool
        ):
            raise ValueError(
                "unique audited courses with one policy and explicit outcomes required"
            )
        seen.add(course)
        groups[key].append(row)
    strata = []
    for (agent, speed), courses in groups.items():
        n = len(courses)
        successes = sum(r["safe"] and r["controlled_reception"] for r in courses)
        safe = all(r["safe"] for r in courses)
        fraction = successes / n if n else 0.0
        strata.append(
            dict(
                agent_id=agent,
                speed_mps=speed,
                courses=n,
                safe_controlled_courses=successes,
                success_fraction=fraction,
                sufficient_coverage=n >= minimum_courses,
                all_courses_safe=safe if n else False,
                ready=n >= minimum_courses and safe and fraction >= target_fraction,
            )
        )
    return dict(
        schema="soccer.receiving_curriculum_progress.v1",
        policy_hash=policy_hash,
        strata=strata,
        ready_for_next_curriculum=all(s["ready"] for s in strata),
        undercovered=[s for s in strata if not s["sufficient_coverage"]],
        activation_ceiling="SIM_ONLY",
        promotion_eligible=False,
    )
