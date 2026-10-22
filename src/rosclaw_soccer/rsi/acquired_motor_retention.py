"""G1 native-exam adapter for Core's task-independent TaskRetention contract.

Original protection and subsequently acquired neural skills are both required.
Teacher success, fit loss, and repeated executions never create new skills here.
"""

import re
from dataclasses import asdict
from typing import Any

from rosclaw.continual.stability import TaskRetention


def native_successes(rows: list[dict[str, Any]]) -> set[int]:
    if (
        len(rows) != 44
        or any(
            type(r["context"]) is not int
            or type(r["repeat"]) is not int
            or type(r["quality"]) is not bool
            for r in rows
        )
        or {(r["context"], r["repeat"]) for r in rows}
        != {(i, j) for i in range(22) for j in range(2)}
    ):
        raise ValueError(
            "all44 explicitly typed complete independently replayed native outcomes required"
        )
    good = set()
    for i in range(22):
        outcomes = [r["quality"] for r in rows if r["context"] == i]
        if outcomes[0] != outcomes[1]:
            raise ValueError("native deterministic repeated qualification differs")
        if outcomes[0]:
            good.add(i)
    return good


def acquired_motor_retention(
    *,
    context_hashes: list[str],
    original_protected: list[int],
    historical: list[tuple[str, list[dict[str, Any]]]],
    current: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        len(context_hashes) != 22
        or len(set(context_hashes)) != 22
        or any(re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None for h in context_hashes)
        or not historical
        or len(historical) > 32
        or len({h for h, _ in historical}) != len(historical)
        or any(re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None for h, _ in historical)
        or not original_protected
        or len(set(original_protected)) != len(original_protected)
        or any(type(i) is not int or not 0 <= i < 20 for i in original_protected)
    ):
        raise ValueError(
            "fixed physical contexts and nonempty independently audited skill history required"
        )
    required = set(original_protected)
    witnesses = []
    for h, rows in historical:
        successes = native_successes(rows)
        required.update(successes)
        witnesses.append(dict(evidence_hash=h, successful_contexts=sorted(successes)))
    achieved = native_successes(current)
    tasks = [
        TaskRetention(
            task_id="soccer.native_first_touch." + context_hashes[i],
            parent_score=1.0,
            candidate_score=float(i in achieved),
            critical=True,
        )
        for i in sorted(required)
    ]
    lost = sorted(required - achieved)
    return dict(
        required_contexts=sorted(required),
        achieved_contexts=sorted(achieved),
        lost_contexts=lost,
        historical_witnesses=witnesses,
        task_retention=[asdict(t) for t in tasks],
        original_protection_retained=set(original_protected) <= achieved,
        acquired_frontier_retained=not lost,
        worst_critical_drop=max(t.drop for t in tasks),
        core_contract="rosclaw.continual.stability.TaskRetention",
        scope="same_physical_native22_public_TRAIN_only_not_legacy52_or_Fresh",
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
