"""Complete paired development outcomes, not independent-course statistics.

Caller authenticates receipts and the predeclared pairing. This module never
opens private holdouts, trains models, executes physics or grants promotion.
"""

from typing import Any


def assess_pairs(
    controls: dict[str, dict[str, Any]], evaluations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not controls or controls.keys() != evaluations.keys():
        raise ValueError("entire matching declared pair bank required, including failures")
    rows: list[dict[str, Any]] = []
    for pair_id, baseline in controls.items():
        evaluated = evaluations[pair_id]
        for outcome in (baseline, evaluated):
            if (
                type(outcome) is not dict
                or type(outcome.get("task_pass")) is not bool
                or type(outcome.get("safe")) is not bool
                or (outcome["task_pass"] and not outcome["safe"])
            ):
                raise ValueError("explicit primitive safety and pass labels required")
        rows.append(
            dict(
                pair_id=pair_id,
                baseline=baseline,
                evaluated=evaluated,
                rescued=not baseline["task_pass"] and evaluated["task_pass"],
                regressed=baseline["task_pass"] and not evaluated["task_pass"],
                newly_unsafe=baseline["safe"] and not evaluated["safe"],
            )
        )
    rescued = sum(r["rescued"] for r in rows)
    regressed = sum(r["regressed"] for r in rows)
    return dict(
        pairs=len(rows),
        baseline_passes=sum(r["baseline"]["task_pass"] for r in rows),
        evaluated_passes=sum(r["evaluated"]["task_pass"] for r in rows),
        rescued_pairs=rescued,
        regressed_pairs=regressed,
        newly_unsafe_pairs=sum(r["newly_unsafe"] for r in rows),
        net_success_gain=rescued - regressed,
        rows=rows,
        primitive_labels_authenticated_here=False,
        independent_course_count_claimed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
