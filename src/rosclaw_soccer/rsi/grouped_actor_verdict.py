"""Whole40 research-only comparison with the declared4/40 development anchor."""

import math
from typing import Any


def grouped_actor_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(rows) != 40
        or any(type(r["course_index"]) is not int for r in rows)
        or {r["course_index"] for r in rows} != set(range(40))
    ):
        raise ValueError("whole unique original40 outcomes required")
    passed, balance, nonfoot = set(), 0, 0
    for row in rows:
        s, p = row["score"], row["prefix_diagnostics"]
        flags = (
            s["qualified"],
            p["prefix_physical_safe"],
            p["post_boundary_control_can_cause_first_touch"],
        )
        metrics = (s["minimum_pelvis_z_m"], s["minimum_upright"], s["maximum_nonfoot_force_n"])
        if any(type(f) is not bool for f in flags) or not all(math.isfinite(v) for v in metrics):
            raise ValueError("typed flags and finite native metrics required")
        if all(flags):
            passed.add(row["course_index"])
        balance += metrics[0] >= 0.65 and metrics[1] >= 0.9
        nonfoot += metrics[2] > 1
    anchor = {0, 10, 15, 16}
    return dict(
        success=len(passed),
        success_indices=sorted(passed),
        anchor_success_indices=sorted(anchor),
        rescues=sorted(passed - anchor),
        regressions=sorted(anchor - passed),
        old_bc_success_regressions=sorted({0, 10} - passed),
        body_balance=balance,
        nonfoot_cases=nonfoot,
        validation_success=sum(
            r["course_index"] in passed and r["split"] == "validation" for r in rows
        ),
        necessary_research_growth=bool(
            len(passed) > 4 and {0, 10} <= passed and balance == 40 and nonfoot <= 18
        ),
        research_gate_is_not_canonical_promotion=True,
        original_far_or_private_fresh_pass=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
