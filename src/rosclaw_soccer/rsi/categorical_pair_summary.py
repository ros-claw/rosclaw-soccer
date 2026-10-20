"""Whole96 paired evidence math, with three-comparison multiplicity control."""

import math
from collections import Counter
from typing import Any


def summarize_categorical_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(rows) != 96
        or any(type(r["course"]["course_index"]) is not int for r in rows)
        or {r["course"]["course_index"] for r in rows} != set(range(96))
    ):
        raise ValueError("complete unique96 paired contexts required")
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("anchor", "seed17", "seed29", "seed43")
    }
    outcomes = {}
    strata = Counter(r["course"]["stratum"] for r in rows)
    if len(strata) != 6 or set(strata.values()) != {16}:
        raise ValueError("all six balanced strata required")
    for row in rows:
        if [v["variant"] for v in row["variants"]] != list(counts):
            raise ValueError("ordered complete four-arm comparison required")
        flags = []
        for v in row["variants"]:
            s, p = v["score"], v["prefix_diagnostics"]
            f = (
                s["qualified"],
                p["prefix_physical_safe"],
                p["post_boundary_control_can_cause_first_touch"],
            )
            metrics = (s["minimum_pelvis_z_m"], s["minimum_upright"], s["maximum_nonfoot_force_n"])
            if (
                any(type(b) is not bool for b in (*f, v["causal_qualified"]))
                or any(not math.isfinite(x) for x in metrics)
                or v["causal_qualified"] != all(f)
            ):
                raise ValueError("typed outcomes must agree with finite native primitives")
            c = counts[v["variant"]]
            c["success"] += all(f)
            c["balance"] += metrics[0] >= 0.65 and metrics[1] >= 0.9
            c["nonfoot"] += metrics[2] > 1
            c["prefix_blocked"] += not (f[1] and f[2])
            flags.append(all(f))
        outcomes[row["course"]["course_index"]] = flags
    comparisons = {}
    for offset, name in enumerate(("seed17", "seed29", "seed43"), start=1):
        rescues = sorted(i for i, f in outcomes.items() if f[offset] and not f[0])
        regressions = sorted(i for i, f in outcomes.items() if f[0] and not f[offset])
        n, r = len(rescues) + len(regressions), len(rescues)
        p = sum(math.comb(n, k) for k in range(r, n + 1)) / 2**n
        adjusted = min(1.0, 3 * p)
        guard = (
            counts[name]["balance"] == 96 and counts[name]["nonfoot"] <= counts["anchor"]["nonfoot"]
        )
        comparisons[name] = dict(
            rescues=rescues,
            regressions=regressions,
            net_gain=len(rescues) - len(regressions),
            paired_one_sided_sign_p=p,
            bonferroni_three_comparison_p=adjusted,
            physical_guard_not_worse=guard,
            necessary_generalization_evidence=bool(
                len(rescues) > len(regressions) and adjusted <= 0.05 and guard
            ),
        )
    return dict(
        independent_contexts=96,
        paired_comparisons=3,
        totals={k: dict(v) for k, v in counts.items()},
        strata=dict(strata),
        comparisons=comparisons,
        research_gate_is_not_canonical_promotion=True,
        original_far_or_private_fresh_pass=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
