"""Read-only paired-course accounting; frames/replays are not extra samples."""

import math
from collections import Counter, defaultdict


def summarize_pairs(rows):
    if not rows or len({r["course"]["course_index"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique paired courses required")
    strata = defaultdict(Counter)
    owners = Counter()
    for row in rows:
        variants = row["variants"]
        if [v["variant"] for v in variants] != ["baseline", "portfolio"]:
            raise ValueError("ordered baseline/portfolio pair required")
        c = strata[row["course"]["stratum"]]
        c["courses"] += 1
        outcomes = []
        for v in variants:
            score, prefix = v["score"], v["prefix_diagnostics"]
            causal = bool(
                score["qualified"]
                and prefix["prefix_physical_safe"]
                and prefix["post_boundary_control_can_cause_first_touch"]
            )
            if type(v["causal_qualified"]) is not bool or v["causal_qualified"] != causal:
                raise ValueError("primary outcome contradicts raw score/prefix")
            outcomes.append(causal)
            key = v["variant"]
            z, u, force = (
                score["minimum_pelvis_z_m"],
                score["minimum_upright"],
                score["maximum_nonfoot_force_n"],
            )
            if not all(math.isfinite(x) for x in (z, u, force)):
                raise ValueError("finite physical guard metrics required")
            c[key + "_success"] += causal
            c[key + "_balance"] += z >= 0.65 and u >= 0.9
            c[key + "_nonfoot"] += force > 1
            c[key + "_prefix_blocked"] += not (
                prefix["prefix_physical_safe"]
                and prefix["post_boundary_control_can_cause_first_touch"]
            )
        b, p = outcomes
        rescue, regression = not b and p, b and not p
        if row["rescue"] != rescue or row["regression"] != regression:
            raise ValueError("paired change metadata contradicts outcomes")
        c["rescues"] += rescue
        c["regressions"] += regression
        c["both_success"] += b and p
        c["both_failure"] += not b and not p
        owners[variants[1]["portfolio_audit"]["selected_owner"]] += 1
    total = Counter()
    for c in strata.values():
        total.update(c)
    n, r = total["rescues"] + total["regressions"], total["rescues"]
    pvalue = sum(math.comb(n, k) for k in range(r, n + 1)) / 2**n
    return dict(
        totals=dict(total),
        strata={k: dict(v) for k, v in sorted(strata.items())},
        selected_owners={str(k): v for k, v in sorted(owners.items())},
        paired_one_sided_sign_p=pvalue,
        net_gain=total["portfolio_success"] - total["baseline_success"],
        original_far_exam_pass=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
