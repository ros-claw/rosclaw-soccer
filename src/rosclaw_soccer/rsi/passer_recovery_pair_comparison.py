"""Compare two audited passer recovery studies on the same actual courses."""

import math

import numpy as np


def compare_passer_recovery_reviews(old: dict, new: dict) -> dict:
    if any(
        not isinstance(r, dict)
        or r.get("status") != "PASSER_VELOCITY_RECOVERY_REPLAY_REVIEWED"
        or r.get("independent_fresh") is not False
        or r.get("promotion_authorized") is not False
        or r.get("hardware_authorized") is not False
        for r in (old, new)
    ) or old.get("prior_hash") != new.get("prior_hash"):
        raise ValueError("two replay-reviewed unpromoted studies with the same prior required")

    def index(review):
        records = review.get("records")
        if not isinstance(records, list) or len(records) != 8:
            raise ValueError("eight actual conditions required")
        rows = {(r["context"], r["side"]): r for r in records}
        if len(rows) != 8 or any(
            r.get("baseline_matches_prior") is not True
            or r.get("pre_switch_prefix_exact") is not True
            or r.get("full_prefix_available") is not True
            for r in records
        ):
            raise ValueError("unique exact baseline and fully observed pre-switch prefix required")
        return rows

    a, b = index(old), index(new)
    if a.keys() != b.keys():
        raise ValueError("matching actual conditions required")
    rows = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if (
            x["baseline_outcome"] != y["baseline_outcome"]
            or x["baseline_precise"] != y["baseline_precise"]
            or x["health"]["raw_controls"] != y["health"]["raw_controls"]
            or x["boundary_control"] != y["boundary_control"]
        ):
            raise ValueError("same actual baseline and switch boundary required")
        for r in (x, y):
            for field in ("boundary_target_jump_rad", "transition_window_max_target_jump_rad"):
                if (
                    type(r[field]) not in (int, float)
                    or not math.isfinite(r[field])
                    or r[field] < 0
                ):
                    raise ValueError("finite measured target-jump values required")
        rows.append(dict(context=key[0], side=key[1], old=x, new=y))
    summary = {}
    for field in ("boundary_target_jump_rad", "transition_window_max_target_jump_rad"):
        changes = [r["new"][field] - r["old"][field] for r in rows]
        summary[field] = dict(
            old_median=float(np.median([r["old"][field] for r in rows])),
            new_median=float(np.median([r["new"][field] for r in rows])),
            median_change=float(np.median(changes)),
            decreased=sum(x < -1e-9 for x in changes),
            increased=sum(x > 1e-9 for x in changes),
        )
    for label, accessor in (
        ("strict", lambda r: r["candidate_outcome"]["strict_both_clean_handoff"]),
        ("precise", lambda r: r["candidate_precise"]),
    ):
        pairs = [(accessor(r["old"]), accessor(r["new"])) for r in rows]
        if any(type(x) is not bool or type(y) is not bool for x, y in pairs):
            raise ValueError("actual boolean task outcomes required")
        summary[label] = dict(
            old=sum(x for x, _ in pairs),
            new=sum(y for _, y in pairs),
            gained=sum(not x and y for x, y in pairs),
            lost=sum(x and not y for x, y in pairs),
        )
    equal = [
        r
        for r in rows
        if r["old"]["health"]["executed_controls"] == r["new"]["health"]["executed_controls"]
    ]
    for r in rows:
        for review in (r["old"], r["new"]):
            health = review["health"]
            if (
                type(health["executed_controls"]) is not int
                or health["executed_controls"] <= 0
                or type(health["executed_healthy_full"]) is not bool
            ):
                raise ValueError("actual exposure and boolean health required")
    summary["health_equal_exposure"] = dict(
        pairs=len(equal),
        old=sum(r["old"]["health"]["executed_healthy_full"] for r in equal),
        new=sum(r["new"]["health"]["executed_healthy_full"] for r in equal),
        unequal_pairs=len(rows) - len(equal),
    )
    return dict(
        conditions=8,
        summary=summary,
        records=rows,
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
