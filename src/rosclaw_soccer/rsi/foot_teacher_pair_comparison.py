"""Compare matched audited teacher reviews, without counting replay as new data."""

import math

import numpy as np

from rosclaw_soccer.rsi.paired_health_exposure import paired_health_from_first_failures


def compare_foot_teacher_reviews(old: dict, new: dict) -> dict:
    if any(
        not isinstance(r, dict)
        or r.get("status") != "RECEIVER_FOOT_TEACHER_REPLAY_REVIEWED"
        or r.get("independent_fresh") is not False
        or r.get("promotion_authorized") is not False
        or r.get("zero_identity_count") != 8
        for r in (old, new)
    ) or old.get("prior_hash") != new.get("prior_hash"):
        raise ValueError("two complete unpromoted reviews with the same physical parent required")

    def indexed(review):
        rows = review.get("records")
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError("eight actual paired conditions required")
        result = {(r["context"], r["side"]): r for r in rows}
        if len(result) != 8 or not all(
            r.get("zero_teacher_matches_lateral_baseline") is True for r in rows
        ):
            raise ValueError("unique paired contexts and exact zero identity required")
        for row in rows:
            for arm in ("baseline", "teacher"):
                values = row[arm]
                if (
                    type(values["precise_healthy_return"]) is not bool
                    or type(values["outcome"]["strict_both_clean_handoff"]) is not bool
                    or type(values["outcome"]["both_bodies_healthy"]) is not bool
                    or type(values["slew_exceeding_controls"]) is not int
                    or values["slew_exceeding_controls"] < 0
                    or type(values["max_total_joint_jump_rad"]) not in (float, int)
                    or not math.isfinite(values["max_total_joint_jump_rad"])
                    or values["max_total_joint_jump_rad"] < 0
                ):
                    raise ValueError("finite physical outcomes and jump metrics required")
        return result

    a, b = indexed(old), indexed(new)
    if a.keys() != b.keys():
        raise ValueError("same actual development conditions required")
    rows = []
    for key in sorted(a):
        if a[key]["baseline"] != b[key]["baseline"]:
            raise ValueError("same zero-teacher physical baseline required")
        rows.append(dict(context=key[0], side=key[1], old=a[key]["teacher"], new=b[key]["teacher"]))

    def foot_first_after_pass(row):
        outcome = row["outcome"]
        passer = outcome["actual_passer_foot_substep"]
        foot = outcome["actual_receiver_foot_substep"]
        nonfoot = outcome["receiver_nonfoot_substep"]
        for index in (passer, foot, nonfoot):
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("actual nonnegative contact substeps required")
        return bool(
            passer is not None
            and foot is not None
            and passer < foot
            and (nonfoot is None or foot < nonfoot)
        )

    metrics = {
        "strict_clean": lambda x: x["outcome"]["strict_both_clean_handoff"],
        "healthy": lambda x: x["outcome"]["both_bodies_healthy"],
        "precise": lambda x: x["precise_healthy_return"],
        "slew_violation": lambda x: x["slew_exceeding_controls"] > 0,
        "receiver_foot_first_after_pass": foot_first_after_pass,
        "receiver_nonfoot_contact": lambda x: x["outcome"]["receiver_nonfoot_substep"] is not None,
    }
    summary = {}
    for name, metric in metrics.items():
        pairs = [(metric(r["old"]), metric(r["new"])) for r in rows]
        summary[name] = dict(
            old=sum(x for x, _ in pairs),
            new=sum(y for _, y in pairs),
            gained=sum(not x and y for x, y in pairs),
            lost=sum(x and not y for x, y in pairs),
        )
    timeline_fields = [
        (r[arm].get("observed_controls"), r[arm].get("first_body_failure_controls"))
        for r in rows
        for arm in ("old", "new")
    ]
    if all(n is None and first is None for n, first in timeline_fields):
        summary["common_exposure_available"] = False
    else:
        exposure = [paired_health_from_first_failures(r["old"], r["new"]) for r in rows]
        for row, item in zip(rows, exposure, strict=True):
            row["common_health"] = item
        pairs = [(x["first_healthy_common"], x["second_healthy_common"]) for x in exposure]
        summary["common_exposure_available"] = True
        summary["unequal_exposure_pairs"] = sum(not x["equal_exposure"] for x in exposure)
        summary["healthy_common"] = dict(
            old=sum(x for x, _ in pairs),
            new=sum(y for _, y in pairs),
            gained=sum(not x and y for x, y in pairs),
            lost=sum(x and not y for x, y in pairs),
        )
    changes = [
        r["new"]["max_total_joint_jump_rad"] - r["old"]["max_total_joint_jump_rad"] for r in rows
    ]
    summary["max_total_joint_jump_change_rad"] = dict(
        median=float(np.median(changes)),
        decreased=sum(x < -1e-9 for x in changes),
        increased=sum(x > 1e-9 for x in changes),
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
