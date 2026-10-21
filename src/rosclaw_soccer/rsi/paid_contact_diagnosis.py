"""Explain frozen coarse-contact failures without changing their score.

Diagnostic tags can overlap. In particular, quality0 is not evidence of no
contact: a foot may touch correctly but send a wide ball or hit the shin.
"""

import math
from collections import Counter
from typing import Any


def contact_failure_causes(row: dict[str, Any]) -> list[str]:
    post = row["postcontact"]
    if any(
        type(x) is not bool
        for x in (
            row["body_failed"],
            row["forbidden"],
            post["correct_foot_first"],
            post["post_contact_quality_observed"],
        )
    ):
        raise ValueError("explicit audited contact flags required")
    other = post["other_foot_contact_in_window"]
    if (
        post["post_contact_quality_observed"]
        and type(other) is not bool
        or not post["post_contact_quality_observed"]
        and other is not None
    ):
        raise ValueError("unobserved recontact must remain unknown, not false")
    first = post["first_foot_contact_substep"]
    if first is not None and (type(first) is not int or first < 0):
        raise ValueError("actual first-contact index required")
    causes = []
    if row["body_failed"]:
        causes.append("body_failure")
    if row["forbidden"]:
        causes.append("forbidden_nonfoot_contact")
    if first is None:
        if post["post_contact_quality_observed"] or post["correct_foot_first"]:
            raise ValueError("no contact cannot have observed post-contact quality")
        causes.append("no_foot_contact")
    elif not post["correct_foot_first"]:
        causes.append("wrong_foot_first")
    if first is not None and not post["post_contact_quality_observed"]:
        causes.append("post_contact_window_unobserved")
    if post["post_contact_quality_observed"]:
        names = (
            "goal_progress_1p2s_m",
            "goal_lateral_1p2s_m",
            "maximum_pelvis_retreat_after_contact_m",
        )
        if any(type(post[k]) not in (int, float) or not math.isfinite(post[k]) for k in names):
            raise ValueError("finite actually observed contact outcome required")
        if post[names[0]] < 1:
            causes.append("insufficient_forward_progress")
        if abs(post[names[1]]) > 0.75:
            causes.append("excessive_lateral_error")
        if post[names[2]] > 0.05:
            causes.append("excessive_post_contact_retreat")
    if post["other_foot_contact_in_window"]:
        causes.append("other_foot_recontact")
    return causes


def summarize_contact_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("nonempty declared cohort required")
    tags = [contact_failure_causes(row) for row in rows]
    counts = dict(Counter(tag for group in tags for tag in group))
    contacts = sum(row["postcontact"]["first_foot_contact_substep"] is not None for row in rows)
    if contacts == 0:
        direction = "INVESTIGATE_INTERCEPTION_AND_ACTION_AUTHORITY"
    elif any(
        counts.get(k, 0)
        for k in ("forbidden_nonfoot_contact", "excessive_lateral_error", "other_foot_recontact")
    ):
        direction = "INVESTIGATE_CONTACT_CLEANLINESS_TIMING_AND_BALL_DIRECTION"
    elif counts.get("post_contact_window_unobserved", 0):
        direction = "INVESTIGATE_MEASUREMENT_WINDOW_WITHOUT_REWRITING_SCORES"
    else:
        direction = "REVIEW_PER_CASE_OUTCOME_NOT_JUST_TOTAL_SCORE"
    return dict(
        cases=len(rows),
        actual_foot_contacts=contacts,
        quality_pass=sum(not x for x in tags),
        overlapping_failure_counts=counts,
        per_case_causes=tags,
        diagnostic_direction=direction,
        score_contract_changed=False,
        training_authorized_by_this_diagnostic=False,
        promotion_authorized=False,
    )
