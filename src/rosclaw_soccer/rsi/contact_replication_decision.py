"""Bounded experiment-family decision; not a task pass or policy promotion."""

from typing import Any


def ball_outcome(row: dict[str, Any]) -> bool:
    """Diagnostic only: measured 1.2s progress, not the original H300 exam."""
    post = row["postcontact"]
    if not post["post_contact_quality_observed"] or not row["stable"] or row["forbidden"]:
        return False
    progress = post["goal_progress_1p2s_m"]
    lateral = post["goal_lateral_1p2s_m"]
    if progress is None or lateral is None:
        raise ValueError("observed physical outcome must include displacement")
    return bool(progress >= 1 and abs(lateral) <= 0.3 * progress)


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(seed, task) for seed in (7602, 7605, 7606) for task in ("train64", "dev64")}
    identities = [(r["seed"], r["task_set"]) for r in rows]
    if len(identities) != 6 or set(identities) != expected:
        raise ValueError("all three fixed seeds and both task sets exactly once required")
    safety_regression = any(
        r["final"]["stable"] < r["initial"]["stable"]
        or r["final"]["forbidden"] > r["initial"]["forbidden"]
        for r in rows
    )
    # Independent repetitions must each improve held-out ball outcomes, not just
    # contact-event counts. Do not choose the best seed or pool away regressions.
    replication_gain = all(
        r["final"]["ball_outcome"] > r["initial"]["ball_outcome"]
        and r["final"]["clean"] >= r["initial"]["clean"]
        for r in rows
        if r["seed"] in (7605, 7606) and r["task_set"] == "dev64"
    )
    return dict(
        decision=(
            "REJECT_SAFETY_REGRESSION"
            if safety_regression
            else "REQUIRES_ORIGINAL_FORMAL52_AND_SEALED_FRESH"
            if replication_gain
            else "STOP_UNCHANGED_CONTACT24_FAMILY_DIAGNOSE_AUTHORITY_AND_OUTCOME_CREDIT"
        ),
        rows=sorted(rows, key=lambda r: (r["seed"], r["task_set"])),
        original_formal52_passed=False,
        goal_complete=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
