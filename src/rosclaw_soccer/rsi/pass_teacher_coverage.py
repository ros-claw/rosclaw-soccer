"""Functional teacher coverage is not a precision pass or a learned-policy gain."""

from typing import Any

from rosclaw_soccer.rsi.pass_teacher_transfer_plan import EXPANDED_STATES
from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label


def _indexed(
    rows: list[dict[str, Any]], states: tuple[int, ...]
) -> dict[tuple[int, int], dict[str, Any]]:
    expected = {(s, t) for s in states for t in (0, 1)}
    if len(rows) != len(expected):
        raise ValueError("whole balanced TRAIN task matrix required")
    result = {}
    for r in rows:
        if type(r["state_id"]) is not int or type(r["task"]) is not int:
            raise ValueError("typed public state/task identities required")
        key = (r["state_id"], r["task"])
        if key not in expected or key in result:
            raise ValueError("no missing, duplicate or replacement tasks")
        if type(r["outcome"]["combined_precision_and_functional_receive"]) is not bool:
            raise ValueError("unchanged typed precision result required")
        controlled_pass_label(r["outcome"])
        result[key] = r
    return result


def teacher_coverage(
    baseline: list[dict[str, Any]], selected: list[dict[str, Any]], transfers: list[dict[str, Any]]
) -> dict[str, Any]:
    b = _indexed(baseline, tuple(range(20, 30)))
    s = _indexed(selected, tuple(range(20, 30)))
    t = _indexed(transfers, EXPANDED_STATES)
    before = {k: controlled_pass_label(r["outcome"]) for k, r in b.items()}
    after = {k: controlled_pass_label(r["outcome"]) for k, r in s.items()}
    rescues = sorted(k for k in s if after[k] and not before[k])
    regressions = sorted(k for k in s if before[k] and not after[k])
    healthy_clean = all(
        o["whole_body_healthy"]
        and o["passer_full100NNrecovery"]
        and o["passer_clean_foot"]
        and not o["B_excluded_contact"]
        for o in (r["outcome"] for r in s.values())
    )
    for key in t:
        if type(s[key]["transfer_selected"]) is not bool:
            raise ValueError("explicit transfer-versus-local-fallback selection required")
    strict = {k: r["outcome"]["combined_precision_and_functional_receive"] for k, r in s.items()}
    qualified = (
        sum(after.values()) >= 8
        and len(rescues) - len(regressions) >= 2
        and not regressions
        and healthy_clean
    )
    return dict(
        baseline_controlled=sum(before.values()),
        selected_controlled=sum(after.values()),
        rescues=[list(k) for k in rescues],
        regressions=[list(k) for k in regressions],
        net_controlled_change=len(rescues) - len(regressions),
        all_selected_healthy_clean_full100NNrecovery=healthy_clean,
        strict_composite=sum(strict.values()),
        actual_contact_and_strict=sum(strict[k] and after[k] for k in s),
        transferred_proposals_controlled=sum(
            controlled_pass_label(r["outcome"]) for r in t.values()
        ),
        selected_transfer_count=sum(s[k]["transfer_selected"] for k in t),
        local_fallback_count=sum(not s[k]["transfer_selected"] for k in t),
        functional_correction_teacher_qualified=qualified,
        distinct_initial_states=10,
        goal_tasks=20,
        source_contexts=1,
        public_TRAIN_only=True,
        uncertainty_calibrated=False,
        teacher_search_is_not_neural_training=True,
        learned_policy_improvement_verified=False,
        correction_bank_ready=False,
        Fresh_verified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
