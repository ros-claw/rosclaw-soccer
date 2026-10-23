"""TRAIN labels for physically controlled passes, not static-point regrading.

These labels may select supervised teacher sequences only AFTER whole native
audits. They never claim the unchanged static-point composite passed, neural
training occurred, unseen generalization, or policy promotion.
"""

from typing import Any


def controlled_pass_label(outcome: dict[str, Any]) -> bool:
    flags = (
        "whole_body_healthy",
        "passer_full100NNrecovery",
        "passer_clean_foot",
        "B_legal_functional_receive_before_deadline",
        "B_excluded_contact",
    )
    if any(type(outcome[k]) is not bool for k in flags):
        raise ValueError("actual typed physical health/contact/recovery outcomes required")
    incoming = outcome["incoming"]
    a, b = incoming["A_first_foot_strike_substep"], incoming["B_first_ball_contact_substep"]
    if any(v is not None and (type(v) is not int or v < 0) for v in (a, b)):
        raise ValueError("actual native contact indices or explicit missing events required")
    return bool(
        outcome["whole_body_healthy"]
        and outcome["passer_full100NNrecovery"]
        and outcome["passer_clean_foot"]
        and not outcome["B_excluded_contact"]
        and outcome["B_legal_functional_receive_before_deadline"]
        and a is not None
        and b is not None
        and a < b
    )


def paired_pass_learning_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 160:
        raise ValueError("all160 paired executions of40 new TRAIN physical states required")
    rows = {}
    contexts = (1, 4, 20, 21)
    for row in records:
        c = row["course"]
        if type(row["outcome"]["combined_precision_and_functional_receive"]) is not bool:
            raise ValueError("unchanged typed strict static-point result required")
        state, task, arm = c["state_id"], c["task"], c["assisted"]
        if (
            type(state) is not int
            or not 0 <= state < 40
            or type(task) is not int
            or task not in (0, 1)
            or type(arm) is not bool
            or c["context"] != contexts[state // 10]
        ):
            raise ValueError("declared state/task/context and exact paired arm required")
        key = (state, task, arm)
        if key in rows:
            raise ValueError("duplicate trial is not a new independent initial state")
        rows[key] = row
    expected = {(s, t, a) for s in range(40) for t in (0, 1) for a in (False, True)}
    if set(rows) != expected:
        raise ValueError("complete factorial paired TRAIN bank required")
    labels = {k: controlled_pass_label(r["outcome"]) for k, r in rows.items()}
    rescues, regressions = [], []
    clusters = []
    for state in range(40):
        base = sum(labels[state, t, False] for t in (0, 1))
        teacher = sum(labels[state, t, True] for t in (0, 1))
        clusters.append(
            dict(state_id=state, context=contexts[state // 10], baseline=base, teacher=teacher)
        )
        for task in (0, 1):
            if labels[state, task, True] and not labels[state, task, False]:
                rescues.append([state, task])
            if labels[state, task, False] and not labels[state, task, True]:
                regressions.append([state, task])
    return dict(
        schema="soccer.pass.controlled_training_label_summary.v1",
        executions=160,
        distinct_physical_initial_conditions=40,
        source_context_clusters=4,
        goal_tasks=80,
        baseline_controlled=sum(v for k, v in labels.items() if not k[2]),
        teacher_controlled=sum(v for k, v in labels.items() if k[2]),
        teacher_positive_state_tasks=[
            [s, t] for s in range(40) for t in (0, 1) if labels[s, t, True]
        ],
        rescues=rescues,
        regressions=regressions,
        paired_net_change=len(rescues) - len(regressions),
        state_clusters=clusters,
        strict_static_point_composite_passes=sum(
            r["outcome"]["combined_precision_and_functional_receive"] for r in records
        ),
        strict_point_exam_regraded=False,
        posthoc_training_label=True,
        TRAIN_only=True,
        Fresh_verified=False,
        audits_required_before_fit=True,
        neural_optimizer_updates=0,
        team_game_verified=False,
        promotion_authorized=False,
        goal_complete=False,
    )
