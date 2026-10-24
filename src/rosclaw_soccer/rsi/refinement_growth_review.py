"""Paired physical learning effects; neither offline fit nor a promotion gate.

The caller must authenticate whole independent native replay receipts first.
Keep controlled receive, strict precision, safety and old retention separate.
"""

from typing import Any

from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label


def paired_refinement_growth(
    parent: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    required_old_contexts: set[int],
    achieved_old_contexts: set[int],
) -> dict[str, Any]:
    """Compare the unchanged public80 tasks, clustered by40 physical states."""

    def admit(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
        if len(rows) != 80:
            raise ValueError("ALL80 actual independently replayed tasks required")
        indexed = {}
        for row in rows:
            course = row["course"]
            state, task = course["state_id"], course["task"]
            if (
                type(state) is not int
                or not 0 <= state < 40
                or type(task) is not int
                or task not in (0, 1)
                or course["assisted"] is not False
                or course["context"] != (1, 4, 20, 21)[state // 10]
                or (state, task) in indexed
            ):
                raise ValueError("exact unassisted40-state two-goal factorial required")
            outcome = row["outcome"]
            controlled_pass_label(outcome)
            if type(outcome["combined_precision_and_functional_receive"]) is not bool:
                raise ValueError("unchanged typed strict precision result required")
            indexed[state, task] = row
        return indexed

    old, new = admit(parent), admit(candidate)
    if any(old[k]["course"] != new[k]["course"] for k in old):
        raise ValueError("paired initial states, targets and action protocol must match")
    if not required_old_contexts or any(
        type(i) is not int or not 0 <= i < 22 for i in required_old_contexts | achieved_old_contexts
    ):
        raise ValueError("explicit original22-context skill protection required")
    before = {k: controlled_pass_label(r["outcome"]) for k, r in old.items()}
    after = {k: controlled_pass_label(r["outcome"]) for k, r in new.items()}
    rescues = sorted(k for k in old if after[k] and not before[k])
    regressions = sorted(k for k in old if before[k] and not after[k])
    new_nonfoot = sorted(
        k
        for k in old
        if old[k]["outcome"]["passer_clean_foot"] and not new[k]["outcome"]["passer_clean_foot"]
    )
    new_unhealthy = sorted(
        k
        for k in old
        if old[k]["outcome"]["whole_body_healthy"] and not new[k]["outcome"]["whole_body_healthy"]
    )

    def metrics(rows: dict[tuple[int, int], dict[str, Any]]) -> dict[str, int]:
        outcomes = [r["outcome"] for r in rows.values()]
        controlled = [controlled_pass_label(o) for o in outcomes]
        return dict(
            controlled=sum(controlled),
            raw_strict=sum(o["combined_precision_and_functional_receive"] for o in outcomes),
            strict_with_actual_controlled_contact=sum(
                c and o["combined_precision_and_functional_receive"]
                for c, o in zip(controlled, outcomes, strict=True)
            ),
            no_A_foot=sum(o["incoming"]["A_first_foot_strike_substep"] is None for o in outcomes),
            no_B_contact=sum(
                o["incoming"]["B_first_ball_contact_substep"] is None for o in outcomes
            ),
            nonfoot=sum(not o["passer_clean_foot"] for o in outcomes),
            unhealthy=sum(not o["whole_body_healthy"] for o in outcomes),
            incomplete_recovery=sum(not o["passer_full100NNrecovery"] for o in outcomes),
        )

    clusters = [
        dict(
            state_id=s,
            context=(1, 4, 20, 21)[s // 10],
            parent=sum(before[s, t] for t in (0, 1)),
            candidate=sum(after[s, t] for t in (0, 1)),
            net_change=sum(int(after[s, t]) - int(before[s, t]) for t in (0, 1)),
        )
        for s in range(40)
    ]
    lost = sorted(required_old_contexts - achieved_old_contexts)
    parent_metrics, candidate_metrics = metrics(old), metrics(new)
    eligible = (
        len(rescues) > len(regressions)
        and not regressions
        and not lost
        and not new_nonfoot
        and not new_unhealthy
        and candidate_metrics["nonfoot"] == 0
        and candidate_metrics["unhealthy"] == 0
        and candidate_metrics["incomplete_recovery"] == 0
    )
    return dict(
        parent=parent_metrics,
        candidate=candidate_metrics,
        rescues=[list(k) for k in rescues],
        regressions=[list(k) for k in regressions],
        new_nonfoot=[list(k) for k in new_nonfoot],
        new_unhealthy=[list(k) for k in new_unhealthy],
        paired_net_change=len(rescues) - len(regressions),
        lost_old_contexts=lost,
        state_clusters=clusters,
        distinct_physical_states=40,
        goal_tasks=80,
        source_context_clusters=4,
        all_current_successes_retained=not regressions,
        eligible_for_new_sealed_Fresh_research=eligible,
        Fresh_verified=False,
        team_game_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
