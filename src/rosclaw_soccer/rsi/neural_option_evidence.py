"""TRAIN-only expert composition potential; never an executable case lookup.

Temporal commitment is motivated by call-and-return options, not a claim that
Option-Critic has been trained here: https://arxiv.org/abs/1609.05140 .
Initiation learning: https://arxiv.org/abs/2001.00271 . Physical re-execution and
Fresh remain necessary even if observed initial features separate all labels.
"""

from typing import Any

from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label


def option_safe(outcome: dict[str, Any]) -> bool:
    controlled_pass_label(outcome)  # Reuse typed contact/recovery contract.
    return bool(
        outcome["whole_body_healthy"]
        and outcome["passer_full100NNrecovery"]
        and outcome["passer_clean_foot"]
        and not outcome["B_excluded_contact"]
        and outcome["incoming"]["A_first_foot_strike_substep"] is not None
    )


def neural_option_evidence(
    parent: list[dict[str, Any]], experts: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    if len(parent) != 80 or sorted(experts) != [115023, 115024, 115025]:
        raise ValueError("all80 parent and3 preregistered complete neural experts required")
    keys = {(r["course"]["state_id"], r["course"]["task"]) for r in parent}
    if keys != {(s, t) for s in range(40) for t in (0, 1)} or any(
        r["course"]["assisted"] is not False
        or type(r["course"]["state_id"]) is not int
        or type(r["course"]["task"]) is not int
        for r in parent
    ):
        raise ValueError("unchanged complete unassisted public40x2 courses required")
    for rows in experts.values():
        if len(rows) != 80 or any(
            r["course"] != p["course"] for r, p in zip(rows, parent, strict=True)
        ):
            raise ValueError("expert evidence must match every exact parent physical course")
    plan = []
    for i, p in enumerate(parent):
        base = p["outcome"]
        controlled_pass_label(base)
        seed = None
        if controlled_pass_label(base):
            reason = "protect_current_success"
        else:
            possible = [
                s
                for s, rows in sorted(experts.items())
                if controlled_pass_label(rows[i]["outcome"])
            ]
            if possible:
                seed = possible[0]
                reason = "new_controlled_neural_expert"
            elif not option_safe(base):
                possible = [
                    s for s, rows in sorted(experts.items()) if option_safe(rows[i]["outcome"])
                ]
                if possible:
                    seed = possible[0]
                    reason = "clean_contact_neural_expert_not_goal_success"
                else:
                    reason = "no_qualified_neural_repair"
            else:
                reason = "retain_parent_no_observed_improvement"
        chosen = base if seed is None else experts[seed][i]["outcome"]
        plan.append(
            dict(
                index=i,
                state_id=p["course"]["state_id"],
                task=p["course"]["task"],
                selected_seed=seed,
                option_label=0 if seed is None else [115023, 115024, 115025].index(seed) + 1,
                reason=reason,
                parent_controlled=controlled_pass_label(base),
                selected_observed_controlled=controlled_pass_label(chosen),
                selected_observed_safe=option_safe(chosen),
                selected_observed_clean_foot=chosen["passer_clean_foot"],
            )
        )
    return dict(
        schema="soccer.neural_option.empirical_composition_potential.v1",
        plan=plan,
        parent_controlled=sum(controlled_pass_label(p["outcome"]) for p in parent),
        selected_observed_controlled=sum(p["selected_observed_controlled"] for p in plan),
        selected_observed_unsafe=sum(not p["selected_observed_safe"] for p in plan),
        selected_observed_nonfoot=sum(not p["selected_observed_clean_foot"] for p in plan),
        new_controlled_cases=[
            [p["state_id"], p["task"]]
            for p in plan
            if p["selected_observed_controlled"] and not p["parent_controlled"]
        ],
        selector_inputs_must_exclude_state_id_task_id_outcomes=True,
        empirical_recombination_only=True,
        composition_executed=False,
        learned_selector_proven=False,
        Fresh_verified=False,
        team_game_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
