"""Choose ONE actual successful motor plan per consumed TRAIN pass task.

Raw task labels and all negative evidence remain unchanged. Selection is fit
eligibility, not a new success score. Prefer already successful autonomous
baseline behavior; use actual own-state expert corrections only where needed.
This function authenticates no external evidence and grants no training or
promotion authority. The caller must admit the complete source/own lineage.
"""

from typing import Any

from rosclaw_soccer.rsi.team_pass_correction import correction_pools
from rosclaw_soccer.sim.contracts import hash_json


def single_mode_pass_curriculum(records: list[dict[str, Any]]) -> dict[str, Any]:
    correction_pools(records)
    pools: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
        role: {} for role in ("new_unassisted", "new_teacher", "own_state_expert")
    }
    for row in records:
        if row["role"] not in pools:
            continue
        course = row["source_course"]
        if (
            type(course["state_id"]) is not int
            or not 0 <= course["state_id"] < 40
            or type(course["task"]) is not int
            or course["task"] not in (0, 1)
            or course["assisted"] is not (row["role"] != "new_unassisted")
        ):
            raise ValueError("typed unchanged actual task and explicit source arm required")
        key = (course["state_id"], course["task"])
        if key in pools[row["role"]]:
            raise ValueError("one actual complete trajectory per source task/arm required")
        pools[row["role"]][key] = row
    keys = {(i, j) for i in range(40) for j in (0, 1)}
    if any(set(pool) != keys for pool in pools.values()):
        raise ValueError("ALL80 original task identities required in every source arm")
    selected, uncovered, choices = [], [], []
    for key in sorted(keys):
        rows = [pools[role][key] for role in ("new_unassisted", "own_state_expert", "new_teacher")]
        tasks = [{k: v for k, v in row["source_course"].items() if k != "assisted"} for row in rows]
        if tasks[0] != tasks[1] or tasks[0] != tasks[2]:
            raise ValueError("matched physical states/goal/intent must remain unchanged")
        winner = next((row for row in rows if row["task_success"]), None)
        if winner is None:
            uncovered.append(dict(state_id=key[0], task=key[1]))
        else:
            selected.append(winner["index"])
            choices.append(
                dict(state_id=key[0], task=key[1], index=winner["index"], role=winner["role"])
            )
    anchors = [r["index"] for r in records if r["role"] == "current_acquired17"]
    fitting = set(anchors + selected)
    raw_positive = {r["index"] for r in records if r["task_success"]}
    if len(anchors) != 17 or not fitting <= raw_positive or not selected:
        raise ValueError("all17 latest acquired anchors and nonempty qualified task plans required")
    plan = dict(
        schema="soccer.single_mode_actual_pass_curriculum.v1",
        input_records_hash=hash_json(records),
        fit_indices=sorted(fitting),
        current17_anchor_indices=anchors,
        selected_goal_indices=selected,
        task_choices=choices,
        selected_teacher_indices=[i for i in selected if records[i]["role"] != "new_unassisted"],
        selected_autonomous_indices=[i for i in selected if records[i]["role"] == "new_unassisted"],
        unselected_positive_indices=sorted(raw_positive - fitting),
        all_negative_indices=[r["index"] for r in records if not r["task_success"]],
        uncovered_tasks=uncovered,
        task_success_labels_modified=False,
        baseline_success_replaced=False,
        records_removed=0,
        priority=["new_unassisted", "own_state_expert", "new_teacher"],
        evidence_authenticated=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    plan["selection_hash"] = hash_json(plan)
    return plan
