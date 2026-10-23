"""Bounded v3 ablation: coherent task plans, not extra epochs or bigger nets."""

import random
from typing import Any

from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
from rosclaw_soccer.rsi.team_pass_correction import correction_config

KIND = "team-pass-selected-v3"
STATUS = "TEAM_PASS_SINGLE_MODE_ADAPTED_PENDING_UNASSISTED_NATIVE_EXAM"


def selected_config(seed: int, device: str) -> dict[str, Any]:
    if type(seed) is not int or seed not in (95023, 95024, 95025):
        raise ValueError("three preregistered single-mode sample-order seeds required")
    cfg = correction_config(seed - 10000, device)
    cfg.update(
        seed=seed,
        protocol="dual256.team_pass_single_mode_curriculum.v3",
        fit_selection="one_qualified_plan_per_task_current17_anchors",
        raw_positive_not_automatically_fit=True,
        paired_v2_sample_order_seed=seed - 10000,
    )
    return cfg


def selected_schedule(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected_config(seed, "cpu")
    plan = single_mode_pass_curriculum(records)
    old, teacher, new = [
        list(plan[k])
        for k in (
            "current17_anchor_indices",
            "selected_teacher_indices",
            "selected_autonomous_indices",
        )
    ]
    if not teacher or not new:
        raise ValueError("nonempty actual corrective and autonomous plan pools required")
    rng = random.Random(seed - 10000)
    for pool in (old, teacher, new):
        rng.shuffle(pool)
    result: list[dict[str, Any]] = [
        dict(
            step=i + 1,
            teacher_indices=[teacher[(2 * i + j) % len(teacher)] for j in range(2)],
            anchor_indices=[old[(4 * i + j) % len(old)] for j in range(4)],
            new_anchor_index=new[i % len(new)],
        )
        for i in range(64)
    ]
    covered = {j for row in result for j in row["teacher_indices"] + row["anchor_indices"]} | {
        row["new_anchor_index"] for row in result
    }
    if covered != set(plan["fit_indices"]):
        raise ValueError("all and only coherent selected histories must receive fit coverage")
    return result
