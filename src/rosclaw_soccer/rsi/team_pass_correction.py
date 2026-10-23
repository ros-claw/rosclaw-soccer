"""Explicit next offline research phase with complete teacher coverage.

Versioning extends the declared actor budget128->192 without changing the
legacy default128 acceptance ceiling. This is NOT online RL or promotion.
"""

import random
from typing import Any

from rosclaw_soccer.rsi.team_pass_consolidation import RETAINED, pass_pools

CORRECTION_BANK_STATUS = "TEAM_PASS_ACTUAL_STUDENT_CORRECTION_BANK274_READY_NOT_TRAINED"
CORRECTION_STATUS = "TEAM_PASS_OWN_STATE_CORRECTED_PENDING_UNASSISTED_NATIVE_EXAM"
CORRECTION_KIND = "team-pass-corrected-v2"


def correction_config(seed: int, device: str) -> dict[str, Any]:
    if (
        type(seed) is not int
        or seed not in (85023, 85024, 85025)
        or device not in ("cpu", "cuda:0")
    ):
        raise ValueError("three explicit correction sample-order seeds and device required")
    return dict(
        protocol="dual256.team_pass_actual_student_correction.v2",
        seed=seed,
        device=device,
        steps=64,
        actor_lr=1e-5,
        gradient_norm_limit=1.0,
        old_anchor_weight=4.0,
        new_anchor_weight=1.0,
        old_anchors_per_step=4,
        teachers_per_step=2,
        new_anchor_per_step=1,
        schedule="seeded_permutation_round_robin_complete_coverage",
        whole_history_from_zero=True,
        critic_frozen=True,
        initial_actor_steps=128,
        final_actor_steps=192,
        critic_steps=48,
        inherited_PPO_actor_steps=16,
        PPO_steps_this_stage=0,
        learning_phase="offline_supervised_actual_student_state_NOT_PPO",
        final_only=True,
        runtime_teacher=False,
    )


def correction_pools(records: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    if len(records) != 274:
        raise ValueError("complete274 original/current/correction histories required")
    old, teachers, new = pass_pools(records[:177])
    latest, corrections = records[177:194], records[194:]
    for i, r in enumerate(records[177:], 177):
        if (
            type(r["index"]) is not int
            or r["index"] != i
            or type(r["task_success"]) is not bool
            or type(r["bc_weight"]) not in (int, float)
            or r["bc_weight"] not in (0, 1)
            or bool(r["bc_weight"]) != r["task_success"]
        ):
            raise ValueError("complete ordered positive/negative current trajectories required")
    if sorted(r["source_context"] for r in latest) != list(RETAINED) or any(
        r["role"] != "current_acquired17" or not r["task_success"] for r in latest
    ):
        raise ValueError("all17 actual latest acquired-skill anchors required")
    if any(
        r["role"] != "own_state_expert" or r["source_course"]["assisted"] is not True
        for r in corrections
    ) or {(r["source_course"]["state_id"], r["source_course"]["task"]) for r in corrections} != {
        (i, j) for i in range(40) for j in (0, 1)
    }:
        raise ValueError("all80 actual distinct student-prefix expert goal tasks required")
    old.extend(range(177, 194))
    teachers.extend(r["index"] for r in corrections if r["task_success"])
    if not 41 <= len(teachers) <= 120:
        raise ValueError("nonempty actually qualified correction pool required")
    return old, teachers, new


def correction_schedule(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    correction_config(seed, "cpu")
    pools = correction_pools(records)
    rng = random.Random(seed)
    for pool in pools:
        rng.shuffle(pool)
    old, teachers, new = pools
    schedule: list[dict[str, Any]] = [
        dict(
            step=i + 1,
            teacher_indices=[teachers[(2 * i + j) % len(teachers)] for j in range(2)],
            anchor_indices=[old[(4 * i + j) % len(old)] for j in range(4)],
            new_anchor_index=new[i % len(new)],
        )
        for i in range(64)
    ]
    if (
        {j for r in schedule for j in r["teacher_indices"]} != set(teachers)
        or {j for r in schedule for j in r["anchor_indices"]} != set(old)
        or {r["new_anchor_index"] for r in schedule} != set(new)
    ):
        raise ValueError("bounded schedule must cover ALL admitted positive histories")
    return schedule
