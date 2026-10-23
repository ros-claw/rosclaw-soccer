"""Explicit small offline pass-learning stage; NOT PPO or policy promotion."""

from typing import Any

BANK_STATUS = "TEAM_PASS_WHOLE_HISTORY_BANK177_READY_NOT_TRAINED"
STATUS = "TEAM_PASS_NEURAL_ADAPTED_PENDING_UNASSISTED_NATIVE_EXAM"
KIND = "team-pass-consolidated-v1"
RETAINED = (1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21)


def pass_config(seed: int, device: str) -> dict[str, Any]:
    if (
        type(seed) is not int
        or seed not in (75023, 75024, 75025)
        or device not in ("cpu", "cuda:0")
    ):
        raise ValueError("three explicit pass-adaptation sample-order seeds and device required")
    return dict(
        protocol="dual256.team_pass_offline_consolidation.v1",
        seed=seed,
        device=device,
        steps=16,
        actor_lr=1e-5,
        old_anchor_weight=4.0,
        new_anchor_weight=1.0,
        old_anchors_per_step=4,
        teacher_per_step=1,
        new_anchor_per_step=1,
        gradient_norm_limit=1.0,
        whole_history_from_zero=True,
        critic_frozen=True,
        initial_actor_steps=112,
        final_actor_steps=128,
        critic_steps=48,
        inherited_PPO_actor_steps=16,
        PPO_steps_this_stage=0,
        learning_phase="offline_supervised_NOT_PPO",
        final_only=True,
        runtime_teacher=False,
    )


def pass_pools(records: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    if len(records) != 177:
        raise ValueError("complete177 sequence bank required")
    old: list[int] = []
    teachers: list[int] = []
    new: list[int] = []
    roles = {"retained_neural17": 0, "new_teacher": 0, "new_unassisted": 0}
    old_contexts = []
    for i, r in enumerate(records):
        if (
            type(r["index"]) is not int
            or r["index"] != i
            or r["role"] not in roles
            or type(r["task_success"]) is not bool
            or type(r["bc_weight"]) not in (int, float)
            or r["bc_weight"] not in (0, 1)
            or bool(r["bc_weight"]) != r["task_success"]
        ):
            raise ValueError(
                "exact ordered bank and typed positive/negative learning labels required"
            )
        roles[r["role"]] += 1
        if r["role"] == "retained_neural17":
            if not r["task_success"]:
                raise ValueError("all acquired old-skill anchors must remain positive")
            old_contexts.append(r["source_context"])
            old.append(i)
        elif r["bc_weight"]:
            (teachers if r["role"] == "new_teacher" else new).append(i)
    if (
        roles != {"retained_neural17": 17, "new_teacher": 80, "new_unassisted": 80}
        or sorted(old_contexts) != list(RETAINED)
        or not teachers
        or not new
    ):
        raise ValueError(
            "all17 old anchors, all160 new outcomes and nonempty safe teaching pools required"
        )
    return old, teachers, new
