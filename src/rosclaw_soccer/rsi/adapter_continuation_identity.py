"""Explicit versioned TRAIN continuation identity, not policy promotion.

Old75024 continuations stay tied to bank177. New consolidated heads were fit
on bank274 and must prove its exact bank177 ancestry and frozen65023 memory.
No new model may enter through the legacy kind or a guessed default.
"""

from typing import Any


def continuation_course(course: dict[str, Any], prefix: int, teacher_basis: str) -> dict[str, Any]:
    """Self-based control arm has ZERO stroke; treatment retains the old pulse.

    Both start from the same actual student. No stored h/c or body state is
    restored at80. The old retained-teacher protocol remains byte-for-byte in
    its course intent; this helper never changes that arm's latent.
    """
    if (
        type(prefix) is not int
        or prefix not in (0, 80)
        or teacher_basis not in ("retained65023", "student")
    ):
        raise ValueError("explicit0/80 arm and versioned teacher basis required")
    return course | (
        dict(latent=[[0.0] * 6 for _ in range(2)])
        if teacher_basis == "student" and prefix == 0
        else {}
    )


def continuation_protocol(
    *,
    kind: str,
    seed: int,
    training: dict[str, Any],
    bank: dict[str, Any],
    bank_hash: str,
    child_bank: dict[str, Any] | None = None,
    child_bank_hash: str | None = None,
    teacher_basis: str = "retained65023",
) -> str:
    if teacher_basis not in ("retained65023", "student"):
        raise ValueError("explicit retained or current-student teacher basis required")
    if training["config"]["seed"] != seed:
        raise ValueError("exact predeclared student seed required")
    if kind == "team-pass-consolidated-v1":
        if (
            seed != 75024
            or training["data_hash"] != bank_hash
            or training["warm_start_hash"] != bank["warm_start_hash"]
            or child_bank is not None
            or child_bank_hash is not None
            or teacher_basis != "retained65023"
        ):
            raise ValueError("unchanged original75024 bank177 continuation required")
        return "soccer.student_actual_prefix_expert_suffix_TRAIN80.v1"
    if kind == "consolidated-adapter-v1":
        if (
            seed not in (105024, 105025)
            or child_bank is None
            or child_bank_hash is None
            or training["data_hash"] != child_bank_hash
            or training["memory_training_hash"] != bank["warm_start_hash"]
            or child_bank["status"]
            != "TEAM_PASS_ACTUAL_STUDENT_CORRECTION_BANK274_READY_NOT_TRAINED"
            or child_bank["base_data_hash"] != bank_hash
            or len(child_bank["records"]) != 274
            or child_bank["Fresh_fit_rows"] != 0
        ):
            raise ValueError("retained new-head family with exact274-to177 ancestry required")
        return (
            "soccer.adapter_self_based_stroke_TRAIN80.v1"
            if teacher_basis == "student"
            else "soccer.adapter_actual_prefix_expert_suffix_TRAIN80.v1"
        )
    raise ValueError("explicit separately admitted continuation policy kind required")
