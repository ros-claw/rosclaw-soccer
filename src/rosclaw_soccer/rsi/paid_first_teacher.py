"""Bounded TRAIN-only first-contact pilot: knee clearance and lateral accuracy."""

from typing import Any

FIRST_PAIRS = ((4, 5), (14, 15))


def first_teacher_config(seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("three predeclared teacher initialization seeds required")
    return dict(
        protocol="paid.first_contact.TRAIN20.two_pair_pilot.v1",
        seed_index=seed,
        random_seed=2026102700 + seed,
        pairs=[list(p) for p in FIRST_PAIRS],
        generations=2,
        population=12,
        elite_count=4,
        selected_repeats=2,
        max_episodes=108,
        max_native_steps=388800,
        pose_scale=[0.04, 0.035, 0.04, 0.15, 0.2, 0.15],
        joint_bound_rad=0.2,
        neural_optimizer_updates=0,
        Fresh_rows_read=0,
        purpose="first-control R2 composite teacher, not learned deployment policy",
    )


def first_teacher_rank(rows: list[dict[str, Any]], protected: list[bool]) -> tuple[Any, ...]:
    from scripts.rsi_search_paid_foot_contact import contact_rank

    if len(rows) != 2 or len(protected) != 2 or any(type(v) is not bool for v in protected):
        raise ValueError("two actual paired TRAIN outcomes and explicit protection required")
    kicks = [r["kick"] for r in rows]
    good = [bool(contact_rank([r, r])[1]) for r in kicks]
    return (
        int(all(r["stand"] and r["stand"]["healthy"] for r in rows)),
        int(all(r["stand"] and r["stand"]["stability_observed"] for r in rows)),
        int(all(not old or new for old, new in zip(protected, good, strict=True))),
        *contact_rank(kicks),
    )
