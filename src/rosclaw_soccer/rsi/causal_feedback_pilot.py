"""Finite current-state feedback teacher study; no fitting or sealed exam access."""

from typing import Any

from rosclaw_soccer.rsi.causal_foot_ball_feedback import feedback_parameters
from rosclaw_soccer.rsi.joint_delivery_teacher import PILOTS, joint_delivery_rank


def pilot_config(state: int) -> dict[str, Any]:
    if type(state) is not int or state not in PILOTS:
        raise ValueError("two diagnosed public TRAIN states only")
    seed, task = PILOTS[state]
    return dict(
        protocol="soccer.causal_foot_ball_feedback_TRAIN10.v1",
        state_id=state,
        source_expert_seed=seed,
        task=task,
        variants=[feedback_parameters(i) for i in range(4)],
        search_variants=[1, 2, 3],
        nominal_repeats=2,
        unsearched_TRAIN_offsets_forward_m=[-0.005, 0.005],
        repeats_per_TRAIN_offset=2,
        maximum_worlds=10,
        maximum_native_steps=40000,
        maximum_actor_calls=8000,
        formal_controls=300,
        A_neural_recovery_controls=100,
        original_total_authority_rad=0.2,
        new_optimizer_updates=0,
        Fresh_cases_read=0,
    )


def selection(baseline: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 6:
        raise ValueError("all six nominal paired outcomes required for selection")
    candidates: list[dict[str, Any]] = [dict(variant=0, row=baseline)]
    for variant in (1, 2, 3):
        a, b = rows[2 * (variant - 1) : 2 * variant]
        if (
            a["feedback_variant"] != variant
            or b["feedback_variant"] != variant
            or a["outcome"] != b["outcome"]
        ):
            raise ValueError("same causal variant must repeat the actual physical outcome")
        candidates.append(dict(variant=variant, row=a))
    best = max(range(4), key=lambda k: (joint_delivery_rank(candidates[k]["row"]), -k))
    chosen = candidates[best]
    return dict(
        variant=chosen["variant"],
        outcome=chosen["row"]["outcome"],
        rank=list(joint_delivery_rank(chosen["row"])),
        all_nominal_ranks=[list(joint_delivery_rank(c["row"])) for c in candidates],
    )


def verify_study(
    report: dict[str, Any], baseline: dict[str, Any], variants: list[dict[str, Any]]
) -> None:
    rows = report["records"]
    cfg = pilot_config(report["config"]["state_id"])
    if report["config"] != cfg or len(rows) != 10 or len(variants) != 2:
        raise ValueError("all10 predeclared feedback worlds required")
    selected = selection(baseline, rows[:6])
    for i, row in enumerate(rows):
        expected_course = baseline["course"] if i < 6 else variants[(i - 6) // 2]
        expected_variant = 1 + i // 2 if i < 6 else selected["variant"]
        if (
            row["index"] != i
            or row["course"] != expected_course
            or row["feedback_variant"] != expected_variant
            or row["neural_actor_optimizer_updates"] != 0
            or (
                i < 6
                and (
                    type(row["prefix_identity_fields"]) is not int
                    or row["prefix_identity_fields"] <= 0
                )
            )
            or (i >= 6 and row["prefix_identity_fields"] is not None)
        ):
            raise ValueError("ordered original-course own-state feedback evidence required")
    for i in (6, 8):
        if rows[i]["outcome"] != rows[i + 1]["outcome"]:
            raise ValueError("unsearched TRAIN offset physical repetition differs")
    if report["selected"] != selected:
        raise ValueError("whole-candidate physical selection differs")


def teacher_qualified(report: dict[str, Any]) -> bool:
    """Research screen only, never a runtime promotion or full-team result."""
    return bool(
        report["selected"]["variant"] != 0
        and report["selected"]["rank"][:3] == [1, 1, 1]
        and all(list(joint_delivery_rank(r))[:3] == [1, 1, 1] for r in report["records"][6:])
    )
