"""Causal same-state motor supervision; no simulator or training authority."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.neural_motor_student import recorded_student_sequence


def same_state_teacher_sequence(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Labels are teacher advice; lagged observations use EXECUTED mixed actions.

    This validates data layout, not physical truth: a separate full neural/native
    review must authorize the raw recording before a trainer calls this adapter.
    """
    consumed = raw["consumed_action_isaac"]
    features, _ = recorded_student_sequence(
        dict(
            canonical_qpos=raw["canonical_qpos"],
            canonical_qvel=raw["canonical_qvel"],
            neural_action_isaac=consumed,
        )
    )
    for key, expected in (
        ("student_features", features),
        ("mixed_consumed_action_isaac", consumed),
    ):
        actual = raw[key]
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.tobytes() != expected.tobytes()
        ):
            raise ValueError("same-state feature/history or consumed-action binding differs")
    advice = raw["teacher_advice_action_isaac"]
    if advice.shape != (300, 29) or advice.dtype != np.float32 or not np.isfinite(advice).all():
        raise ValueError("full finite same-state teacher advice required")
    return features, advice.copy()


def same_state_actor_mask(
    rows: list[dict[str, Any]], originals: list[dict[str, Any]]
) -> np.ndarray:
    """Keep every course, but never imitate failed/unsafe/noncausal episodes."""
    if len(rows) != 40 or len(originals) != 40:
        raise ValueError("all40 whole-course records required")
    eligible = []
    for index, (row, original) in enumerate(zip(rows, originals, strict=True)):
        if (
            row["course"] != original["course"]
            or row["course"]["course_index"] != index
            or row["split"] != original["split"]
        ):
            raise ValueError("same identities and original whole-course split required")
        if (
            row["split"] not in ("train", "validation")
            or type(row["score"]["qualified"]) is not bool
        ):
            raise ValueError("typed split/qualification required")
        diagnostics = row["prefix_diagnostics"]
        if any(
            type(diagnostics[k]) is not bool
            for k in ("prefix_physical_safe", "post_boundary_control_can_cause_first_touch")
        ):
            raise ValueError("typed safety/causal attribution required")
        eligible.append(
            bool(
                row["score"]["qualified"]
                and diagnostics["prefix_physical_safe"]
                and diagnostics["post_boundary_control_can_cause_first_touch"]
            )
        )
    return np.asarray(eligible, dtype=np.bool_)
