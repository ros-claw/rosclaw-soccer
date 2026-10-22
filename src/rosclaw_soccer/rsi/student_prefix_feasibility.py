"""Diagnostic evidence only: can an expert recover a learner-visited prefix?"""

from typing import Any

import numpy as np

SEEDS = (65023, 65024, 65025)
CONTEXTS = (4, 5, 8, 17, 20, 21)
BOUNDARIES = (40, 80)


def verify_physical_prefix(
    source: dict[str, np.ndarray], actual: dict[str, np.ndarray], controls: int
) -> None:
    if type(controls) is not int or controls not in BOUNDARIES:
        raise ValueError("declared40/80 actual student prefix required")
    for key in ("NN_obs", "NN_action", "qpos", "qvel", "forces", "torque"):
        if (
            len(source[key]) < controls
            or len(actual[key]) < controls
            or source[key].dtype != actual[key].dtype
            or not np.array_equal(source[key][:controls], actual[key][:controls])
        ):
            raise ValueError("source actual physical student prefix differs")


def prefix_feasibility_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(s, c, b) for s in SEEDS for c in CONTEXTS for b in BOUNDARIES}
    if (
        len(rows) != 36
        or {(r["seed"], r["context"], r["prefix_controls"]) for r in rows} != expected
        or any(type(r[k]) is not bool for r in rows for k in ("baseline_quality", "quality"))
    ):
        raise ValueError("all36 fixed paired outcomes, no failure dropping, required")
    lost, recoverable = [], []
    for seed in SEEDS:
        for context in CONTEXTS:
            pair = [r for r in rows if r["seed"] == seed and r["context"] == context]
            if len({r["baseline_quality"] for r in pair}) != 1:
                raise ValueError("same physical baseline must bind both prefix boundaries")
            if context in (5, 8, 17) and not pair[0]["baseline_quality"]:
                lost.append([seed, context])
                if any(r["quality"] for r in pair):
                    recoverable.append([seed, context])
    return dict(
        worlds=36,
        source_contexts=18,
        boundaries_per_source=2,
        teacher_successes=sum(r["quality"] for r in rows),
        rescues=sum(r["quality"] and not r["baseline_quality"] for r in rows),
        regressions=sum(not r["quality"] and r["baseline_quality"] for r in rows),
        actual_student_regression_contexts=lost,
        teacher_recoverable_regression_contexts=recoverable,
        all_observed_regressions_recoverable=bool(lost) and lost == recoverable,
        source_distribution="consumed_public_TRAIN_not_Fresh",
        student_learning_proven=False,
        teacher_generalization_proven=False,
        calibrated_probability=False,
        goal_complete=False,
    )
