"""Whole public-bank structural checks, not physical or density verification."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, cast

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan, _hash


def validate_public_training_rows(
    plan: PublicFirstTouchTrainingPlan, records: list[dict[str, Any]]
) -> dict[str, Any]:
    if type(plan) is not PublicFirstTouchTrainingPlan or type(records) is not list:
        raise ValueError("exact public declaration and ordinary complete records required")
    draws = plan.draws()
    if len(records) != len(draws):
        raise ValueError("whole public training bank required; no partial admission")
    plan_hash = plan.contract()["plan_hash"]
    quality = safe = 0
    for draw, row in zip(draws, records, strict=True):
        if type(row) is not dict:
            raise ValueError("ordinary complete public row required")
        for key, value in asdict(draw).items():
            actual = row.get(key)
            # Canonical JSON converts the immutable tuple to a list.
            if key == "course":
                actual_course = cast(tuple[Any, ...] | list[Any], actual)
                if (
                    type(actual) not in (tuple, list)
                    or tuple(actual_course) != value
                    or any(type(v) is not float for v in actual_course)
                ):
                    raise ValueError("declared physical course must not change")
            elif type(actual) is not type(value) or actual != value:
                raise ValueError("all public course and draw identities must match in order")
        if (
            row.get("actual_behavior_model_hash") != plan.behavior_model_hash
            or type(row.get("frame_sample_count")) is not int
            or row["frame_sample_count"] != 270
            or row.get("physical_replay_performed_here") is not True
            or row.get("execution_origin") != "NEW"
            or row.get("partition") != "DECLARED_PUBLIC_TRAINING"
            or row.get("training_plan_hash") != plan_hash
            or any(
                not _hash(row.get(key))
                for key in (
                    "view_hash",
                    "physical_report_hash",
                    "physical_trace_hash",
                    "independent_review_hash",
                    "foundation_review_hash",
                    "learning_frames_hash",
                )
            )
            or any(
                row.get(key) is not False
                for key in (
                    "private_fresh_accessed",
                    "promotion_authorized",
                    "hardware_authorized",
                )
            )
        ):
            raise ValueError(
                "current public behavior, physical witnesses and SIM-only scope required"
            )
        outcome = row.get("outcome")
        if type(outcome) is not dict or any(
            type(outcome.get(k)) is not bool for k in ("high_quality", "safety_passed")
        ):
            raise ValueError("explicit measured outcome labels required, including failures")
        if (
            type(row.get("terminal_mc_return")) not in (int, float)
            or not math.isfinite(row["terminal_mc_return"])
            or abs(row["terminal_mc_return"]) > 1e6
        ):
            raise ValueError("finite unchanged terminal score required")
        quality += outcome["high_quality"]
        safe += outcome["safety_passed"]
    return dict(
        episodes=len(draws),
        contexts=len(draws) // 4,
        frame_rows=len(draws) * 270,
        four_x_direction_strata_counts=[len(draws) // 4] * 4,
        high_quality_episodes=quality,
        safe_episodes=safe,
        all_success_and_failure_rows_retained=True,
        physics_validated_here=False,
        behavior_density_reconstructed_here=False,
        training_authorized=False,
        global_ancestry_separation_proven=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
