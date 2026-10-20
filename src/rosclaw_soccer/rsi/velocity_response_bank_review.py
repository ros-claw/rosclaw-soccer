"""Read-only complete-bank review; no actor inference, physics or promotion."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.velocity_policy_native_evidence import review_velocity_recording


def review_episode(
    raw: dict[str, np.ndarray], receipt: dict[str, Any], params: Any, sensor: Any, course: Any
) -> dict[str, Any]:
    if (
        receipt.get("course_index") != course.index
        or receipt.get("split") != course.split
        or receipt.get("native_steps") != 3000
        or receipt.get("neural_calls") != 300
        or receipt.get("attempted_neural_calls") != 300
        or receipt.get("optimizer_steps") != 0
        or receipt.get("hardware_authorized") is not False
        or receipt.get("promotion_authorized") is not False
    ):
        raise ValueError("whole actual neural/native SIM episode required")
    report = receipt.get("body_report", {})
    if (
        report.get("frames") != 300
        or report.get("integrated_steps") != 3000
        or report.get("attempted_steps") != 3000
    ):
        raise ValueError("whole actual native body counts required")
    summary = review_velocity_recording(raw, params, sensor, course)
    initial = NativeFirstTouchInitialState(
        course.ball_course, np.asarray(params.defaults, dtype=np.float64)
    )
    initial.verify(list(course.ball_course), raw["canonical_qpos"][0], raw["canonical_qvel"][0])
    if any(receipt.get(k) != v for k, v in summary.items()):
        raise ValueError("episode result differs from independent raw derivation")
    return dict(course_index=course.index, split=course.split, **summary)


def summarize_bank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(rows) != 15
        or {r["course_index"] for r in rows} != set(range(15))
        or any(r["split"] != ("train" if r["course_index"] < 9 else "validation") for r in rows)
    ):
        raise ValueError("all fifteen unique original interventions required")
    return dict(
        interventions=15,
        independent_initial_worlds=1,
        physical_guard_passes=sum(r["physical_response_guard"] is True for r in rows),
        research_response_gate_passes=sum(
            r["command_response_research_gate"] is True for r in rows
        ),
        development_validation_gate_passes=sum(
            r["command_response_research_gate"] is True for r in rows if r["course_index"] >= 9
        ),
        maximum_pulse_mean_velocity_error_mps=max(r["pulse_mean_velocity_error_mps"] for r in rows),
        maximum_terminal_mean_speed_mps=max(r["terminal_mean_speed_mps"] for r in rows),
        maximum_after_pulse_drift_m=max(r["after_pulse_root_drift_m"] for r in rows),
        qualification_not_learning=True,
        new_optimizer_steps=0,
        football_skill_proven=False,
        team_skill_proven=False,
        goal_complete=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
