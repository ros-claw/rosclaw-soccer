"""Current-state entry intent reviewed separately from exogenous response scores."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.ball_entry_teacher import ball_entry_decision
from rosclaw_soccer.rsi.locomotion_response_protocol import LocomotionResponseCourse
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.velocity_policy_native_evidence import review_velocity_recording


def review_entry(
    raw: dict[str, np.ndarray],
    params: Any,
    sensor: Any,
    affordance: Any,
    course: tuple[float, float, float],
    arm: str,
) -> dict[str, Any]:
    if arm not in ("idle", "measured_entry"):
        raise ValueError("exact predeclared entry arm required")
    keys = {"entry_target_root_xy_m", "entry_error_m", "entry_ready", "entry_timed_out"}
    if not keys <= set(raw):
        raise ValueError("full current-state entry decisions required")
    decisions = [
        ball_entry_decision(q, v, frame=f, affordance=affordance)
        for f, (q, v) in enumerate(zip(raw["canonical_qpos"], raw["canonical_qvel"], strict=True))
    ]
    expected = dict(
        entry_target_root_xy_m=np.array([d.target_root_xy_m for d in decisions]),
        entry_error_m=np.array([d.horizontal_error_m for d in decisions]),
        entry_ready=np.array([d.ready for d in decisions], dtype=np.bool_),
        entry_timed_out=np.array([d.timed_out for d in decisions], dtype=np.bool_),
    )
    if any(
        raw[k].dtype != v.dtype or raw[k].shape != v.shape or not np.array_equal(raw[k], v)
        for k, v in expected.items()
    ):
        raise ValueError("current measured entry geometry/gates differ from independent review")
    commands = tuple(d.command if arm == "measured_entry" else (0.0, 0.0, 0.0) for d in decisions)
    # This proxy is ONLY for the existing codec/continuity/PD/physical checks.
    # Its artificial pulse statistics are discarded, not called entry tracking.
    proxy = SimpleNamespace(
        command=lambda f: commands[f], onset_frame=50, pulse_frames=100, command_xy_mps=(0.0, 0.0)
    )
    physical = review_velocity_recording(
        {k: v for k, v in raw.items() if k not in keys},
        params,
        sensor,
        cast(LocomotionResponseCourse, proxy),
    )
    initial = NativeFirstTouchInitialState(course, np.asarray(params.defaults, dtype=np.float64))
    initial.verify(list(course), raw["canonical_qpos"][0], raw["canonical_qvel"][0])
    ready = np.flatnonzero(expected["entry_ready"])
    return dict(
        arm=arm,
        first_ready_frame=int(ready[0]) if len(ready) else None,
        entry_ready_before_original_deadline=bool(len(ready)),
        qualified_entry_readiness=bool(len(ready) and physical["physical_response_guard"]),
        minimum_entry_error_frames30_to90_m=float(expected["entry_error_m"][30:91].min()),
        entry_error_at_deadline_m=float(expected["entry_error_m"][90]),
        terminal_entry_error_m=float(expected["entry_error_m"][-1]),
        minimum_pelvis_height_m=physical["minimum_pelvis_height_m"],
        minimum_upright=physical["minimum_upright"],
        maximum_robot_ball_force_n=physical["maximum_robot_ball_force_n"],
        physical_response_guard=physical["physical_response_guard"],
        all_entry_and_policy_features_and_native_pd_reconstructed=True,
        entry_radius_m=0.04,
        earliest_ready_frame=30,
        latest_ready_frame=90,
        motor_handoff_performed=False,
        football_skill_proven=False,
        new_optimizer_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
