"""Original-IK whole-contact and motion diagnostics, never fabricated GRU traces."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.original_ik_planner_evidence import require_original_ik_parity
from rosclaw_soccer.rsi.original_ik_planner_scope import assess_ik_scope
from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import fixed_six_contact_rows
from rosclaw_soccer.rsi.reference_feedback_divergence import CHANNELS
from rosclaw_soccer.sim.contracts import hash_json


def ik_contact_readout(
    declaration: dict[str, Any],
    complete: dict[str, Any],
    replay: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rows = fixed_six_contact_rows(declaration, complete, replay, metrics)
    assessment = assess_ik_scope(declaration["scope"], complete["results"])
    if assessment != complete["assessment"]:
        raise ValueError("all original-IK paired stage outcomes must reconstruct")
    unsafe = sum(not r["original_outcome"]["safe"] for r in rows if r["job"]["role"] == "ik_pulse")
    nonfoot = sum(
        r["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        for r in rows
        if r["job"]["role"] == "ik_pulse"
    )
    return dict(
        rows=rows,
        assessment=assessment,
        pulse_unsafe_endpoints=unsafe,
        pulse_endpoints_with_nonfoot_over1n=nonfoot,
        research_decision="UNSAFE_OR_NONFOOT_IK_COMPOSITION_NOT_ADMITTED"
        if unsafe or nonfoot
        else "MEASURED_ORIGINAL_IK_COMPOSITION_REQUIRES_MECHANISM_REVIEW_NOT_PROMOTION",
        original_task_labels_unchanged=True,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
        derived_courses=3,
        source_scene_clusters=3,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def ik_motion_readout(off: dict[str, Any], pulse: dict[str, Any]) -> dict[str, Any]:
    prefix = require_original_ik_parity(off, pulse, frames=100)
    measured = {}
    # Reuse channel definitions, never fabricate missing learned-reference arrays.
    for label, key, tail, unit in (
        ("navigation", "navigation_command", (3,), "m/s_and_yaw_rate"),
        ("original_ik_increment", "taskspace_joint_increment_rad", (29,), "rad"),
        *CHANNELS[2:],
    ):
        a, b = (np.asarray(t[key]) for t in (off, pulse))
        if any(
            v.shape != (300, 1, *tail)
            or v.dtype != np.float64
            or not np.isfinite(v).all()
            or np.max(np.abs(v)) > 1e6
            for v in (a, b)
        ):
            raise ValueError("whole finite original-IK physical diagnostic channels required")
        delta = np.abs(a - b).reshape(300, -1).max(axis=1)
        measured[label] = dict(
            unit=unit,
            maximum_component_difference=float(delta.max()),
            first_frame_above={
                str(t): int(np.flatnonzero(delta > t)[0]) if np.any(delta > t) else None
                for t in (1e-8, 1e-5, 0.001, 0.01, 0.1)
            },
        )
    result = dict(
        schema="soccer.rsi.original_ik_planner_motion_readout.v1",
        before_action_prefix=prefix,
        measured_channels=measured,
        all300_frames_retained=True,
        reference_gru_arrays_fabricated=False,
        foot_origins_not_mesh_surface=True,
        mixed_root_units_not_position_distance=True,
        first_difference_is_not_proven_causal_order=True,
        action_authority_is_not_task_improvement=True,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
