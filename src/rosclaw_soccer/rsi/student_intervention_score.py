"""Actual student30 + teacher270 continuity; the original280 task gate is intact."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry


def fuse_physical_intervention30(
    model: Any, prefix: dict[str, np.ndarray], suffix: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    shapes = dict(
        canonical_qpos=(43,),
        canonical_qvel=(41,),
        joint_target_rad=(29,),
        pd_requested_torque_nm=(10, 29),
        torque_nm=(10, 29),
        actual_actuator_force_nm=(10, 29),
        pelvis_z_per_substep_m=(10,),
        force_n=(6,),
        ball_position_after_step_m=(3,),
        target_body_canonical_qpos_after_substep=(10, 43),
        target_body_canonical_qvel_after_substep=(10, 41),
        ball_contact_force_by_body_n=(10, model.nbody),
    )
    for arrays, frames in ((prefix, 30), (suffix, 270)):
        for k, trailing in shapes.items():
            a = arrays[k]
            if (
                type(a) is not np.ndarray
                or a.shape != (frames, *trailing)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("complete physical student30/teacher270 required: " + k)
        if np.any(arrays["ball_contact_force_by_body_n"] < 0):
            raise ValueError("nonnegative actual native contact magnitudes required")
    joined = {k: np.concatenate((prefix[k], suffix[k])) for k in shapes}
    for before, after in (
        ("canonical_qpos", "target_body_canonical_qpos_after_substep"),
        ("canonical_qvel", "target_body_canonical_qvel_after_substep"),
    ):
        a, b = joined[before][1:], joined[after][:-1, -1]
        if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            raise ValueError("physical continuity broken within/between actual student/teacher")
    return joined


def score_student_intervention30(
    model: Any, prefix: dict[str, np.ndarray], suffix: dict[str, np.ndarray]
) -> dict[str, Any]:
    joined = fuse_physical_intervention30(model, prefix, suffix)
    score = score_full_entry(model, {k: v[20:] for k, v in joined.items()})
    diagnostic = diagnose_entry_prefix(model, prefix, frames=30)
    score["original_complete280_qualified"] = score["qualified"]
    score["actual_preintervention30_safe"] = diagnostic["prefix_physical_safe"]
    score["post_boundary_control_can_cause_first_touch"] = diagnostic[
        "post_boundary_control_can_cause_first_touch"
    ]
    score["physical_prefix_and_suffix_safe"] = bool(
        score["physical_prefix_and_suffix_safe"] and diagnostic["prefix_physical_safe"]
    )
    if not (
        diagnostic["prefix_physical_safe"]
        and diagnostic["post_boundary_control_can_cause_first_touch"]
    ):
        score["qualified"], score["cost"] = False, None
    score["actual_intervention_frame"] = 30
    return score
