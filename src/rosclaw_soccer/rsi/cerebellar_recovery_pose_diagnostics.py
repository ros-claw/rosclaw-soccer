"""Separate measured tilt, heading, translation and angular motion after touch.

Diagnostic only: angular RMS is not a jitter detector or balance certificate.
The frozen training return and finite-horizon task are never changed here.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cerebellar_recovery_return import training_return


def diagnose_recovery_pose(trace: dict[str, Any]) -> dict[str, Any]:
    objective = training_return(trace)
    q = np.asarray(trace.get("cerebellar_canonical_qpos_before_substep"))
    if q.shape != (300, 1, 10, 43) or q.dtype != np.float64 or not np.isfinite(q).all():
        raise ValueError("whole finite canonical before-integration pose capture required")
    q = q.reshape(3000, 43)
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("unit current root quaternion required, no silent normalization")
    result: dict[str, Any] = dict(
        schema="soccer.rsi.cerebellar_recovery_pose_diagnostics.v1",
        measured_training_return_hash=objective["report_hash"],
        formal_outcome=objective["formal_outcome"],
        diagnostics=None,
        raw_trace_authenticated_here=False,
        angular_rms_is_not_jitter_classification=True,
        new_physical_steps=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    outcome = objective["formal_outcome"]
    if not outcome["quality_observed"]:
        return result
    first = outcome["first_foot_contact_frame"]
    section = slice((first + 1) * 10, (first + 61) * 10)
    q = q[section]
    v = np.asarray(trace["cerebellar_canonical_qvel_before_substep"]).reshape(3000, 41)[section]
    w, x, y, z = q[:, 3:7].T
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    heading_x, heading_y = 1 - 2 * (y * y + z * z), 2 * (w * z + x * y)
    heading_observed = bool(np.all(np.hypot(heading_x, heading_y) >= 1e-6))
    yaw = np.unwrap(np.arctan2(heading_y, heading_x)) if heading_observed else None
    energy = np.mean(v[:, 3:6] ** 2, axis=0)
    total = float(np.sum(energy))
    result["diagnostics"] = dict(
        actual_substep_samples=len(q),
        first_frame=first + 1,
        last_frame=first + 60,
        root_tilt_maximum_rad=float(np.max(tilt)),
        root_tilt_start_rad=float(tilt[0]),
        root_tilt_end_rad=float(tilt[-1]),
        root_world_heading_projection_observed=heading_observed,
        root_world_heading_change_rad=None if yaw is None else float(yaw[-1] - yaw[0]),
        root_local_angular_velocity_component_rms_rad_s=np.sqrt(energy).tolist(),
        root_local_angular_energy_fractions=None if total == 0 else (energy / total).tolist(),
        root_world_translation_m=(q[-1, :3] - q[0, :3]).tolist(),
        entire_recovery_motion_explained_by_these_metrics=False,
    )
    return result
