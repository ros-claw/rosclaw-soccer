"""Locate reference -> body -> ball divergence in already recorded SIM pairs.

No new physics, success labels, training or source admission. A first differing
sample is not an identified causal effect: clocks differ across channels and
feedback is closed loop. Callers authenticate both full same-course receipts.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

CHANNELS = (
    ("reference_request", "reference_student_requested_reference_increment_rad", (12,), "rad"),
    ("reference_applied", "reference_student_applied_reference_increment_rad", (12,), "rad"),
    ("pre_parent_target", "pre_motor_joint_target_rad", (29,), "rad"),
    ("parent_output", "motor_delta_rad", (12,), "rad"),
    ("executed_target", "joint_target_rad", (29,), "rad"),
    ("pd_torque", "torque_nm", (10, 29), "Nm"),
    ("actuator_force", "actual_actuator_force_nm", (10, 29), "Nm"),
    ("root_before", "root_pose_xyzw_m", (7,), "mixed_xyz_m_quaternion"),
    ("joint_position_before", "joint_position_rad", (29,), "rad"),
    ("foot_origins_before", "foot_geometry_position_before_step_m", (4, 3), "m"),
    ("ball_before", "ball_position_before_step_m", (3,), "m"),
    ("ball_after", "ball_position_after_step_m", (3,), "m"),
    ("ball_robot_force", "force_n", (6,), "N"),
)


def reference_feedback_divergence(zero: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    """Keep all 300 frames, including failures; units never share a score."""
    for key, width in (("canonical_qpos", 43), ("canonical_qvel", 41)):
        pair = [np.asarray(t[key]) for t in (zero, student)]
        if (
            any(
                a.shape != (300, 1, width) or a.dtype != np.float64 or not np.isfinite(a).all()
                for a in pair
            )
            or pair[0][0].tobytes() != pair[1][0].tobytes()
        ):
            raise ValueError("whole same initial canonical state required")
    measured = {}
    for label, key, tail, unit in CHANNELS:
        pair = [np.asarray(t[key]) for t in (zero, student)]
        if any(
            a.shape != (300, 1, *tail)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
            for a in pair
        ):
            raise ValueError("whole finite aligned reference/body/ball channels required")
        if label == "reference_applied" and (np.any(pair[0] != 0) or np.any(pair[1][:30] != 0)):
            raise ValueError("recorded NAV must be zero and student first30 must not intervene")
        if (
            label not in ("reference_request", "reference_applied")
            and pair[0][:30].tobytes() != pair[1][:30].tobytes()
        ):
            raise ValueError("all physical channels must preserve exact first30 control prefix")
        delta = np.abs(pair[0] - pair[1]).reshape(300, -1).max(axis=1)
        measured[label] = dict(
            unit=unit,
            maximum_component_difference=float(delta.max()),
            first_frame_above={
                str(t): int(np.flatnonzero(delta > t)[0]) if np.any(delta > t) else None
                for t in (1e-8, 1e-5, 0.001, 0.01, 0.1)
            },
        )
    result = dict(
        schema="soccer.rsi.reference_feedback_divergence.v1",
        channels=measured,
        initial_canonical_state_bit_exact=True,
        first30_physical_prefix_bit_exact=True,
        all300_frames_retained=True,
        frame_duration_s=0.02,
        sample_timing=(
            "state/foot/ball_before pre-step; target current-control; "
            "torque ten 2ms substeps; ball_after post-step"
        ),
        mixed_root_units_not_position_distance=True,
        foot_origins_not_mesh_surface=True,
        first_difference_is_not_proven_causal_order=True,
        thresholds_are_diagnostics_not_task_gates=True,
        physical_source_authentication_performed_here=False,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
