"""Measure neural request -> hard projection -> recorded actuator transmission.

Numeric diagnostic only. The caller authenticates traces and replay receipts;
this does not execute physics or establish a counterfactual learning gain.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cerebellar_torque_policy import CAPS
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def paired_feedback_divergence(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Same captured clock, separate units; caller authenticates the pairing."""
    for name, width in (("qpos", 43), ("qvel", 41)):
        key = f"cerebellar_canonical_{name}_before_substep"
        for trace in (before, after):
            value = np.asarray(trace[key])
            if (
                value.shape != (300, 1, 10, width)
                or value.dtype != np.float64
                or not np.isfinite(value).all()
                or np.max(np.abs(value)) > 1e6
            ):
                raise ValueError("whole aligned finite canonical capture required")
        if not np.array_equal(np.asarray(before[key])[0, 0, 0], np.asarray(after[key])[0, 0, 0]):
            raise ValueError("identical canonical initial physical state required")
    channels = (
        ("root_position", "root_pose_xyzw_m", slice(0, 3), "m"),
        ("root_velocity", "root_velocity_world", slice(None), "m/s"),
        (
            "foot_body_origins_not_mesh_surface",
            "foot_geometry_position_before_step_m",
            slice(None),
            "m",
        ),
        ("joint_position", "joint_position_rad", slice(None), "rad"),
        ("joint_target", "joint_target_rad", slice(None), "rad"),
        ("parent_motor_delta", "motor_delta_rad", slice(None), "rad"),
        ("navigation", "navigation_command", slice(None), "recorded-command-components"),
        ("actuator_force", "actual_actuator_force_nm", slice(None), "Nm"),
        ("ball_robot_contact_force", "force_n", slice(None), "N"),
        ("ball_before", "ball_position_before_step_m", slice(None), "m"),
        ("ball_after", "ball_position_after_step_m", slice(None), "m"),
        ("parent_recurrent_hidden", "recurrent_residual_hidden_state", slice(None), "raw"),
    )
    rows = {}
    for label, key, components, unit in channels:
        left, right = (np.asarray(trace[key]) for trace in (before, after))
        if (
            left.shape != right.shape
            or left.ndim < 3
            or left.size == 0
            or left.shape[:2] != (300, 1)
            or left.dtype != right.dtype
            or left.dtype.kind != "f"
            or any(
                not np.isfinite(value).all() or np.max(np.abs(value)) > 1e6
                for value in (left, right)
            )
        ):
            raise ValueError("whole finite aligned 300-frame captured channel required")
        difference = (
            np.abs(left[..., components] - right[..., components]).reshape(300, -1).max(axis=1)
        )
        rows[label] = dict(
            unit=unit,
            baseline_maximum_absolute_component=float(np.max(np.abs(left[..., components]))),
            evaluated_maximum_absolute_component=float(np.max(np.abs(right[..., components]))),
            maximum_absolute_component_difference=float(difference.max()),
            first_frame_above={
                str(threshold): None
                if not np.any(difference > threshold)
                else int(np.flatnonzero(difference > threshold)[0])
                for threshold in (1e-8, 1e-5, 0.001, 0.01, 0.1)
            },
        )
    return dict(
        schema="soccer.rsi.cerebellar_paired_feedback_divergence.v1",
        channels=rows,
        identical_initial_canonical_state=True,
        frame_duration_s=0.02,
        first_frame_threshold_is_not_substep_onset=True,
        foot_origins_are_not_contact_surface_positions=True,
        thresholds_are_diagnostics_not_changed_exam_gates=True,
        caller_must_authenticate_intervention_and_receipts=True,
        new_physical_steps=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def diagnose_transmission(trace: dict[str, Any], *, kp: Any, kd: Any) -> dict[str, Any]:
    def array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        raw = np.asarray(value)
        if (
            raw.shape != shape
            or raw.dtype != np.float64
            or not np.isfinite(raw).all()
            or np.max(np.abs(raw)) > 1e6
        ):
            raise ValueError("whole finite float64 canonical capture required")
        return raw

    proportional, derivative = (array(value, (29,)) for value in (kp, kd))
    if np.any(proportional < 0) or np.any(derivative < 0):
        raise ValueError("nonnegative source-bound PD gains required")
    q = array(trace.get("cerebellar_canonical_qpos_before_substep"), (300, 1, 10, 43))
    v = array(trace.get("cerebellar_canonical_qvel_before_substep"), (300, 1, 10, 41))
    target = array(trace.get("joint_target_rad"), (300, 1, 29))
    requested, applied, control, force = [
        array(trace.get(name), (300, 1, 10, 29)).reshape(3000, 29)
        for name in (
            "requested_cerebellar_torque_offset_nm",
            "applied_cerebellar_torque_offset_nm",
            "torque_nm",
            "actual_actuator_force_nm",
        )
    ]
    if np.any(np.abs(requested) > CAPS + 1e-10):
        raise ValueError("requested residual exceeds unchanged neural caps")
    q, v = q.reshape(3000, 43), v.reshape(3000, 41)
    parent = (np.repeat(target[:, 0], 10, axis=0) - q[:, 7:36]) * proportional
    parent -= v[:, 6:35] * derivative
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    expected = np.clip(parent + requested, -limits, limits)
    expected_applied = expected - np.clip(parent, -limits, limits)
    errors = dict(
        control_max_abs_error_nm=float(np.max(np.abs(control - expected))),
        applied_offset_max_abs_error_nm=float(np.max(np.abs(applied - expected_applied))),
        actuator_force_max_abs_error_nm=float(np.max(np.abs(force - control))),
    )
    if any(error > 1e-8 for error in errors.values()):
        raise ValueError("recorded PD projection/applied offset/actuator transmission differs")

    def magnitude(values: np.ndarray) -> dict[str, float]:
        return dict(
            maximum_abs_joint_nm=float(np.max(np.abs(values))),
            rms_joint_nm=float(np.sqrt(np.mean(values**2))),
            mean_vector_norm_nm=float(np.linalg.norm(np.mean(values, axis=0))),
            control_frame_mean_rms_joint_nm=float(
                np.sqrt(np.mean(values.reshape(300, 10, 29).mean(axis=1) ** 2))
            ),
        )

    requested_norm = float(np.linalg.norm(requested))
    return dict(
        schema="soccer.rsi.cerebellar_torque_transmission_diagnostics.v1",
        substep_samples=3000,
        joint_samples=87000,
        parent_hard_clipped_joint_samples=int(np.count_nonzero(np.abs(parent) > limits)),
        neural_projection_changed_joint_samples=int(
            np.count_nonzero(np.abs(requested - applied) > 1e-10)
        ),
        applied_to_requested_l2_ratio=None
        if requested_norm == 0
        else float(np.linalg.norm(applied) / requested_norm),
        requested=magnitude(requested),
        applied=magnitude(applied),
        errors=errors,
        pd_parent_reconstructed_at_same_recorded_states=True,
        zero_residual_closed_loop_counterfactual_replayed=False,
        force_transmission_is_not_football_improvement=True,
        frame_mean_is_not_a_physical_impulse_estimate=True,
        raw_trace_authenticated_here=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
