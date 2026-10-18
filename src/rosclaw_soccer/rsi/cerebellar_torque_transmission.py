"""Measure neural request -> hard projection -> recorded actuator transmission.

Numeric diagnostic only. The caller authenticates traces and replay receipts;
this does not execute physics or establish a counterfactual learning gain.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cerebellar_torque_policy import CAPS
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


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
