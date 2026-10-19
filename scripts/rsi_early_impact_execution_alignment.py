"""Compare selected predictions with actual rollout captures, no new physics.

This checks controls/forces at2ms and state/neural boundaries at20ms. Actual
rollouts do not capture dense2ms q/v, so no dense-state or independent-neural
qualification is claimed. A documented actual safety cancellation is partial,
not a successful full80ms model-fidelity result.
"""

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS
from scripts.rsi_impact_torque_authority_screen import equal


def verify_actual_torques(trace):
    """Derive physical intervention from raw PD and hard limits, not its label."""
    raw = trace["pd_requested_torque_nm"]
    requested = trace["early_impact_feedback_requested_offset_nm"]
    if raw.shape != (300, 1, 10, 29) or requested.shape != (300, 1, 29):
        raise ValueError("whole actual PD and requested action required")
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    expected = np.clip(raw + requested[:, :, None, :], -limits, limits)
    for name, target in (
        ("torque_nm", expected),
        ("actual_actuator_force_nm", expected),
        ("early_impact_feedback_applied_offset_nm", expected - np.clip(raw, -limits, limits)),
    ):
        actual = trace[name]
        if (
            actual.shape != target.shape
            or actual.dtype != np.float64
            or not np.isfinite(actual).all()
        ):
            raise ValueError("whole finite actual torque records required: " + name)
        equal(actual, target, "actual clipped action attribution " + name)


def compare_execution(trace, forecast, frame, *, safety_cancel_frame=None):
    return _compare_execution(
        trace,
        forecast,
        frame,
        safety_cancel_frame=safety_cancel_frame,
        schema="soccer.rsi.selected_early80ms_execution_alignment.v1",
    )


def compare_postcontact_execution(trace, forecast, frame, *, safety_cancel_frame=None):
    if "postcontact_exit_requested_offset_nm" not in trace or any(
        k.startswith(("early_impact_feedback_", "impact_feedback_", "postcontact_release_"))
        for k in trace
    ):
        raise ValueError("actual distinct postcontact exit trace required for alignment")
    return _compare_execution(
        trace,
        forecast,
        frame,
        safety_cancel_frame=safety_cancel_frame,
        schema="soccer.rsi.selected_postcontact80ms_execution_alignment.v1",
    )


def _compare_execution(trace, forecast, frame, *, safety_cancel_frame, schema):
    if (
        type(frame) is not int
        or not 30 <= frame <= 294
        or frame // 20 != (frame + 3) // 20
        or (
            safety_cancel_frame is not None
            and (
                type(safety_cancel_frame) is not int or not frame < safety_cancel_frame < frame + 3
            )
        )
    ):
        raise ValueError(
            "registered complete80ms window and explicit actual cancellation boundary required"
        )
    periods = 4 if safety_cancel_frame is None else safety_cancel_frame - frame
    steps = periods * 10
    fields = (
        ("requested_pd_torque_nm", "pd_requested_torque_nm", (29,)),
        ("applied_control_nm", "torque_nm", (29,)),
        ("actual_actuator_force_nm", "actual_actuator_force_nm", (29,)),
        ("pelvis_height_solver_epoch_m", "pelvis_z_per_substep_m", ()),
    )
    maximum_errors = {}

    def compare(name, actual, predicted, dtype, shape, *, bit_exact=False):
        if (
            actual.shape != shape
            or predicted.shape != shape
            or actual.dtype != dtype
            or predicted.dtype != dtype
            or not np.isfinite(actual).all()
            or not np.isfinite(predicted).all()
        ):
            raise ValueError("complete typed finite actual/predicted arrays required: " + name)
        if bit_exact and actual.tobytes() != predicted.tobytes():
            raise ValueError("actual selected-branch neural bits differ: " + name)
        equal(actual, predicted, "selected forecast versus actual " + name)
        maximum_errors[name] = float(np.max(abs(actual - predicted)))

    for predicted, actual, tail in fields:
        expected = forecast[predicted]
        current = trace[actual]
        if expected.shape != (40, *tail) or current.shape != (300, 1, 10, *tail):
            raise ValueError("all40 predicted and300 actual substep records required")
        compare(
            predicted,
            current[frame : frame + periods, 0].reshape(steps, *tail),
            expected[:steps],
            np.float64,
            (steps, *tail),
        )
    for key, size in (("canonical_qpos", 43), ("canonical_qvel", 41)):
        dense = forecast[key + "_after_substep"]
        if dense.shape != (40, size) or trace[key].shape != (300, 1, size):
            raise ValueError("whole boundary body-state captures required")
        compare(
            key + "_20ms_boundaries",
            trace[key][frame + 1 : frame + periods + 1, 0],
            dense[np.arange(periods) * 10 + 9],
            np.float64,
            (periods, size),
        )
    predicted_peaks = forecast["completed_contact_peak_force_n"]
    if predicted_peaks.shape != (40, 6) or trace["force_n"].shape != (300, 1, 6):
        raise ValueError("whole causal six-channel contact bins required")
    compare(
        "causal_contact_bins",
        trace["force_n"][frame : frame + periods, 0],
        predicted_peaks[np.arange(periods) * 10 + 9],
        np.float64,
        (periods, 6),
    )
    # The observation/target at the cancellation boundary was computed before
    # cancelling torque. It must still match the selected branch's body/NN.
    target_count = min(3, periods)
    for j in range(1, target_count + 1):
        for name, actual, size in (
            ("navigation_target_rad", "foundation_navigation_joint_target_rad", 29),
            ("taskspace_increment_rad", "taskspace_joint_increment_rad", 29),
            ("motor_delta_rad", "motor_delta_rad", 12),
            ("joint_target_rad", "joint_target_rad", 29),
            ("canonical_qpos", "canonical_qpos", 43),
            ("canonical_qvel", "canonical_qvel", 41),
        ):
            compare(
                f"next{j}_" + name,
                trace[actual][frame + j, 0],
                forecast[f"next_{j}_" + name],
                np.float64,
                (size,),
            )
        for name, width in (
            ("encoder_features", 640),
            ("latent_token", 64),
            ("decoder_input", 994),
            ("raw_action_isaac", 29),
            ("joint_target_mujoco_rad", 29),
        ):
            compare(
                f"neural_next{j}_" + name,
                trace["foundation_neural_" + name][frame + j],
                forecast[f"neural_next_{j}_" + name],
                np.float32,
                (1, width),
                bit_exact=True,
            )
    return dict(
        schema=schema,
        forecast_start_frame=frame,
        actual_safety_cancel_frame=safety_cancel_frame,
        physical_substeps_compared=steps,
        actual_control_state_boundaries_compared=periods,
        actual_neural_target_updates_compared=target_count,
        full80ms_execution_alignment_verified=safety_cancel_frame is None,
        actual_safety_cancellation_requires_separate_review=safety_cancel_frame is not None,
        maximum_absolute_errors=maximum_errors,
        dense2ms_actual_qv_captured=False,
        independent_neural_implementation=False,
        new_physical_steps=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
