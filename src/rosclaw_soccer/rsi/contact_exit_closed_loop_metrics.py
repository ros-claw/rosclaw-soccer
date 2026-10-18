"""Measured closed-loop diagnostics; 50Hz peak bins are not contact durations."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch


def longest_run(mask: np.ndarray) -> int:
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError("one-dimensional boolean bins required")
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def closed_loop_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    outcome = label_first_touch(trace)
    fields = {
        "root_pose_xyzw_m": (300, 1, 7),
        "ball_linear_velocity_before_step_m_s": (300, 1, 3),
        "torque_nm": (300, 1, 10, 29),
    }
    for key, shape in fields.items():
        a = np.asarray(trace.get(key))
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError("complete finite native closed-loop measurements required")
    pose = trace["root_pose_xyzw_m"][:, 0]
    if not np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("unit measured root quaternions required")
    first = outcome.first_foot_contact_frame
    start = 0 if first is None else first
    stop = min(300, start + 61)
    foot = trace["force_n"][start:stop, 0, :2] > 1
    nonfoot = np.flatnonzero(np.any(trace["force_n"][:, 0, 2:] > 1, axis=1))
    vx = trace["ball_linear_velocity_before_step_m_s"][:, 0, 0]
    reverse_after_forward = None
    if first is not None:
        forward_seen = False
        for frame in range(first + 1, 300):
            if vx[frame] > 0:
                forward_seen = True
            if forward_seen and vx[frame] < 0:
                reverse_after_forward = frame
                break
    result = dict(
        schema="soccer.rsi.contact_exit_closed_loop_metrics.v1",
        outcome=outcome.to_dict(),
        sampled_control_dt_s=0.02,
        postcontact_window_start_frame=None if first is None else start,
        postcontact_window_stop_frame_exclusive=None if first is None else stop,
        window_right_censored=first is not None and first + 60 >= 300,
        foot_contact_peak_bins_50hz=None if first is None else foot.sum(axis=0).tolist(),
        longest_foot_contact_peak_bin_run_50hz=None
        if first is None
        else [longest_run(foot[:, i]) for i in (0, 1)],
        peak_bins_are_not_continuous_contact_duration=True,
        first_nonfoot_contact_peak_bin_50hz=int(nonfoot[0]) if len(nonfoot) else None,
        first_reverse_after_positive_ball_velocity_sample_50hz=reverse_after_forward,
        postcontact_root_backward_sampled_travel_m=None
        if first is None
        else float(np.maximum(-np.diff(pose[start:stop, 0]), 0).sum()),
        postcontact_maximum_sampled_root_tilt_rad=None
        if first is None
        else float(
            np.arccos(np.clip(1 - 2 * np.sum(pose[start:stop, 3:5] ** 2, axis=1), -1, 1)).max()
        ),
        postcontact_actual_torque_rms_nm=None
        if first is None
        else float(np.sqrt(np.mean(trace["torque_nm"][start:stop] ** 2))),
        successor_ready=None,
        trained_policy=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if "contact_exit_mpc_requested_offset_nm" in trace:
        requested = trace["contact_exit_mpc_requested_offset_nm"]
        applied = trace["contact_exit_mpc_applied_offset_nm"]
        steps = trace["contact_exit_mpc_forecast_steps"]
        for a, shape in ((requested, (300, 1, 29)), (applied, (300, 1, 10, 29))):
            if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError("finite complete requested and actual teacher offsets required")
        if steps.shape != (300, 1, 1) or steps.dtype != np.int64 or np.any(steps < 0):
            raise ValueError("whole typed nonnegative planning step counts required")
        result["teacher_intervention"] = dict(
            nonzero_request_frames=np.flatnonzero(np.any(requested[:, 0] != 0, axis=1)).tolist(),
            nonzero_actual_offset_frames=np.flatnonzero(
                np.any(applied[:, 0] != 0, axis=(1, 2))
            ).tolist(),
            forecast_frames=np.flatnonzero(steps[:, 0, 0] > 0).tolist(),
            forecast_integrated_substeps=int(steps.sum()),
            maximum_requested_offset_nm=float(np.max(abs(requested))),
            maximum_actual_applied_offset_nm=float(np.max(abs(applied))),
        )
    if "directional_compliance_requested_delta_nm" in trace:
        requested = trace["directional_compliance_requested_delta_nm"]
        applied = trace["directional_compliance_applied_delta_nm"]
        steps = trace["directional_compliance_forecast_steps"]
        for a in (requested, applied):
            if a.shape != (300, 1, 10, 29) or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError("complete finite recorded gain authority required")
        if steps.shape != (300, 1, 1) or steps.dtype != np.int64 or np.any(steps < 0):
            raise ValueError("typed gain forecast accounting required")
        result["directional_gain_intervention"] = dict(
            nonzero_request_frames=np.flatnonzero(
                np.any(requested[:, 0] != 0, axis=(1, 2))
            ).tolist(),
            nonzero_actual_delta_frames=np.flatnonzero(
                np.any(applied[:, 0] != 0, axis=(1, 2))
            ).tolist(),
            forecast_frames=np.flatnonzero(steps[:, 0, 0] > 0).tolist(),
            forecast_integrated_substeps=int(steps.sum()),
            maximum_requested_delta_nm=float(np.max(abs(requested))),
            maximum_actual_applied_delta_nm=float(np.max(abs(applied))),
            this_is_not_the_old_capped_additive_residual=True,
        )
    return result
