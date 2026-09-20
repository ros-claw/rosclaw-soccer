"""Read-only 500 Hz receiving action-path diagnostics, not success labels."""

from typing import Any

import numpy as np


def receiving_authority_diagnostics(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    """Validate and summarize explicit microstep requests and executed torques.

    Recorded arrays must be authenticated externally. Tracking error does not
    distinguish foundation error, contact loading and residual authority, so no
    controller blame, learning permission or success is inferred here.
    """
    prefix = "receiving_authority_"
    names = (
        "foundation_target_rad",
        "requested_residual_rad",
        "added_residual_rad",
        "pd_target_rad",
        "joint_position_rad",
        "joint_velocity_radps",
        "kp",
        "kd",
        "raw_torque_nm",
        "projected_torque_nm",
        "executed_torque_nm",
    )
    try:
        time = np.asarray(trace[prefix + "time_sec"])
        frames = np.asarray(trace[prefix + "control_frame"])
        values = {name: np.asarray(trace[prefix + name]) for name in names}
    except KeyError as error:
        raise ValueError("complete measured authority path required") from error
    n = len(time) if time.ndim == 1 else 0
    if (
        not 10 <= n <= 30000
        or n % 10 != 0
        or time.dtype.kind not in "fiu"
        or not np.isfinite(time).all()
        or not np.allclose(time, np.arange(n) * 0.002, rtol=0, atol=1e-7)
        or frames.shape != (n,)
        or frames.dtype.kind not in "iu"
        or not np.array_equal(frames, np.arange(n) // 10)
        or any(
            v.shape != (n, 29) or v.dtype.kind not in "fiu" or not np.isfinite(v).all()
            for v in values.values()
        )
    ):
        raise ValueError("contiguous finite 500 Hz authority and 50 Hz frame records required")
    requested = values["requested_residual_rad"]
    added = values["added_residual_rad"]
    if (
        np.any(abs(requested) > 0.100000001)
        or np.any(abs(added) > 0.100000001)
        or np.any(values["kp"] < 0)
        or np.any(values["kd"] < 0)
        or not np.array_equal(values["foundation_target_rad"] + added, values["pd_target_rad"])
        or not np.all((added == requested) | (added == 0))
    ):
        raise ValueError("bounded residual and exact foundation-to-target decomposition required")

    def rms(value: np.ndarray) -> list[float]:
        # Inputs are finite, but arithmetic overflow must not become a valid report.
        result = np.sqrt(np.mean(value**2, axis=0))
        if not np.isfinite(result).all():
            raise ValueError("nonfinite derived authority statistic")
        return [float(v) for v in result]

    tracking = values["pd_target_rad"] - values["joint_position_rad"]
    pd_torque = values["kp"] * tracking - values["kd"] * values["joint_velocity_radps"]
    requested_mask = abs(requested) > 1e-9
    support_names = ("completed_support_time_sec", "completed_ground_force_n")
    support_present = [prefix + name in trace for name in support_names]
    support_mean = None
    if any(support_present):
        if not all(support_present):
            raise ValueError("paired completed support clock and forces required")
        support_time = np.asarray(trace[prefix + support_names[0]])
        support_force = np.asarray(trace[prefix + support_names[1]])
        if (
            support_time.shape != (n,)
            or support_time.dtype.kind not in "fiu"
            or not np.isfinite(support_time).all()
            or not np.allclose(support_time, time + 0.002, rtol=0, atol=1e-7)
            or support_force.shape != (n, 2)
            or support_force.dtype.kind not in "fiu"
            or not np.isfinite(support_force).all()
            or np.any(support_force < 0)
        ):
            raise ValueError("finite left/right completed-interval support required")
        support_mean = np.mean(support_force, axis=0)
        if not np.isfinite(support_mean).all():
            raise ValueError("nonfinite derived support statistic")
    return dict(
        schema="soccer.receiving_authority_diagnostics.v1",
        microsteps=n,
        control_frames=n // 10,
        requested_joint_samples=int(requested_mask.sum()),
        suppressed_requested_joint_samples=int((requested_mask & (added == 0)).sum()),
        requested_residual_rms_rad=rms(requested),
        added_residual_rms_rad=rms(added),
        tracking_error_rms_rad=rms(tracking),
        executed_torque_rms_nm=rms(values["executed_torque_nm"]),
        additive_non_pd_torque_rms_nm=rms(values["raw_torque_nm"] - pd_torque),
        joint_guard_changed_samples=int(
            np.count_nonzero(values["raw_torque_nm"] != values["projected_torque_nm"])
        ),
        torque_clip_changed_samples=int(
            np.count_nonzero(values["projected_torque_nm"] != values["executed_torque_nm"])
        ),
        evidence_authenticated_here=False,
        contact_success_inferred=False,
        promotion_authorized=False,
        completed_ground_force_mean_n=(None if support_mean is None else support_mean.tolist()),
        support_implies_balance_or_readiness=False,
    )
