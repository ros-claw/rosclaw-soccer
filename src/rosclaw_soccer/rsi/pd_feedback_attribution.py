"""Exact matched-clock PD arithmetic attribution, not a causal intervention.

Use only authenticated paired episodes with the same body, course and gains.
Projection ratios do not measure energy or total torque cancellation.
"""

from typing import Any

import numpy as np


def attribute_pd_feedback(
    original: dict[str, Any], changed: dict[str, Any], *, kp: np.ndarray, kd: np.ndarray
) -> dict[str, Any]:
    for gain in (kp, kd):
        if gain.shape != (29,) or gain.dtype != np.float64 or not np.isfinite(gain).all():
            raise ValueError("finite unchanged 29-joint gains required")
    if np.any(kp <= 0) or np.any(kd < 0):
        raise ValueError("positive stiffness and nonnegative damping required")
    for trace in (original, changed):
        for name, shape in (
            ("joint_target_rad", (300, 1, 29)),
            ("joint_position_rad", (300, 1, 29)),
            ("joint_velocity_rad_s", (300, 1, 29)),
            ("pd_requested_torque_nm", (300, 1, 10, 29)),
            ("torque_nm", (300, 1, 10, 29)),
        ):
            a = np.asarray(trace.get(name))
            if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError("complete finite paired PD measurements required")
        reconstructed = (
            trace["joint_target_rad"][:, 0] - trace["joint_position_rad"][:, 0]
        ) * kp - trace["joint_velocity_rad_s"][:, 0] * kd
        if not np.allclose(
            reconstructed, trace["pd_requested_torque_nm"][:, 0, 0], rtol=0, atol=1e-8
        ):
            raise ValueError("paired first-substep PD equation does not reconstruct")
    offset = np.asarray(changed.get("contact_exit_mpc_requested_offset_nm"))
    if offset.shape != (300, 1, 29) or offset.dtype != np.float64 or not np.isfinite(offset).all():
        raise ValueError("whole requested MPC correction capture required")
    offset = offset[:, 0]
    active = np.any(offset != 0, axis=1)
    terms = {
        "target_reference": kp
        * (changed["joint_target_rad"][:, 0] - original["joint_target_rad"][:, 0]),
        "position_feedback": -kp
        * (changed["joint_position_rad"][:, 0] - original["joint_position_rad"][:, 0]),
        "velocity_feedback": -kd
        * (changed["joint_velocity_rad_s"][:, 0] - original["joint_velocity_rad_s"][:, 0]),
    }
    delta_pd = (
        changed["pd_requested_torque_nm"][:, 0, 0] - original["pd_requested_torque_nm"][:, 0, 0]
    )
    error = float(np.max(abs(sum(terms.values()) - delta_pd)))
    if error > 1e-8:
        raise ValueError("exact paired PD difference identity failed")
    denominator = float(np.sum(offset[active] ** 2))
    if not np.isfinite(denominator) or (np.any(active) and denominator <= 0):
        raise ValueError("finite nondegenerate requested correction norm required")
    projections = None
    if np.any(active):

        def project(value):
            result = float(np.sum(value[active] * offset[active]) / denominator)
            if not np.isfinite(result):
                raise ValueError("finite PD projection required")
            return result

        projections = {name: project(value) for name, value in terms.items()}
        projections["pd_total"] = project(delta_pd)
        projections["direct_requested_correction"] = 1.0
        projections["net_preclip"] = 1.0 + projections["pd_total"]
        projections["actual_clipped_torque_difference"] = project(
            changed["torque_nm"][:, 0, 0] - original["torque_nm"][:, 0, 0]
        )
    return dict(
        schema="soccer.rsi.paired_pd_feedback_attribution.v1",
        sampled_control_dt_s=0.02,
        substep_within_each_control_frame=0,
        active_frames=np.flatnonzero(active).tolist(),
        maximum_pd_difference_identity_error_nm=error,
        projection_denominator_nm_squared=denominator,
        projections_on_requested_correction=projections,
        unchanged_gains_required=True,
        matched_clock_descriptive_attribution=True,
        causal_component_intervention_performed=False,
        total_torque_or_energy_cancellation_claimed=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
