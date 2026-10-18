"""Sampled root motion, not a naturalness, balance or successor-skill judge."""

from typing import Any

import numpy as np


def summarize_root_motion_window(
    pose_world_wxyz: np.ndarray[Any, Any],
    twist_world_linear_local_angular: np.ndarray[Any, Any],
    *,
    sample_dt_s: float,
    forward_axis_world: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Separate retreat, tilt and heading without calling all rotation jitter.

    Caller owns synchronized observations, regular sampling and root-frame
    conventions. Travel from position differences undercounts motion between
    samples. Angular components are body-local, not world yaw/pitch/roll rates.
    """
    q, v, axis = pose_world_wxyz, twist_world_linear_local_angular, forward_axis_world
    if (
        q.ndim != 2
        or q.shape[1:] != (7,)
        or not 2 <= len(q) <= 3000
        or v.shape != (len(q), 6)
        or axis.shape != (3,)
        or any(
            a.dtype != np.float64 or not np.isfinite(a).all() or np.max(np.abs(a)) > 1e6
            for a in (q, v, axis)
        )
        or type(sample_dt_s) is not float
        or not np.isfinite(sample_dt_s)
        or not 0 < sample_dt_s <= 0.1
        or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, rtol=0, atol=1e-6)
        or not np.isclose(np.linalg.norm(axis), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError(
            "finite synchronized root pose/twist, unit quaternion/axis and clock required"
        )
    displacement = np.diff(q[:, :3], axis=0)
    along = displacement @ axis
    w, x, y, z = q[:, 3:7].T
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
    heading_x, heading_y = 1 - 2 * (y * y + z * z), 2 * (w * z + x * y)
    heading_observed = bool(np.all(np.hypot(heading_x, heading_y) >= 1e-6))
    yaw = np.unwrap(np.arctan2(heading_y, heading_x)) if heading_observed else None
    energy = np.mean(v[:, 3:] ** 2, axis=0)
    angular_energy = float(energy.sum())
    return dict(
        schema="soccer.physics.sampled_root_motion_window.v1",
        samples=len(q),
        sample_dt_s=sample_dt_s,
        position_endpoint_span_s=(len(q) - 1) * sample_dt_s,
        forward_axis_world=axis.tolist(),
        net_root_translation_world_m=(q[-1, :3] - q[0, :3]).tolist(),
        sampled_root_path_length_m=float(np.linalg.norm(displacement, axis=1).sum()),
        sampled_forward_root_travel_m=float(np.maximum(along, 0).sum()),
        sampled_backward_root_travel_m=float(-np.minimum(along, 0).sum()),
        negative_root_velocity_samples=int(np.count_nonzero(v[:, :3] @ axis < 0)),
        root_tilt_maximum_rad=float(tilt.max()),
        root_tilt_start_rad=float(tilt[0]),
        root_tilt_end_rad=float(tilt[-1]),
        heading_projection_observed=heading_observed,
        root_heading_change_rad=None if yaw is None else float(yaw[-1] - yaw[0]),
        root_local_angular_component_rms_rad_s=np.sqrt(energy).tolist(),
        root_local_angular_energy_fractions=None
        if angular_energy == 0
        else (energy / angular_energy).tolist(),
        end_root_linear_velocity_world_m_s=v[-1, :3].tolist(),
        end_root_angular_velocity_local_rad_s=v[-1, 3:].tolist(),
        between_sample_motion_unobserved=True,
        root_retreat_is_not_footstep_classification=True,
        angular_speed_is_not_jitter_classification=True,
        balance_or_next_skill_success_qualified=False,
    )
