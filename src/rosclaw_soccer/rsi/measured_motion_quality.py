"""Read-only motion diagnostics, never reward, safety or promotion truth.

Metrics expose retreat and rapid motion without labelling a necessary balance
step as failure or pretending numerical smoothness proves human naturalness.
Input windows must contain actual unpadded native samples at a fixed clock.
"""

import numpy as np


def measured_motion_quality(
    root_positions: np.ndarray,
    root_angular_velocities: np.ndarray,
    joint_velocities: np.ndarray,
    control_targets: np.ndarray,
    heading_xy: np.ndarray,
) -> dict[str, float | int | None]:
    n = len(root_positions)
    if (
        root_positions.shape != (n, 3)
        or root_angular_velocities.shape != (n, 3)
        or joint_velocities.shape != (n, 29)
        or control_targets.ndim != 2
        or control_targets.shape[1] != 29
        or not 1 <= len(control_targets) <= 400
        or not 1 <= n <= 4000
        or not 10 * (len(control_targets) - 1) < n <= 10 * len(control_targets)
        or heading_xy.shape != (2,)
        or not np.isclose(np.linalg.norm(heading_xy), 1.0, atol=1e-8, rtol=0)
        or any(
            x.dtype.kind != "f" or not np.isfinite(x).all()
            for x in (
                root_positions,
                root_angular_velocities,
                joint_velocities,
                control_targets,
                heading_xy,
            )
        )
    ):
        raise ValueError(
            "finite actual native2ms and control20ms windows and unit heading required"
        )
    projected = (root_positions[:, :2] - root_positions[0, :2]) @ heading_xy
    angular = np.linalg.norm(root_angular_velocities, axis=1)
    target_jerk = (
        None
        if len(control_targets) < 4
        else float(np.sqrt(np.mean((np.diff(control_targets, n=3, axis=0) / 0.02**3) ** 2)))
    )
    return dict(
        observed_native_samples=n,
        observed_control_samples=len(control_targets),
        window_duration_s=n * 0.002,
        maximum_retreat_from_window_start_along_initial_body_heading_m=float(
            max(0, -projected.min())
        ),
        net_forward_from_window_start_along_initial_body_heading_m=float(projected[-1]),
        root_height_min_m=float(root_positions[:, 2].min()),
        root_height_max_m=float(root_positions[:, 2].max()),
        root_angular_speed_peak_rad_s=float(angular.max()),
        root_angular_speed_rms_rad_s=float(np.sqrt(np.mean(angular**2))),
        joint_acceleration_rms_rad_s2=(
            None
            if n < 2
            else float(np.sqrt(np.mean((np.diff(joint_velocities, axis=0) / 0.002) ** 2)))
        ),
        control_target_jerk_rms_rad_s3=target_jerk,
    )
