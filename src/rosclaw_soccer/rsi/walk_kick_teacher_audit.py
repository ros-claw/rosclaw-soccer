"""Reconstruct fixed task labels from actual native arrays, not saved scores."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch


def relabel_arrays(
    arrays: dict[str, np.ndarray], *, controls: int, ball_address: int, root_address: int
) -> dict[str, Any]:
    if (
        type(controls) is not int
        or not 1 <= controls <= 360
        or any(type(a) is not int or not 0 <= a <= 36 for a in (ball_address, root_address))
    ):
        raise ValueError("bounded actual clock and model-bound free joint addresses required")
    n = controls * 10
    for key, shape in (
        ("qpos", (n, 43)),
        ("qvel", (n, 41)),
        ("torque", (n, 29)),
        ("ball_forces", (n, 1, 3)),
        ("joint_targets", (controls, 29)),
        ("commands", (controls, 3)),
        ("ball_position_after_step_m", (controls, 3)),
        ("pelvis_z_per_substep_m", (controls, 10)),
    ):
        value = arrays[key]
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("complete finite native trace required: " + key)
    ball = arrays["qpos"].reshape(controls, 10, 43)[:, -1, ball_address : ball_address + 3]
    height = arrays["qpos"].reshape(controls, 10, 43)[:, :, root_address + 2]
    if not np.array_equal(ball, arrays["ball_position_after_step_m"]) or not np.array_equal(
        height, arrays["pelvis_z_per_substep_m"]
    ):
        raise ValueError("saved measurements disagree with native model-bound positions")
    force = arrays["ball_forces"][:, 0]
    if (force < 0).any():
        raise ValueError("nonnegative actual contact force required")
    if controls < 300:
        feet = np.flatnonzero(force[:, :2].max(axis=1) > 1)
        return dict(
            first_foot_contact_frame=int(feet[0] // 10) if len(feet) else None,
            event_status="PHYSICALLY_TERMINATED",
            measurement_status="INCOMPLETE_PHYSICAL_EPISODE",
            quality_observed=False,
            forward_60_m=None,
            lateral_60_m=None,
            task_pass=False,
            safe=False,
            training_reward=-10.0,
        )
    grouped = np.zeros((300, 1, 6), np.float64)
    grouped[:, 0, :3] = force[:3000].reshape(300, 10, 3).max(axis=1)
    return label_first_touch(
        dict(
            force_n=grouped,
            pelvis_z_per_substep_m=height[:300, None].astype(np.float64),
            ball_position_after_step_m=ball[:300, None].astype(np.float64),
        )
    ).to_dict()
