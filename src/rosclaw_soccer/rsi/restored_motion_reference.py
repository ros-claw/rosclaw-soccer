"""Select the actor's actual committed motion or standing reference in SIM."""

import numpy as np


def restored_motion_reference(clip, torso_index: int, control: int):
    """Reproduce motion commit then standing transition without future state."""
    if (
        type(control) is not int
        or not 1 <= control < 600
        or type(torso_index) is not int
        or torso_index < 0
        or type(clip.length) is not int
        or clip.length < 1
        or clip.joint_pos.shape != (clip.length, 29)
        or clip.joint_vel.shape != (clip.length, 29)
        or clip.body_ang_vel_w.ndim != 3
        or clip.body_ang_vel_w.shape[0] != clip.length
        or clip.body_ang_vel_w.shape[2] != 3
        or torso_index >= clip.body_ang_vel_w.shape[1]
        or any(
            not np.isfinite(x).all() for x in (clip.joint_pos, clip.joint_vel, clip.body_ang_vel_w)
        )
    ):
        raise ValueError("finite original motion clip and actual control boundary required")
    frame = min(control - 1, clip.length - 1)
    pos = clip.joint_pos[frame].copy()
    if control < clip.length:
        return pos, clip.joint_vel[frame].copy(), clip.body_ang_vel_w[frame, torso_index].copy()
    return pos, np.zeros(29, np.float32), np.zeros(3, np.float32)
