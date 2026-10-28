"""Simulation-only temporal resampling for a frozen whole-body reference."""

import numpy as np


def sample_reference(clip, torso_index: int, position: float, rate: float):
    """Sample own reference causally; velocity scales with elapsed physical time."""
    if (
        type(position) not in (int, float)
        or type(rate) not in (int, float)
        or not np.isfinite([position, rate]).all()
        or not 0 <= position < clip.length
        or not 0.5 <= rate <= 1.0
    ):
        raise ValueError("finite bounded reference position and tempo required")
    i = int(position)
    j = min(i + 1, clip.length - 1)
    weight = position - i

    def interp(values):
        return np.asarray((1.0 - weight) * values[i] + weight * values[j], dtype=np.float32)

    return (
        interp(clip.joint_pos),
        interp(clip.joint_vel) * np.float32(rate),
        interp(clip.body_ang_vel_w[:, torso_index]) * np.float32(rate),
    )
