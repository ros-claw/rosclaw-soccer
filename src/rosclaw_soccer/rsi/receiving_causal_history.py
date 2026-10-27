"""Causal velocity-motor state history; research features only."""

import numpy as np


def augment_foot_with_velocity_history(foot: np.ndarray, motor_obs: np.ndarray) -> np.ndarray:
    """Current 12D foot state + current/previous two actual 98D motor inputs.

    Missing past frames are zero-padded with two availability bits. No future
    observation, contact event or trajectory outcome enters these features.
    """
    if (
        foot.ndim != 2
        or foot.shape[1] != 12
        or motor_obs.shape != (len(foot), 98)
        or len(foot) < 1
        or not np.isfinite(foot).all()
        or not np.isfinite(motor_obs).all()
    ):
        raise ValueError("aligned finite own foot and velocity observations required")
    foot = np.asarray(foot, np.float32)
    motor_obs = np.asarray(motor_obs, np.float32)
    n = len(foot)
    out = np.zeros((n, 308), dtype=np.float32)
    out[:, :12] = foot
    out[:, 12:110] = motor_obs
    if n > 1:
        out[1:, 110:208] = motor_obs[:-1]
        out[1:, 306] = 1
    if n > 2:
        out[2:, 208:306] = motor_obs[:-2]
        out[2:, 307] = 1
    return out
