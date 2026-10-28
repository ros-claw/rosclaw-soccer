"""Explicit SIM-only leg target-limit ablation, never a hardware safety gate."""

import numpy as np


def receiver_target_limit_projection(
    nominal: np.ndarray, limits: np.ndarray, limited: np.ndarray, *, enabled: bool
) -> dict:
    """Project only the 12 leg position targets and record every changed axis.

    This changes the frozen policy's deployment behavior. It does not prove
    stability, balance, torque limits, smoothness, or real-robot safety.
    The caller owns policy recurrent/action history and physics execution.
    """
    if (
        not isinstance(nominal, np.ndarray)
        or nominal.shape != (29,)
        or nominal.dtype != np.float64
        or not np.isfinite(nominal).all()
        or not isinstance(limits, np.ndarray)
        or limits.shape != (29, 2)
        or limits.dtype != np.float64
        or not np.isfinite(limits).all()
        or not isinstance(limited, np.ndarray)
        or limited.shape != (29,)
        or limited.dtype != np.bool_
        or not limited[:12].all()
        or (limits[:12, 0] >= limits[:12, 1]).any()
        or type(enabled) is not bool
    ):
        raise ValueError("finite HG29 target and all twelve physical leg limits required")
    target = nominal.copy()
    if enabled:
        target[:12] = np.clip(target[:12], limits[:12, 0], limits[:12, 1])
    delta = target - nominal
    return dict(
        target=target, nominal=nominal.copy(), delta=delta, changed_axes=delta != 0, enabled=enabled
    )
