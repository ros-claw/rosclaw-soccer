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


def projected_action_memory(
    raw_action: np.ndarray,
    executed_target: np.ndarray,
    action_offset: np.ndarray,
    action_scale: np.ndarray,
    changed_axes: np.ndarray,
) -> np.ndarray:
    """Encode projected targets in changed axes; retain original others bitwise.

    This is a different explicit research policy observation contract, not a
    claim that a network trained on raw actions will perform better with it.
    """
    if (
        not isinstance(raw_action, np.ndarray)
        or raw_action.shape != (29,)
        or raw_action.dtype != np.float32
        or not np.isfinite(raw_action).all()
        or any(
            not isinstance(x, np.ndarray)
            or x.shape != (29,)
            or x.dtype != np.float64
            or not np.isfinite(x).all()
            for x in (executed_target, action_offset, action_scale)
        )
        or (action_scale <= 0).any()
        or not isinstance(changed_axes, np.ndarray)
        or changed_axes.shape != (29,)
        or changed_axes.dtype != np.bool_
        or changed_axes[12:].any()
    ):
        raise ValueError(
            "finite typed action, executed target and explicit leg projection required"
        )
    out = raw_action.copy()
    out[changed_axes] = (
        (executed_target[changed_axes] - action_offset[changed_axes]) / action_scale[changed_axes]
    ).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("finite encoded executed action required")
    return out
