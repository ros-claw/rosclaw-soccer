"""One bounded, research-only passer residual ablation; no motor or physics."""

import numpy as np


def blended_passer_action(
    action: np.ndarray,
    reference: np.ndarray,
    scale_hg: np.ndarray,
    order: np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    """Interpolate in R2 action units, preserving the existing reference envelope.

    This changes a neural actor's residual, not ball velocity directly. It is
    a teacher/authority probe, NOT a trained or physically qualified skill.
    """
    if (
        not isinstance(action, np.ndarray)
        or action.shape != (29,)
        or action.dtype != np.float32
        or not isinstance(reference, np.ndarray)
        or reference.shape != (29,)
        or reference.dtype != np.float32
        or not isinstance(scale_hg, np.ndarray)
        or scale_hg.shape != (29,)
        or not np.isfinite(scale_hg).all()
        or (scale_hg <= 0).any()
        or not isinstance(order, np.ndarray)
        or order.shape != (29,)
        or order.dtype.kind not in ("i", "u")
        or sorted(order.tolist()) != list(range(29))
        or not np.isfinite(action).all()
        or not np.isfinite(reference).all()
        or type(fraction) is not float
        or fraction not in (0.5, 1.0)
    ):
        raise ValueError("finite exact motor codec and declared bounded fraction required")
    scale_action = scale_hg[np.argsort(order)]
    if np.max(np.abs((action - reference) * scale_action)) > 0.20002:
        raise ValueError("original actual action exceeds reference authority")
    if fraction == 1.0:
        return action.copy()
    result = (reference + np.float32(fraction) * (action - reference)).astype(np.float32)
    if (
        not np.isfinite(result).all()
        or np.max(np.abs((result - reference) * scale_action)) > 0.20002
    ):
        raise ValueError("bounded passer residual authority required")
    return result
