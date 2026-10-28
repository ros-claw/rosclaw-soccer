"""Bounded SIM-only receiver transition in the recurrent motor's action ABI."""

import numpy as np


def receiver_transition_action(
    previous_action: np.ndarray,
    proposed_action: np.ndarray,
    reference_action: np.ndarray,
    scale_hg: np.ndarray,
    order: np.ndarray,
    *,
    control_index: int,
    duration: int,
) -> np.ndarray:
    """Blend from the last executed action while retaining reference authority.

    Project onto the existing +/-0.2 rad reference envelope: the result is
    not guaranteed continuous when the previous policy lies outside that
    envelope. The exact returned float32 action must be both executed and
    committed as recurrent feedback. This is not a learned controller.
    """
    actions = (previous_action, proposed_action, reference_action)
    if (
        any(
            not isinstance(x, np.ndarray)
            or x.shape != (29,)
            or x.dtype != np.float32
            or not np.isfinite(x).all()
            for x in actions
        )
        or not isinstance(scale_hg, np.ndarray)
        or scale_hg.shape != (29,)
        or not np.isfinite(scale_hg).all()
        or (scale_hg <= 0).any()
        or not isinstance(order, np.ndarray)
        or order.shape != (29,)
        or order.dtype.kind not in "iu"
        or sorted(order.tolist()) != list(range(29))
        or type(control_index) is not int
        or type(duration) is not int
        or not 1 <= duration <= 100
        or not 0 <= control_index < duration
    ):
        raise ValueError("finite action codec and bounded transition control required")
    scale = scale_hg[np.argsort(order)]
    if np.max(np.abs((proposed_action - reference_action) * scale)) > 0.20002:
        raise ValueError("proposed motor action exceeds reference authority")
    fraction = np.float32((control_index + 1) / duration)
    desired = previous_action + fraction * (proposed_action - previous_action)
    limit = np.asarray(0.19999 / scale, dtype=np.float32)
    result = (reference_action + np.clip(desired - reference_action, -limit, limit)).astype(
        np.float32
    )
    if (
        not np.isfinite(result).all()
        or np.max(np.abs((result - reference_action) * scale)) > 0.20002
    ):
        raise ValueError("transition exceeds unchanged reference authority")
    return result
