"""One bounded, research-only passer residual ablation; no motor or physics."""

import numpy as np


class PasserContactExitGate:
    """Keep a research intervention through impact until a contact-free control.

    Update only with the completed control's actual native force history. The
    result affects the next control, never the already-executed physics step.
    It is not evidence of a successful pass or authority to use a policy.
    """

    def __init__(self) -> None:
        self.active = True
        self.contact_seen = False

    def observe(self, ball_forces: np.ndarray) -> None:
        if (
            not isinstance(ball_forces, np.ndarray)
            or ball_forces.shape != (10, 3)
            or not np.isfinite(ball_forces).all()
            or (ball_forces < 0).any()
        ):
            raise ValueError("ten finite nonnegative native passer contact samples required")
        touching = bool((ball_forces[:, :2] > 10).any())
        if not self.active:
            return
        if touching:
            self.contact_seen = True
        elif self.contact_seen:
            self.active = False


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
