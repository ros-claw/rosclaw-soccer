"""Bounded measured-state feedback on the live SIM return motor.

This is a privileged experimental teacher, not a learned actor or hardware
control. Physical benefit and retention require same-world paired trials.
"""

import numpy as np

from rosclaw_soccer.rsi.own_state_motor_assistance import blend_actual_motor
from rosclaw_soccer.rsi.paid_balance_recovery import body_feedback, feedback_delta


class RelayBalanceTeacher:
    def __init__(self, latent: np.ndarray, *, scale: np.ndarray, order: np.ndarray) -> None:
        if (
            latent.shape != (9, 4)
            or not np.isfinite(latent).all()
            or (np.abs(latent) > 1).any()
            or scale.shape != (29,)
            or scale.dtype != np.float32
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
            or order.shape != (29,)
            or order.dtype.kind not in ("i", "u")
            or sorted(order.tolist()) != list(range(29))
        ):
            raise ValueError("bounded existing feedback and exact HG29 action codec required")
        self.latent = latent.copy()
        self.scale = scale.copy()
        self.order = order.copy()
        self.previous = np.zeros(29, dtype=np.float64)

    def apply(
        self, proposal: np.ndarray, reference: np.ndarray, q: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Apply only a current observation, with actual rounded feedback state."""
        measured = body_feedback(q, v)
        requested = np.clip(feedback_delta(measured, self.latent), -0.08, 0.08)
        limited = self.previous + np.clip(requested - self.previous, -0.02, 0.02)
        oracle = proposal.copy()
        oracle[self.order] = (oracle[self.order] + limited / self.scale).astype(np.float32)
        applied, info = blend_actual_motor(
            proposal,
            oracle,
            reference,
            self.scale[np.argsort(self.order)],
            share=1.0,
        )
        actual = ((applied[self.order] - proposal[self.order]) * self.scale).astype(np.float64)
        if not np.isfinite(actual).all() or np.max(np.abs(actual)) > 0.080002:
            raise ValueError("actual feedback must remain within declared correction authority")
        self.previous = actual.copy()
        return applied, dict(
            BAL_feedback=measured,
            BAL_requested_HG=requested,
            BAL_actual_HG=actual,
            BAL_executed_action=applied.copy(),
            BAL_authority_clipped=info["oracle_authority_clipped"],
            BAL_is_learned_actor=False,
        )
