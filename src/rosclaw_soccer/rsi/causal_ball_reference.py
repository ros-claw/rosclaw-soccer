"""Bounded SIM-only causal correction of a right-ankle motion reference."""

import numpy as np


class CausalBallReference:
    """One policy's immutable source reference and sequential measured ball updates.

    Reference positions are neural guidance, not actuator targets or physics.
    This does not train a network, establish contact, or authorize a policy.
    """

    def __init__(self, policy):
        ankle = policy.ref_right_ankle_future_pos
        ball = policy.ref_object_pos
        if (
            policy.reference_source != "NPZmotion"
            or policy.counter_step != 0
            or not isinstance(ankle, np.ndarray)
            or not isinstance(ball, np.ndarray)
            or ankle.ndim != 2
            or ankle.shape[1:] != (3,)
            or ball.shape != ankle.shape
            or not len(ankle)
            or ankle.dtype.kind != "f"
            or ball.dtype.kind != "f"
            or not np.isfinite(ankle).all()
            or not np.isfinite(ball).all()
        ):
            raise ValueError("fresh finite aligned motion reference required")
        self.policy = policy
        self.ankle = ankle.copy()
        self.ball = ball.copy()
        self.expected_ankle = ankle.copy()
        self.offset = np.zeros(2, dtype=np.float64)
        self.tick = 0

    def apply(self, measured_ball):
        """Update only the policy's right-ankle future guidance once per tick."""
        p = self.policy
        if (
            p.reference_source != "NPZmotion"
            or p.counter_step != self.tick
            or not np.array_equal(p.ref_right_ankle_future_pos, self.expected_ankle)
            or not np.array_equal(p.ref_object_pos, self.ball)
            or not isinstance(measured_ball, np.ndarray)
            or measured_ball.shape != (3,)
            or measured_ball.dtype.kind != "f"
            or not np.isfinite(measured_ball).all()
        ):
            raise ValueError("fresh measured ball and unchanged private reference required")
        if self.tick < len(self.ball):
            target = measured_ball[:2].astype(np.float64) - self.ball[self.tick, :2]
            size = float(np.linalg.norm(target))
            if size > 0.35:
                target *= 0.35 / size
        else:
            target = np.zeros(2, dtype=np.float64)
        change = target - self.offset
        size = float(np.linalg.norm(change))
        if size > 0.02:
            change *= 0.02 / size
        self.offset = self.offset + change
        updated = self.ankle.copy()
        updated[:, :2] += self.offset.astype(updated.dtype)
        p.ref_right_ankle_future_pos = updated
        self.expected_ankle = updated.copy()
        self.tick += 1
        return self.offset.copy()
