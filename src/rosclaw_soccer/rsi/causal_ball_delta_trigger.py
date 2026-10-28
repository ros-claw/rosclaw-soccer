"""A bounded SIM diagnostic trigger; not a certified kick detector."""

import numpy as np


class CausalBallDeltaTrigger:
    """Latch on observed 3D ball velocity changes at successive controls."""

    def __init__(self, previous_velocity: np.ndarray, *, threshold_mps: float) -> None:
        if not np.isfinite(threshold_mps) or not 0.01 <= threshold_mps <= 5:
            raise ValueError("finite bounded speed-change threshold required")
        self.previous = self._velocity(previous_velocity)
        self.threshold = float(threshold_mps)
        self.triggered = False

    @staticmethod
    def _velocity(value: np.ndarray) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("finite current 3D ball velocity required")
        return np.asarray(value, dtype=np.float64).copy()

    def observe(self, current_velocity: np.ndarray) -> tuple[bool, float]:
        current = self._velocity(current_velocity)
        delta = float(np.linalg.norm(current - self.previous))
        self.previous = current
        self.triggered = self.triggered or delta >= self.threshold
        return self.triggered, delta
