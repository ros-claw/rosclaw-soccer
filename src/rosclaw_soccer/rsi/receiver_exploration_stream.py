"""SIM-only bounded own-history receiver exploration, without motor authority."""

import numpy as np

from rosclaw_soccer.rsi.shared_motor_sampling import SharedMotorSampling


class ReceiverExplorationStream:
    """One proposal then one explicit actual-action commit per control.

    The caller owns physical execution and must pass the same applied action to
    its single receiving motor. This object cannot step physics or update an
    actor. Failed/missing commits are terminal; no automatic retry or reset.
    """

    simulation_only = True

    def __init__(self, scale_policy: np.ndarray, *, seed: int) -> None:
        self._sampler = SharedMotorSampling(scale_policy, seed=seed, profile="contact-v1")
        self._pending: np.ndarray | None = None
        self._boundary = 0
        self._faulted = False

    def propose(
        self, mean: np.ndarray, reference: np.ndarray, *, control: int
    ) -> tuple[np.ndarray, dict]:
        if self._faulted:
            raise ValueError("receiver exploration fault is latched")
        try:
            if self._pending is not None or type(control) is not int or control != self._boundary:
                raise ValueError("exactly one proposal before each committed control required")
            action, trace = self._sampler.propose(mean, reference, control=control, standing=False)
            self._pending = action.copy()
            return action.copy(), trace
        except Exception:
            self._faulted = True
            raise

    def commit(self, actual: np.ndarray) -> None:
        if self._faulted:
            raise ValueError("receiver exploration fault is latched")
        try:
            if (
                self._pending is None
                or not isinstance(actual, np.ndarray)
                or actual.shape != (29,)
                or actual.dtype != np.float32
                or not np.array_equal(actual, self._pending)
            ):
                raise ValueError("exact actually executed sampled action required")
            self._pending = None
            self._boundary += 1
        except Exception:
            self._faulted = True
            raise
