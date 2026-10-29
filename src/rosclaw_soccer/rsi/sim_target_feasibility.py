"""Explicitly experimental SIM target projection; never a hardware safety certificate."""

import numpy as np

from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


class TargetFeasibility:
    """Project onto immutable joint bounds and a 0.08 rad/control causal rate bound."""

    def __init__(self, initial: np.ndarray, limits: np.ndarray) -> None:
        if (
            type(initial) is not np.ndarray
            or type(limits) is not np.ndarray
            or initial.shape != (29,)
            or limits.shape != (29, 2)
            or initial.dtype != np.float64
            or limits.dtype != np.float64
            or not np.isfinite(initial).all()
            or not np.isfinite(limits).all()
            or (limits[:, 0] >= limits[:, 1]).any()
            or (initial < limits[:, 0]).any()
            or (initial > limits[:, 1]).any()
        ):
            raise ValueError("in-range finite measured initial HG29 and physical limits required")
        self._previous = initial.copy()
        self._limits = limits.copy()
        self._pending: np.ndarray | None = None

    def propose(self, target: np.ndarray) -> np.ndarray:
        if (
            self._pending is not None
            or type(target) is not np.ndarray
            or target.shape != (29,)
            or target.dtype != np.float64
            or not np.isfinite(target).all()
        ):
            raise ValueError("one finite fresh HG29 target per physical commit required")
        constrained = np.clip(target, self._limits[:, 0], self._limits[:, 1])
        projected = np.clip(constrained, self._previous - 0.08, self._previous + 0.08)
        if (projected < self._limits[:, 0]).any() or (projected > self._limits[:, 1]).any():
            raise ValueError("projected target violates physical range")
        self._pending = projected.copy()
        return projected

    def commit(self, actual: np.ndarray) -> None:
        if self._pending is None or not np.array_equal(actual, self._pending):
            raise ValueError("projected state advances only after exact physical commit")
        self._previous = self._pending
        self._pending = None


class ProjectedSharedOmniContactPlayer(SharedOmniContactPlayer):
    """One actual simulator actor with explicit causal target filtering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.all(self.world.model.jnt_limited[self.layout.joints]):
            self.failed = True
            raise ValueError("all HG joints require explicit physical ranges")
        self.feasibility = TargetFeasibility(
            self.world.data.qpos[self.layout.qi].astype(np.float64, copy=True),
            self.world.model.jnt_range[self.layout.joints].astype(np.float64, copy=True),
        )

    def install_motor_plan(self, plan):
        raise ValueError("residual plans are outside the declared projection experiment")

    def propose(self):
        target, measured = super().propose()
        try:
            projected = self.feasibility.propose(target)
            measured["target_projection"] = projected - target
            self.pending = projected.copy()
            return projected.copy(), measured
        except BaseException:
            self.failed = True
            raise

    def commit(self):
        pending = self.pending
        super().commit()
        try:
            self.feasibility.commit(pending)
        except BaseException:
            self.failed = True
            raise
