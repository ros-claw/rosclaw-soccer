"""First-episode motion tracking accounting, not football scoring."""

from typing import Any

import numpy as np
import numpy.typing as npt


class MotionFirstEpisode:
    """Ignore all post-reset rewards and successes after a world's first done."""

    def __init__(self, worlds: int):
        if type(worlds) is not int or not 1 <= worlds <= 512:
            raise ValueError("bounded evaluation batch required")
        self.active = np.ones(worlds, dtype=np.bool_)
        self.returns = np.zeros(worlds, dtype=np.float64)
        self.steps = np.zeros(worlds, dtype=np.int64)
        self.completed = np.zeros(worlds, dtype=np.bool_)
        self.failed = np.zeros(worlds, dtype=np.bool_)

    def observe(
        self,
        reward: npt.NDArray[np.float32],
        terminated: npt.NDArray[np.bool_],
        truncated: npt.NDArray[np.bool_],
        reference_exhausted: npt.NDArray[np.bool_],
    ) -> None:
        shape = self.active.shape
        if np.asarray(reward).shape != shape or not np.isfinite(reward).all():
            raise ValueError("finite per-world actual reward required")
        for flag in (terminated, truncated, reference_exhausted):
            if np.asarray(flag).shape != shape or np.asarray(flag).dtype != np.bool_:
                raise ValueError("per-world boolean termination flags required")
        if np.any(reference_exhausted & ~truncated):
            raise ValueError("reference exhaustion requires explicit timeout boundary")
        before = self.active.copy()
        self.returns[before] += reward[before]
        self.steps[before] += 1
        finished = before & (terminated | truncated)
        self.completed[finished] = reference_exhausted[finished] & ~terminated[finished]
        self.failed[finished] = terminated[finished]
        self.active[finished] = False

    def summary(self) -> dict[str, Any]:
        # A non-reference timeout is censored, not a fabricated tracking pass.
        censored = ~(self.completed | self.failed)
        return dict(
            worlds=len(self.active),
            completed=self.completed.tolist(),
            failed=self.failed.tolist(),
            right_censored=censored.tolist(),
            first_episode_steps=self.steps.tolist(),
            first_episode_returns=self.returns.tolist(),
            completed_count=int(self.completed.sum()),
            failed_count=int(self.failed.sum()),
            right_censored_count=int(censored.sum()),
            mean_first_episode_return=float(self.returns.mean()),
            mean_first_episode_steps=float(self.steps.mean()),
            football_skill_proven=False,
        )
