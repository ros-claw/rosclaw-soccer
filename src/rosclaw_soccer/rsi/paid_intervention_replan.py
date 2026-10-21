"""Finite on-student-prefix teacher planning, never a deployed skill or RL update."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def replan_config(center: int, horizon: int) -> dict[str, Any]:
    if type(center) is not int or center != 104 or type(horizon) is not int or horizon != 289:
        raise ValueError("fixed audited first-contact fixture required")
    return dict(
        protocol="paid.TRAIN.interventional_own_prefix_CEM.v1",
        cases=[4, 5],
        switch_controls=[center - 34, center - 24],
        center=center,
        horizon=horizon,
        generations=2,
        population=12,
        elite_count=4,
        selected_repeats=2,
        seed=2026103200,
        max_episodes=112,
        max_native_steps=435680,
        neural_optimizer_updates=0,
        student_seed=0,
        original_R2_relative_bound_rad=0.2,
        Fresh_rows_read=0,
        DEV_fit_rows=0,
        purpose="collect physically successful expert actions on current student states",
    )


def sample_latents(
    rng: np.random.Generator,
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    incumbent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    for x in (mean, std, incumbent):
        if not isinstance(x, np.ndarray) or x.shape != (2, 6) or not np.isfinite(x).all():
            raise ValueError("finite twelve-dimensional planner state required")
    if (std < 0.1).any() or (np.abs(incumbent) > 1).any():
        raise ValueError("positive exploration and bounded incumbent required")
    unbounded = rng.normal(mean, std, size=(12, 2, 6))
    unbounded[0] = np.arctanh(np.clip(incumbent, -0.999999, 0.999999))
    bounded = np.tanh(unbounded)
    bounded[0] = incumbent
    return unbounded, bounded


def elite_update(
    unbounded: NDArray[np.float64], ranks: list[tuple[Any, ...]]
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    if (
        unbounded.shape != (12, 2, 6)
        or not np.isfinite(unbounded).all()
        or len(ranks) != 12
        or any(not isinstance(r, tuple) or not r for r in ranks)
        or not np.isfinite(np.asarray(ranks, dtype=np.float64)).all()
    ):
        raise ValueError("all twelve actual finite paired physical ranks required")
    elite = sorted(range(12), key=lambda i: ranks[i], reverse=True)[:4]
    return unbounded[elite].mean(axis=0), np.maximum(unbounded[elite].std(axis=0), 0.1), elite
