"""Public sixteen-state bounded teacher search, never a sealed examination."""

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_course_preflight import (
    FirstTouchCourse,
    preflight_receiving_courses,
)

BOUNDS = np.array([[0.5, 1.0], [0.45, 1.1], [180.0, 250.0], [-0.15, 0.15]])


def teacher_courses() -> tuple[FirstTouchCourse, ...]:
    """Four initial x/signed-v classes, four positions each, no outcome filtering."""
    rows = tuple(
        FirstTouchCourse("striker", 2026102500 + i, (x, y, vx))
        for i, (x, y, vx) in enumerate(
            (x, y, vx)
            for x in (2.25, 2.35, 2.65, 2.75)
            for vx in (-0.4, 0.4)
            for y in (-0.08, 0.08)
        )
    )
    preflight_receiving_courses(rows)
    return rows


def uniform_candidates(seed: int) -> np.ndarray:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("uint32 search seed required")
    return np.random.default_rng(seed).uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(8, 4))


def elite_candidates(
    first: np.ndarray, ranks: tuple[tuple[Any, ...], ...], *, seed: int
) -> np.ndarray:
    """Second and final CEM generation, two elites, bounded conditional normal.

    No clipping, resampling based on physical outcomes, or unlimited restarts.
    This is privileged offline simulation search, not a deployment policy.
    """
    x = np.asarray(first)
    if (
        x.shape != (8, 4)
        or not np.isfinite(x).all()
        or (x < BOUNDS[:, 0]).any()
        or (x > BOUNDS[:, 1]).any()
        or type(ranks) is not tuple
        or len(ranks) != 8
        or any(type(r) is not tuple or not r for r in ranks)
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("complete first eight bounded candidates and ordered ranks required")
    indices = sorted(range(8), key=lambda i: ranks[i], reverse=True)[:2]
    mean = x[indices].mean(axis=0)
    std = np.maximum(x[indices].std(axis=0), 0.15 * (BOUNDS[:, 1] - BOUNDS[:, 0]))
    rng = np.random.default_rng(seed)
    accepted = []
    for _ in range(10000):
        row = rng.normal(mean, std)
        if np.all(row >= BOUNDS[:, 0]) and np.all(row <= BOUNDS[:, 1]):
            accepted.append(row)
            if len(accepted) == 8:
                return np.stack(accepted)
    raise ValueError("bounded conditional sampler exhausted before physical allocation")


def candidate_parameters(row: np.ndarray) -> dict[str, Any]:
    x = np.asarray(row)
    if (
        x.shape != (4,)
        or not np.isfinite(x).all()
        or ((x < BOUNDS[:, 0]).any() or (x > BOUNDS[:, 1]).any())
    ):
        raise ValueError("finite teacher parameter vector within preregistered bounds required")
    return dict(
        speed_mps=float(x[0]),
        entry_gap_m=float(x[1]),
        reference_frame=int(np.rint(x[2])),
        root_lateral_offset_m=float(x[3]),
    )


def outcome_rank(row: dict[str, Any]) -> tuple[int, int, float, int]:
    outcome = row["outcome"]
    safe = row["healthy_full_horizon"] and outcome["safe"]
    first = outcome["first_foot_contact_frame"]
    return (
        int(safe),
        int(safe and outcome["task_pass"]),
        float(outcome["training_reward"]),
        -300 if first is None else -int(first),
    )
