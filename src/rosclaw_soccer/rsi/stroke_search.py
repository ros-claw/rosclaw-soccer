"""Finite two-condition teacher CEM math, never learned motor credit."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motor_option_credit import contact_skill_return

SEED = 55023
GENERATIONS = 2
CANDIDATES = 16
CONTEXTS = (4, 20)
CONTROL = 21
WORLDS = 72


def checked_latent(value: Any) -> np.ndarray:
    z = np.asarray(value, dtype=np.float64)
    if z.shape != (2, 6) or not np.isfinite(z).all() or (abs(z) > 1).any():
        raise ValueError("bounded twelve-dimensional coherent stroke required")
    return z.copy()


def sample_generation(
    rng: np.random.Generator, mean: np.ndarray, std: np.ndarray
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    mean = checked_latent(mean)
    std = np.asarray(std, dtype=np.float64)
    if std.shape != (2, 6) or not np.isfinite(std).all() or (std < 0.1).any() or (std > 0.4).any():
        raise ValueError("preregistered bounded CEM spread required")
    accepted, draws = [], []
    for _ in range(4096):
        z = rng.normal(mean, std)
        valid = bool(np.isfinite(z).all() and (abs(z) <= 1).all())
        draws.append(dict(latent=z.tolist(), accepted=valid))
        if valid:
            accepted.append(z)
        if len(accepted) == CANDIDATES:
            return accepted, draws
    raise ValueError("bounded CEM rejection budget exhausted; no clipping/replacement")


def pair_rank(rows: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    if len(rows) != 2 or [r["context"] for r in rows] != list(CONTEXTS):
        raise ValueError("both fixed physical conditions in declared order required")
    return (
        int(all(not r["result"]["body_failed"] for r in rows)),
        sum(r["healthy_quality"] for r in rows),
        sum(not r["result"]["body_failed"] and not r["result"]["forbidden"] for r in rows),
        sum(contact_skill_return(r["result"]) for r in rows),
    )


def update_elites(
    points: list[np.ndarray], ranks: list[tuple[int, int, int, float]]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if len(points) != 17 or len(ranks) != 17:
        raise ValueError("one retained best plus sixteen actual paired trials required")
    values = np.stack([checked_latent(p) for p in points])
    elite = sorted(range(17), key=lambda i: (ranks[i], -i), reverse=True)[:4]
    return values[elite].mean(0), np.clip(values[elite].std(0), 0.1, 0.4), elite
