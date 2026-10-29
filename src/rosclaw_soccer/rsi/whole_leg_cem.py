"""Deterministic paired SIM-only CEM scoring, without policy execution."""

import numpy as np


def initial_population(seed: int = 20261029) -> tuple[np.random.Generator, np.ndarray, np.ndarray]:
    if seed != 20261029:
        raise ValueError("predeclared seed required")
    return np.random.default_rng(seed), np.zeros(12), np.full(12, 0.125)


def population(rng: np.random.Generator, mean: np.ndarray, sigma: np.ndarray):
    if (
        not isinstance(rng, np.random.Generator)
        or mean.shape != (12,)
        or sigma.shape != (12,)
        or not np.isfinite(mean).all()
        or not np.isfinite(sigma).all()
        or (sigma < 0.025).any()
    ):
        raise ValueError("finite declared 12-D normal distribution required")
    raw = rng.normal(mean, sigma, size=(16, 12))
    return raw, np.clip(raw, -0.25, 0.25)


def case_score(result: dict, ball_qpos: int) -> dict:
    """Retain censoring explicitly; this is a search rank, not task success."""
    qpos = np.asarray(result["qpos"])
    forces = np.asarray(result["ball_forces"])
    controls = result["controls"]
    if (
        type(ball_qpos) is not int
        or type(controls) is not int
        or not 0 <= controls <= 300
        or qpos.ndim != 2
        or qpos.shape[0] != controls * 10
        or not 0 <= ball_qpos < qpos.shape[1]
        or forces.shape != (controls * 10, 1, 3)
        or not np.isfinite(qpos).all()
        or not np.isfinite(forces).all()
    ):
        raise ValueError("finite, shape-consistent native forecast required")
    complete = controls == 300 and result["runtime_failure"] is None and not result["body_failed"]
    foot = np.flatnonzero((forces[:, 0, :2] > 10).any(axis=1))
    nonfoot = np.flatnonzero(forces[:, 0, 2] > 1)
    safe = complete and len(nonfoot) == 0
    strong = safe and len(foot) > 0
    contact = int(foot[0]) if len(foot) else None
    end = contact + 500 if contact is not None else None
    displacement = (
        float(qpos[end, ball_qpos] - qpos[contact, ball_qpos])
        if strong and end is not None and end < len(qpos)
        else 0.0
    )
    return dict(
        complete=int(complete),
        safe=int(safe),
        strong_foot=int(strong),
        forward_displacement_m=displacement,
        first_nonfoot_index=int(nonfoot[0]) if len(nonfoot) else (3000 if complete else 0),
        first_strong_foot_index=contact,
        successor_censored=bool(strong and end is not None and end >= len(qpos)),
    )


def pair_rank(first: dict, second: dict) -> tuple:
    fields = ("complete", "safe", "strong_foot", "forward_displacement_m", "first_nonfoot_index")
    return tuple(first[k] + second[k] for k in fields)


def update_elites(
    candidates: np.ndarray, ranks: list[tuple]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if candidates.shape != (16, 12) or len(ranks) != 16 or not np.isfinite(candidates).all():
        raise ValueError("all 16 finite paired candidate outcomes required")
    indices = sorted(range(16), key=lambda i: (tuple(-x for x in ranks[i]), i))[:4]
    elites = candidates[indices]
    return elites.mean(axis=0), np.maximum(elites.std(axis=0), 0.025), indices
