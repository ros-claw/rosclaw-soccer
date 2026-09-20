"""Pure bounded proposal refinement, separate from physics and task scoring.

An unchanged policy is a valid incumbent but is not a zero action vector.
Never fit the search distribution to that fictitious vector.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleEvaluation:
    proposal: tuple[float, ...] | None
    captured: bool
    safe: bool
    score: float

    def __post_init__(self) -> None:
        if (
            type(self.captured) is not bool
            or type(self.safe) is not bool
            or type(self.score) not in (int, float)
            or not math.isfinite(self.score)
        ):
            raise ValueError("finite score and explicit physical judgments required")
        if self.proposal is not None and (
            type(self.proposal) is not tuple
            or not 1 <= len(self.proposal) <= 512
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1
                for v in self.proposal
            )
        ):
            raise ValueError("bounded normalized proposal or explicit nonvector incumbent required")

    @property
    def rank(self) -> tuple[int, int, float]:
        return int(self.safe and self.captured), int(self.safe), float(self.score)


def refine_oracle_population(
    evaluations: tuple[OracleEvaluation, ...], *, seed: int, count: int = 8
) -> tuple[tuple[float, ...], ...]:
    """Top-two diagonal CEM with fixed variance floor and normalized clipping.

    Retaining a nonvector incumbent for final selection is the caller's job;
    it is excluded only from the proposal distribution fit. This function
    neither authenticates physical scores nor authorizes another rollout.
    """
    if (
        type(evaluations) is not tuple
        or not 2 <= len(evaluations) <= 64
        or any(not isinstance(row, OracleEvaluation) for row in evaluations)
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or type(count) is not int
        or not 1 <= count <= 64
    ):
        raise ValueError("bounded complete search observations and seed required")
    for row in evaluations:
        row.__post_init__()
    rows = [row for row in evaluations if row.proposal is not None]
    if len(rows) < 2 or len({len(row.proposal or ()) for row in rows}) != 1:
        raise ValueError("two actual equal-dimension proposal evaluations required")
    elites = sorted(rows, key=lambda row: row.rank, reverse=True)[:2]
    vectors = np.asarray([row.proposal for row in elites], dtype=np.float64)
    mean = vectors.mean(axis=0)
    sigma = np.maximum(vectors.std(axis=0), 0.08)
    proposals = np.clip(
        np.random.default_rng(seed).normal(mean, sigma, size=(count, vectors.shape[1])), -1, 1
    )
    return tuple(tuple(float(value) for value in row) for row in proposals)
