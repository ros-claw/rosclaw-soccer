"""Duration-aware returns for decision points in one terminated SIM episode.

This numeric primitive does not establish causal responsibility, authorize
training, or choose which actions are eligible for a policy gradient. Callers
must validate on-policy action selection, reward timing, and episode boundaries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def duration_aware_gae(
    rewards: ArrayLike,
    decisions: ArrayLike,
    values: ArrayLike,
    *,
    terminated: bool,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute decision-point GAE across intervening environment transitions.

    ``rewards[t]`` belongs to transition t -> t+1; the final transition must be
    a genuine terminal with zero continuation value. ``decisions`` contains
    strictly increasing transition indices, and ``values`` holds one prediction
    per decision state. Intervening rewards are discounted, while each GAE
    carry uses (gamma * lambda) ** elapsed transitions. Truncations, partial
    episodes and reset crossings require a different contract and are rejected
    by the caller; this function cannot infer them from numeric arrays.
    """
    r = np.asarray(rewards, dtype=np.float64)
    raw_d = np.asarray(decisions)
    v = np.asarray(values, dtype=np.float64)
    if (
        terminated is not True
        or r.ndim != 1
        or not len(r)
        or raw_d.ndim != 1
        or not len(raw_d)
        or raw_d.dtype.kind not in "iu"
        or v.shape != raw_d.shape
        or not np.isfinite(r).all()
        or not np.isfinite(v).all()
        or not np.isfinite(gamma)
        or not 0 <= gamma <= 1
        or not np.isfinite(gae_lambda)
        or not 0 <= gae_lambda <= 1
    ):
        raise ValueError("finite rewards, integer decisions, values and bounded discounts required")
    d = raw_d.astype(np.int64, copy=False)
    if d[0] < 0 or d[-1] >= len(r) or np.any(np.diff(d) <= 0):
        raise ValueError("decisions must be unique ordered transition indices")
    advantage = np.empty(len(d), dtype=np.float64)
    carry = 0.0
    for i in range(len(d) - 1, -1, -1):
        start = int(d[i])
        stop = int(d[i + 1]) if i + 1 < len(d) else len(r)
        duration = stop - start
        accumulated = float(np.dot(gamma ** np.arange(duration), r[start:stop]))
        next_value = float(v[i + 1]) if i + 1 < len(d) else 0.0
        delta = accumulated + gamma**duration * next_value - v[i]
        carry = delta + (gamma * gae_lambda) ** duration * carry
        advantage[i] = carry
    return advantage, advantage + v
