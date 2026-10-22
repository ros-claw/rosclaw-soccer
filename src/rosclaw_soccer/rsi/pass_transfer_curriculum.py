"""Explicit nearby TRAIN ball states; not an independent blind test bank.

Forty physical initial states, not160 independent samples. Goal metadata and
paired action arms do not multiply the distinct-state count. No optimization,
replacement of failures or simulator allocation occurs in this module.
"""

from typing import Any

import numpy as np


def initial_ball_offset(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (2,)
        or value.dtype.kind != "f"
        or not np.isfinite(value).all()
        or np.linalg.norm(value) > 0.05 + 1e-12
    ):
        raise ValueError("explicit finite planar initial ball perturbation <=.05m required")
    return value.copy()


def train_ball_states(forward_by_context: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    if set(forward_by_context) != {1, 4, 20, 21} or any(
        not isinstance(v, np.ndarray)
        or v.shape != (2,)
        or not np.isfinite(v).all()
        or abs(np.linalg.norm(v) - 1) > 1e-10
        for v in forward_by_context.values()
    ):
        raise ValueError("all4 declared unit pass directions required")
    rows: list[dict[str, Any]] = []
    offsets = (-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05)
    for context in (1, 4, 20, 21):
        forward = forward_by_context[context]
        lateral = np.array([-forward[1], forward[0]])
        for k, offset in enumerate(offsets):
            world = initial_ball_offset(lateral * offset)
            assert world is not None
            rows.append(
                dict(
                    state_id=len(rows),
                    context=context,
                    side=-0.55 if k % 2 == 0 else 0.55,
                    lateral_ball_offset_m=offset,
                    ball_planar_offset_world=world.tolist(),
                    bank="TRAIN",
                    not_fresh=True,
                )
            )
    return rows
