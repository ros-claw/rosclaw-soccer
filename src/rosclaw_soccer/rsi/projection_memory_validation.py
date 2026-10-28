"""Validate actual policy observation history for target projection research."""

import numpy as np

from rosclaw_soccer.rsi.receiver_target_limit_projection import projected_action_memory


def validate_projection_memory(
    *,
    observations: np.ndarray,
    raw_actions: np.ndarray,
    targets: np.ndarray,
    changed_axes: np.ndarray,
    memory_after: np.ndarray,
    offset: np.ndarray,
    scale: np.ndarray,
    reflect_projection: bool,
) -> dict:
    """Check each own next observation, not a substitute/donor recurrent history."""
    if not isinstance(raw_actions, np.ndarray) or raw_actions.ndim != 2:
        raise ValueError("finite action history required")
    n = len(raw_actions)
    if (
        type(reflect_projection) is not bool
        or n < 1
        or not isinstance(observations, np.ndarray)
        or observations.shape != (n, 98)
        or observations.dtype != np.float32
        or not isinstance(raw_actions, np.ndarray)
        or raw_actions.shape != (n, 29)
        or raw_actions.dtype != np.float32
        or not isinstance(targets, np.ndarray)
        or targets.shape != (n, 29)
        or targets.dtype != np.float64
        or not isinstance(changed_axes, np.ndarray)
        or changed_axes.shape != (n, 29)
        or changed_axes.dtype != np.bool_
        or not isinstance(memory_after, np.ndarray)
        or memory_after.shape != (n, 29)
        or memory_after.dtype != np.float32
        or any(not np.isfinite(x).all() for x in (observations, raw_actions, targets, memory_after))
    ):
        raise ValueError("full finite typed same-episode action and observation histories required")
    expected = np.stack(
        [
            projected_action_memory(raw_actions[i], targets[i], offset, scale, changed_axes[i])
            for i in range(n)
        ]
    )
    wanted = expected if reflect_projection else raw_actions
    if not np.array_equal(memory_after, wanted):
        raise ValueError("declared actor action-memory semantics differ from execution")
    if not np.array_equal(observations[1:, -29:], memory_after[:-1]):
        raise ValueError(
            "next actual actor observation did not carry the prior owned action memory"
        )
    mismatch = np.abs(expected.astype(np.float64) - raw_actions.astype(np.float64))
    return dict(
        controls=n,
        verified_next_observations=n - 1,
        projection_changed_controls=int(changed_axes.any(axis=1).sum()),
        max_normalized_raw_vs_executed_delta=float(mismatch.max()),
        reflected_projection=reflect_projection,
        actual_neural_history_verified=True,
    )
