"""Read back a passer's own shadow and active velocity observation history."""

import numpy as np


def validate_passer_velocity_history(
    *,
    observations: np.ndarray,
    proposed: np.ndarray,
    actual_targets: np.ndarray,
    memory_after: np.ndarray,
    active: np.ndarray,
    offset: np.ndarray,
    scale: np.ndarray,
    transition: np.ndarray | None = None,
) -> dict:
    if not isinstance(active, np.ndarray) or active.ndim != 1:
        raise ValueError("one explicit consecutive active history required")
    n = len(active)
    if transition is None:
        transition = np.zeros(n, dtype=bool)
    arrays = (
        (observations, (n, 98), np.float32),
        (proposed, (n, 29), np.float32),
        (actual_targets, (n, 29), np.float64),
        (memory_after, (n, 29), np.float32),
        (offset, (29,), np.float64),
        (scale, (29,), np.float64),
    )
    if (
        n < 1
        or active.dtype != np.bool_
        or not isinstance(transition, np.ndarray)
        or transition.shape != (n,)
        or transition.dtype != np.bool_
        or np.any(transition & ~active)
        or np.any(active[:-1] & ~active[1:])
        or any(
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype != dtype
            or not np.isfinite(a).all()
            for a, shape, dtype in arrays
        )
        or np.any(scale <= 0)
    ):
        raise ValueError("finite typed own trajectory and irreversible activation required")
    encoded = ((actual_targets - offset) / scale).astype(np.float32)
    direct = active & ~transition
    wanted = np.where(direct[:, None], proposed, encoded)
    if not np.array_equal(memory_after, wanted):
        raise ValueError("shadow memory must encode actual targets, active memory own actions")
    if np.any(observations[0, -29:] != 0) or not np.array_equal(
        observations[1:, -29:], memory_after[:-1]
    ):
        raise ValueError("each actual actor observation must carry its own previous action")
    if np.any(direct) and not np.allclose(
        actual_targets[direct],
        offset + proposed[direct].astype(np.float64) * scale,
        atol=3e-7,
        rtol=0,
    ):
        raise ValueError("active actor targets must be actually applied")
    first = int(np.flatnonzero(active)[0]) if active.any() else None
    return dict(
        controls=n,
        verified_next_observations=n - 1,
        shadow_controls=int((~active).sum()),
        active_controls=int(active.sum()),
        blended_controls=int(transition.sum()),
        first_active_control=first,
        actual_owned_action_history_verified=True,
    )
