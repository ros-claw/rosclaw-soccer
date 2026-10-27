"""Causal SIM relay fall-window samples, not an action policy or safety gate."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelayLayout:
    root_q: int
    root_v: int
    ball_q: int
    ball_v: int
    joints_q: tuple[int, ...]
    joints_v: tuple[int, ...]


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    if quaternion.shape != (4,) or abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-5:
        raise ValueError("measured unit body orientation required")
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def relay_fall_samples(
    *,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_b: np.ndarray,
    body_failed: np.ndarray,
    layout: RelayLayout,
    horizon_controls: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-action features, future B-failure labels and source controls.

    A surviving suffix shorter than the horizon is right-censored and dropped.
    A known B-failure inside that suffix remains a positive. Inputs use only
    the prior physical boundary and last actual target; no future action,
    termination, course identity, reward or outcome is an input feature.
    """
    n = len(qpos)
    arrays = (initial_qpos, initial_qvel, qpos, qvel, target_b)
    if (
        n < 1
        or initial_qpos.shape != (79,)
        or initial_qvel.shape != (76,)
        or qpos.shape != (n, 10, 79)
        or qvel.shape != (n, 10, 76)
        or target_b.shape != (n, 29)
        or body_failed.shape != (n, 2)
        or body_failed.dtype != np.bool_
        or body_failed[:-1].any()
        or any(not np.isfinite(x).all() for x in arrays)
        or type(horizon_controls) is not int
        or not 1 <= horizon_controls <= 100
        or not isinstance(layout, RelayLayout)
        or len(layout.joints_q) != 29
        or len(layout.joints_v) != 29
        or len(set(layout.joints_q)) != 29
        or len(set(layout.joints_v)) != 29
        or any(type(i) is not int or not 0 <= i < 79 for i in layout.joints_q)
        or any(type(i) is not int or not 0 <= i < 76 for i in layout.joints_v)
        or type(layout.root_q) is not int
        or type(layout.ball_q) is not int
        or type(layout.root_v) is not int
        or type(layout.ball_v) is not int
        or not 0 <= layout.root_q <= 72
        or not 0 <= layout.ball_q <= 72
        or not 0 <= layout.root_v <= 70
        or not 0 <= layout.ball_v <= 70
    ):
        raise ValueError("valid complete native relay trace, layout and causal horizon required")
    q_indices = (
        *range(layout.root_q, layout.root_q + 7),
        *layout.joints_q,
        *range(layout.ball_q, layout.ball_q + 7),
    )
    v_indices = (
        *range(layout.root_v, layout.root_v + 6),
        *layout.joints_v,
        *range(layout.ball_v, layout.ball_v + 6),
    )
    if len(set(q_indices)) != 43 or len(set(v_indices)) != 41:
        raise ValueError("distinct measured root, joint and ball addresses required")
    event = n - 1 if body_failed[-1, 1] else None
    features, labels, controls = [], [], []
    for control in range(n):
        positive = event is not None and control <= event < control + horizon_controls
        if not positive and control + horizon_controls > n:
            continue
        q = initial_qpos if control == 0 else qpos[control - 1, -1]
        v = initial_qvel if control == 0 else qvel[control - 1, -1]
        rotation = _rotation(q[layout.root_q + 3 : layout.root_q + 7])
        root_velocity = v[layout.root_v : layout.root_v + 3]
        feature = np.concatenate(
            (
                rotation.T @ np.array([0.0, 0.0, -1.0]),
                np.array([q[layout.root_q + 2]]),
                rotation.T @ root_velocity,
                v[layout.root_v + 3 : layout.root_v + 6],
                rotation.T
                @ (q[layout.ball_q : layout.ball_q + 3] - q[layout.root_q : layout.root_q + 3]),
                rotation.T @ (v[layout.ball_v : layout.ball_v + 3] - root_velocity),
                q[list(layout.joints_q)],
                v[list(layout.joints_v)],
                np.zeros(29) if control == 0 else target_b[control - 1],
                np.array([float(control > 0)]),
            )
        )
        features.append(feature)
        labels.append(positive)
        controls.append(control)
    return (
        np.asarray(features, dtype=np.float32).reshape(-1, 104),
        np.asarray(labels, dtype=np.bool_),
        np.asarray(controls, dtype=np.int64),
    )
