"""Current physical control features and measured body/ball response labels.

Unlike body-only recovery, these labels include the ball's real linear
velocity. No desired velocity, outcome, contact time or future state enters
the features. This is SIM dataset math, not a policy or motion permission.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.body_response_features import _matrix, central_response_labels


def _rotation(quaternion: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("normalized measured root and ball quaternions required")
    w, x, y, z = quaternion.T
    result = np.empty((len(quaternion), 3, 3))
    result[:, 0, :] = np.column_stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))
    )
    result[:, 1, :] = np.column_stack(
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))
    )
    result[:, 2, :] = np.column_stack(
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
    )
    return result


def measured_body_ball_features(qpos: Any, qvel: Any, nominal_target: Any) -> np.ndarray[Any, Any]:
    """112 current physical features, invariant to global horizontal translation.

    Root rotation retains yaw. Ball spin is rotated from its native free-joint
    axes into world axes. Relative ball state uses current canonical physical
    origin/linear velocity, not the historical cached-COM feature convention.
    Callers must verify canonical joint order against the source body contract.
    """
    q, v, target = (_matrix(a, n) for a, n in ((qpos, 43), (qvel, 41), (nominal_target, 29)))
    if not len(q) == len(v) == len(target):
        raise ValueError("aligned current physical body and ball rows required")
    root_rotation, ball_rotation = _rotation(q[:, 3:7]), _rotation(q[:, 39:43])
    spin = np.einsum("nij,nj->ni", ball_rotation, v[:, 38:41])
    features: np.ndarray[Any, Any] = np.concatenate(
        (
            q[:, 2:3],
            root_rotation.reshape(-1, 9),
            q[:, 7:36],
            v[:, :35],
            target,
            q[:, 36:39] - q[:, :3],
            v[:, 35:38] - v[:, :3],
            spin,
        ),
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != 112 or not np.isfinite(features).all():
        raise ValueError("complete finite physical body-ball features required")
    features.flags.writeable = False
    return features


def central_body_ball_labels(
    actual_effects: Any, coordinates: Any, *, increment_rad: float, action_dimensions: int = 29
) -> np.ndarray[Any, Any]:
    """38-by-action secants: native root/joints35 followed by world ball velocity3."""
    effects = _matrix(actual_effects, 38)
    body = central_response_labels(
        effects[:, :35],
        coordinates,
        increment_rad=increment_rad,
        action_dimensions=action_dimensions,
    ).reshape(-1, 35, action_dimensions)
    paired = effects[:, 35:38].reshape(-1, action_dimensions, 2, 3)
    ball = ((paired[:, :, 1] - paired[:, :, 0]) / (2 * increment_rad)).transpose(0, 2, 1)
    labels: np.ndarray[Any, Any] = (
        np.concatenate((body, ball), axis=1).reshape(-1, 38 * action_dimensions).astype(np.float32)
    )
    if not np.isfinite(labels).all():
        raise ValueError("finite measured body and ball response labels required")
    labels.flags.writeable = False
    return labels
