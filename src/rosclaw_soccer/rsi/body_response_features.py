"""Measured G1 state and local central-response labels, no controller.

The response matrix is a local secant of executed SIM interventions. It is not
a global dynamics model, policy, torque output, or authority to execute motion.
"""

from typing import Any

import numpy as np


def _matrix(value: Any, width: int) -> np.ndarray[Any, Any]:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != width or raw.dtype.kind not in "fiu":
        raise ValueError("complete numeric measured response matrix required")
    if not 1 <= len(raw) <= 20000:
        raise ValueError("bounded measured response rows required")
    result: np.ndarray[Any, Any] = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e6:
        raise ValueError("finite bounded measured response matrix required")
    return result


def measured_response_features(
    qpos: Any, qvel: Any, nominal_target: Any, relative_ball: Any
) -> np.ndarray[Any, Any]:
    """103 current-state features; no future state or foundation latent token.

    Canonical DDS joint order is caller-verified against source evidence.
    Ball coordinates retain the source's origin/COM reference, not a new one.
    """
    q, v, target, ball = (
        _matrix(value, width)
        for value, width in ((qpos, 43), (qvel, 41), (nominal_target, 29), (relative_ball, 6))
    )
    if not len(q) == len(v) == len(target) == len(ball):
        raise ValueError("aligned current physical rows required")
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("measured normalized root quaternion required")
    w, x, y, z = (q[:, i] for i in range(3, 7))
    gravity = np.column_stack((2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1))
    result: np.ndarray[Any, Any] = np.concatenate(
        (q[:, 2:3], gravity, q[:, 7:36], v[:, :35], target, ball), axis=1
    )
    result = result.astype(np.float32)
    if result.shape[1] != 103 or not np.isfinite(result).all():
        raise ValueError("finite current-state response features required")
    result.flags.writeable = False
    return result


def central_response_labels(
    actual_effects: Any, coordinates: Any, *, increment_rad: float, action_dimensions: int = 12
) -> np.ndarray[Any, Any]:
    effects = _matrix(actual_effects, 35)
    coords = np.asarray(coordinates)
    if (
        type(increment_rad) is not float
        or not np.isfinite(increment_rad)
        or not 0.001 <= increment_rad <= 0.05
        or type(action_dimensions) is not int
        or action_dimensions not in (12, 29)
        or len(effects) % (2 * action_dimensions)
        or coords.shape != (len(effects), 3)
        or coords.dtype.kind not in "iu"
    ):
        raise ValueError("complete bounded signed joint intervention pairs required")
    frames = coords[:: 2 * action_dimensions, 0]
    expected = np.asarray(
        [(f, j, s) for f in frames for j in range(action_dimensions) for s in (-1, 1)]
    )
    if (
        not np.array_equal(coords, expected)
        or len(set(frames.tolist())) != len(frames)
        or np.any(frames < 0)
        or np.any(frames >= 20000)
    ):
        raise ValueError("unique ordered paired response coordinates required")
    paired = effects.reshape(-1, action_dimensions, 2, 35)
    result: np.ndarray[Any, Any] = (
        (paired[:, :, 1] - paired[:, :, 0]) / (2 * increment_rad)
    ).transpose(0, 2, 1)
    result = result.reshape(-1, 35 * action_dimensions).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("finite local central-response labels required")
    result.flags.writeable = False
    return result


def local_velocity_response(
    jacobians: Any,
    target_increment: Any,
    *,
    action_dimensions: int = 12,
) -> np.ndarray[Any, Any]:
    """Local linear proposal only; exact zero by construction at zero increment."""
    if type(action_dimensions) is not int or action_dimensions not in (12, 29):
        raise ValueError("explicit canonical leg or full-body response dimensions required")
    jac, delta = (
        _matrix(jacobians, 35 * action_dimensions),
        _matrix(target_increment, action_dimensions),
    )
    if len(jac) != len(delta) or np.max(np.abs(delta)) > 0.02:
        raise ValueError("aligned small local target increments required")
    result: np.ndarray[Any, Any] = np.einsum(
        "nij,nj->ni", jac.reshape(-1, 35, action_dimensions), delta
    )
    if not np.isfinite(result).all():
        raise ValueError("finite local velocity proposal required")
    result.flags.writeable = False
    return result
