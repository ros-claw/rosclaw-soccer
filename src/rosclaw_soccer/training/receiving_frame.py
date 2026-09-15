"""Explicit half-turn of world-vector fields in the 135-feature receiver.

This changes only the observation frame, not physics, joint handedness, or
policy weights. It is not an arbitrary rotation or an egocentric observation.
"""

from __future__ import annotations

import math

import numpy as np

CANONICAL_RECEIVER_CONTRACT = "recurrent_receiver_canonical_heading_135_float32.v4"
CANONICAL_POSE_RECEIVER_CONTRACT = "recurrent_receiver_canonical_pose_135_float32.v5"
HALF_TURN_FIELDS = (61, 62, 67, 68, 70, 71, 133, 134)


def canonical_receiving_features(features: np.ndarray, *, half_turn: bool) -> np.ndarray:
    """Return a private float32 copy; invert only world XY and heading pairs."""
    if type(half_turn) is not bool:
        raise ValueError("explicit boolean half-turn required")
    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.float32
        or features.ndim not in (1, 2)
        or features.shape[-1] != 135
        or features.size == 0
        or features.size > 4096 * 135
        or not np.isfinite(features).all()
    ):
        raise ValueError("finite bounded-batch 135-feature float32 observation required")
    result = features.copy()
    if half_turn:
        result[..., list(HALF_TURN_FIELDS)] *= -1
    return result


def receiving_frame_translation(value: tuple[float, float]) -> tuple[float, float]:
    """Validate explicit world-XY translation used by the half-turn projection."""
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 200 for x in value)
    ):
        raise ValueError("explicit bounded immutable receiving frame translation required")
    return float(value[0]), float(value[1])


def canonical_receiving_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    half_turn: bool,
    translation_xy_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Project private feature-building copies BEFORE float32 conversion.

    q'=translation-q in world XY, and Rz(pi) multiplies both free-joint
    quaternions. Joint coordinates and local angular velocities stay unchanged.
    These arrays are neural feature inputs, never commands or simulator writes.
    Pre-quantization matters: subtracting already rounded world positions and
    negating sin(atan2(...)) need not reproduce the legacy opposite-half bits.
    """
    translation = receiving_frame_translation(translation_xy_m)
    if type(half_turn) is not bool:
        raise ValueError("explicit boolean half-turn required")
    if any(
        not isinstance(a, np.ndarray)
        or a.dtype != np.float64
        or a.shape != shape
        or not np.isfinite(a).all()
        or (np.abs(a) > 1e4).any()
        for a, shape in ((qpos, (43,)), (qvel, (41,)))
    ):
        raise ValueError("finite bounded float64 receiving state copies required")
    if any(abs(float(np.linalg.norm(qpos[i : i + 4])) - 1) > 1e-4 for i in (3, 39)):
        raise ValueError("normalized measured receiving free-joint quaternions required")
    p, v = qpos.copy(), qvel.copy()
    if half_turn:
        for start in (0, 36):
            p[start : start + 2] = np.asarray(translation) - p[start : start + 2]
        for start in (3, 39):
            w, x, y, z = p[start : start + 4].copy()
            p[start : start + 4] = (-z, -y, x, w)
        v[[0, 1, 35, 36]] *= -1
    return p, v
