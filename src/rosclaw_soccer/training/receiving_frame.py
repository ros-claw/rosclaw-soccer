"""Explicit half-turn of world-vector fields in the 135-feature receiver.

This changes only the observation frame, not physics, joint handedness, or
policy weights. It is not an arbitrary rotation or an egocentric observation.
"""

from __future__ import annotations

import numpy as np

CANONICAL_RECEIVER_CONTRACT = "recurrent_receiver_canonical_heading_135_float32.v4"
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
