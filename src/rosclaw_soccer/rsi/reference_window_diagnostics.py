"""Explicit temporal envelope for diagnostic-only reference adjustments."""

import numpy as np


def cosine_envelope(frames: np.ndarray, *, center: int, half_width: int) -> np.ndarray:
    """Return a compact C1 envelope; never imply physical tracking feasibility."""
    if (
        type(frames) is not np.ndarray
        or frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or type(center) is not int
        or type(half_width) is not int
        or half_width < 1
    ):
        raise ValueError("integer frames and positive integer window required")
    phase = np.abs(frames.astype(np.float64) - center) / half_width
    return np.where(phase < 1, (1 + np.cos(np.pi * phase)) / 2, 0.0)
