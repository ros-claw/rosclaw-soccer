"""Pitch-aware root navigation constraints, not ball rules or a gait guarantee.

Combine these halfplanes with player clearance in one projection. Sequentially
clipping a collision-cleared command could violate its neighbor constraints.
No wall collision geometry or physical state is created or modified here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchRootBounds:
    left_x_m: float = -1.5
    right_x_m: float = 7.5
    half_width_m: float = 3.0
    root_margin_m: float = 0.4
    approach_gain: float = 0.6

    def __post_init__(self) -> None:
        values = (
            self.left_x_m,
            self.right_x_m,
            self.half_width_m,
            self.root_margin_m,
            self.approach_gain,
        )
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not -100 <= self.left_x_m < self.right_x_m <= 100
            or not 1 <= self.half_width_m <= 50
            or not 0.2 <= self.root_margin_m <= 0.8
            or self.right_x_m - self.left_x_m <= 2 * self.root_margin_m
            or not 0.1 <= self.approach_gain <= 2
        ):
            raise ValueError("bounded finite playable root rectangle required")

    def velocity_halfplanes(self, position_xy: np.ndarray) -> np.ndarray:
        position = np.asarray(position_xy)
        if (
            position.shape != (2,)
            or not np.issubdtype(position.dtype, np.floating)
            or not np.all(np.isfinite(position))
            or np.any(np.abs(position) > 100)
        ):
            raise ValueError("finite world-frame root position required")
        x, y = position
        margin, gain = self.root_margin_m, self.approach_gain
        return np.array(
            [
                [1.0, 0.0, gain * (self.left_x_m + margin - x)],
                [-1.0, 0.0, gain * (x - self.right_x_m + margin)],
                [0.0, 1.0, gain * (-self.half_width_m + margin - y)],
                [0.0, -1.0, gain * (y - self.half_width_m + margin)],
            ],
            dtype=float,
        )
