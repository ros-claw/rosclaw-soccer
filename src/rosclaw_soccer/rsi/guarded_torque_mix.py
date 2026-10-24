"""Recheck the actual mixed SIM command; convex mixing alone is not a guard."""

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rosclaw.simforge.g1_neural_torque import (
        G1TorqueControlFrame,
        G1TorqueProjection,
        G1TorqueSafetyConfig,
    )


def guarded_torque_mix(
    proposed: np.ndarray,
    parent: np.ndarray,
    previous: np.ndarray,
    *,
    frame: "G1TorqueControlFrame",
    safety: "G1TorqueSafetyConfig",
    fraction: float,
) -> "G1TorqueProjection":
    if (
        not math.isfinite(fraction)
        or not 0 < fraction < 1
        or any(
            np.asarray(v).shape != (29,) or not np.isfinite(v).all()
            for v in (proposed, parent, previous)
        )
    ):
        raise ValueError("finite explicit partial command and actual previous torque required")
    from rosclaw.simforge.g1_neural_torque import G1TorqueSafetyProjector

    projector = G1TorqueSafetyProjector(
        replace(
            safety, maximum_parent_deviation_ratio=fraction * safety.maximum_parent_deviation_ratio
        )
    )
    mixed = parent + fraction * (proposed - parent)
    if np.array_equal(mixed, parent):
        return projector.fallback(parent=parent, frame=frame, reason="postcompose_zero_parent")
    return projector.project(mixed, parent=parent, previous=previous, frame=frame)
