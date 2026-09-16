"""Opt-in proposal for a frozen keeper foundation's arm-input ablation.

This is an experimental observation-composition gate, not a learned policy,
health assessment, physical state mutation, or motor authorization. The caller
owns per-player scope, sensor restoration and the existing execution guards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

RELEASE_RESIDUAL_EPSILON_RAD = 1.0e-6


def release_only_arm_observation_gate(
    *,
    reach_active: bool,
    previous_residual_rad: Sequence[float] | NDArray[Any],
) -> bool:
    """Propose arm-input isolation only while an arm-only reach is releasing.

    Inputs must describe this keeper's preceding executed control frame in the
    canonical 29-joint DDS order. The first 15 leg/waist entries must be zero;
    whole-body GMT residuals are outside this ablation's evidence contract.
    The function is stateless and never mutates its input. A caller must NOT
    feed a future residual, another player's history or a proposed action.

    True does not mean stable, a successful save, or permission to move. Invalid
    input raises ValueError, including during active reach; callers must retain
    their established fault/fallback path rather than silently enabling this.
    """
    residual = np.asarray(previous_residual_rad)
    if (
        type(reach_active) is not bool
        or residual.shape != (29,)
        or residual.dtype.kind not in "fiu"
        or not np.isfinite(residual).all()
        or np.any(np.abs(residual.astype(np.float64)) > 10.0)
        or np.any(residual[:15] != 0)
    ):
        raise ValueError(
            "finite bounded canonical arm-only history and boolean reach state required"
        )
    return not reach_active and bool(np.max(np.abs(residual)) > RELEASE_RESIDUAL_EPSILON_RAD)
