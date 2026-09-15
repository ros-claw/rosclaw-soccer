"""Bounded training credit with non-compensable qualification strata.

Numerical learning helpers, not hardware guards or candidate promotion. The
caller owns truthful physical/contact labels and a maximum 400-frame episode.
No robot model, simulator, football coordinate or policy framework is imported.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def joint_margin_penalty(margins_rad: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return 0..-0.05 per frame as the tightest measured margin approaches 0.

    This is a learning signal, not a stopping-distance guarantee. Signed
    margins must come from actual joint positions and unmodified joint ranges.
    At most 400 frames means this component cannot total less than -20.
    """
    if (
        not isinstance(margins_rad, np.ndarray)
        or margins_rad.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or margins_rad.ndim != 2
        or not 1 <= margins_rad.shape[0] <= 400
        or not 1 <= margins_rad.shape[1] <= 256
        or not np.isfinite(margins_rad).all()
        or (np.abs(margins_rad) > 100).any()
    ):
        raise ValueError("bounded finite episode of measured joint margins required")
    tightest = margins_rad.astype(np.float64).min(axis=1)
    result: NDArray[np.float32] = np.asarray(
        -0.05 * np.clip((0.08 - tightest) / 0.08, 0.0, 1.0), dtype=np.float32
    )
    return result


def guarded_terminal_credit(
    *, task_score: float, physically_safe: bool, contact_qualified: bool
) -> float:
    """Keep failed physics below failed contact below qualified task rewards.

    Combine ONLY with the bounded joint-margin penalty above to retain these
    ordered total-return intervals: [-120,-100], [-80,-60], [-40,10]. Adding
    unrelated rewards invalidates the guarantee. This does not relax the
    separate physical qualification or promotion criteria.
    """
    if (
        type(task_score) not in (float, int)
        or not math.isfinite(task_score)
        or type(physically_safe) is not bool
        or type(contact_qualified) is not bool
    ):
        raise ValueError("finite task score and explicit qualification booleans required")
    if not physically_safe:
        return -100.0
    if not contact_qualified:
        return -60.0
    return float(np.clip(task_score, -20.0, 10.0))
