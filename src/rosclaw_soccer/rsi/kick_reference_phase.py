"""Measured joint-state admission to an existing learned motion reference.

This is a bounded nearest-state selector, not learned weights or ball control.
It cannot manufacture earlier history, remove body momentum or prove contact.
"""

from typing import Any

import numpy as np


def nearest_kick_reference_phase(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    reference_position_hg: np.ndarray,
    reference_velocity_hg: np.ndarray,
) -> dict[str, Any]:
    """Fixed 0..250 search, squared pose plus one-control velocity displacement.

    Reference arrays must already use the qualified HG29 permutation. There is
    no outcome feedback, timing sweep, future ball input or implicit clipping.
    """
    q, v, rp, rv = map(
        np.asarray,
        (joint_position, joint_velocity, reference_position_hg, reference_velocity_hg),
    )
    if (
        q.shape != (29,)
        or v.shape != (29,)
        or rp.ndim != 2
        or rp.shape[1:] != (29,)
        or not 251 <= rp.shape[0] <= 10000
        or rv.shape != rp.shape
        or any(x.dtype.kind != "f" or not np.isfinite(x).all() for x in (q, v, rp, rv))
    ):
        raise ValueError("finite HG29 actual state and 251..10000 reference frames required")
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.mean((rp[:251] - q) ** 2 + (0.02 * (rv[:251] - v)) ** 2, axis=1)
    if not np.isfinite(cost).all():
        raise ValueError("finite measured phase costs required")
    frame = int(np.argmin(cost))
    return dict(
        schema="soccer.kick.measured_reference_phase.v1",
        reference_frame=frame,
        selected_cost_rad2=float(cost[frame]),
        frame_zero_cost_rad2=float(cost[0]),
        phase_costs_rad2=cost.tolist(),
        search_first=0,
        search_last=250,
        velocity_weight_seconds=0.02,
        first_tie_selected=True,
        optimization_uses_ball_outcome=False,
        neural_training=False,
        stability_proven=False,
    )
