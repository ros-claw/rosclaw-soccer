"""Compare ordered teammate contacts only within matched real exposure."""

import numpy as np

from rosclaw_soccer.rsi.relay_teammate_successor import teammate_successor


def common_successor(
    history: dict, *, common: int, return_start: int, layout: dict, timestep: float
) -> dict:
    """Describe a return chain without extrapolating or claiming possession."""
    if (
        type(common) is not int
        or type(return_start) is not int
        or common < 1
        or return_start < 0
        or any(
            type(history.get(k)) is not np.ndarray or len(history[k]) < common
            for k in ("qpos", "qvel", "ball_forces")
        )
    ):
        raise ValueError("aligned physical history and bounded common exposure required")
    if return_start >= common:
        return {"status": "RETURN_START_RIGHT_CENSORED", "samples": []}
    return teammate_successor(
        history["qpos"][:common],
        history["qvel"][:common],
        history["ball_forces"][:common],
        return_start=return_start,
        timestep=timestep,
        **layout,
    )
