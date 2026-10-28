"""Restore one SIM shadow's own recurrent state at a qualified boundary.

Caller must independently bind the supplied history to the same player,
model, applied targets and audited trajectory; no new hardware authority.
"""

import numpy as np
import torch

from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor


def restored_shadow_motor(
    world,
    prefix,
    actor,
    *,
    clip,
    torso_index,
    build_observation,
    goal,
    offset,
    scale,
    order,
    previous_action: np.ndarray,
    hidden: np.ndarray,
    cell: np.ndarray,
    observed_controls: int,
) -> SharedShadowMotor:
    if (
        type(observed_controls) is not int
        or observed_controls < 1
        or observed_controls != world.control_steps
        or any(
            type(x) is not np.ndarray or x.dtype != np.float32 or not np.isfinite(x).all()
            for x in (previous_action, hidden, cell)
        )
        or previous_action.shape != (29,)
        or hidden.shape != (2, 1, 256)
        or cell.shape != (2, 1, 256)
    ):
        raise ValueError("own finite recurrent history at exact physical boundary required")
    shadow = SharedShadowMotor(
        world,
        prefix,
        actor,
        clip=clip,
        torso_index=torso_index,
        build_observation=build_observation,
        goal=goal,
        offset=offset,
        scale=scale,
        order=order,
    )
    if not np.array_equal(shadow.previous_action, previous_action):
        raise ValueError("shadow history must encode this body's actual last target")
    shadow.state = (torch.from_numpy(hidden.copy()), torch.from_numpy(cell.copy()))
    shadow.observations = observed_controls
    return shadow
