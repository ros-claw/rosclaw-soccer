"""Construct one new SIM scratch world from a qualified historical boundary.

This is not a reset API for an active world. The caller must separately bind
the integration state and prior targets to independently audited evidence.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld


def restored_scratch_world(
    model: Any,
    integration_state: np.ndarray,
    *,
    players: Mapping[str, NativePDParameters],
    ball_body_name: str,
    control: int,
    prior_targets: Mapping[str, np.ndarray],
) -> SharedNativeMotorWorld:
    """Initialize a new private native clock; never mutate a pre-existing world."""
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    if (
        type(control) is not int
        or not 1 <= control < 300
        or not isinstance(integration_state, np.ndarray)
        or integration_state.dtype != np.float64
        or integration_state.shape != (mujoco.mj_stateSize(model, spec),)
        or not np.isfinite(integration_state).all()
        or set(prior_targets) != set(players)
        or any(
            not isinstance(x, np.ndarray)
            or x.shape != (29,)
            or x.dtype != np.float64
            or not np.isfinite(x).all()
            for x in prior_targets.values()
        )
    ):
        raise ValueError("bounded finite SIM integration state and explicit prior targets required")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, integration_state.copy(), spec)
    if not np.isclose(data.time, control * 0.02, rtol=0, atol=1e-10):
        raise ValueError("physical time must match the exact declared control boundary")
    mujoco.mj_forward(model, data)
    world = SharedNativeMotorWorld(model, data, players=players, ball_body_name=ball_body_name)
    world.native_steps = 10 * control
    world.control_steps = control
    world._last_targets = {key: value.copy() for key, value in prior_targets.items()}
    return world
