"""Measured native velocity98 + exogenous reference70 + measured ball/goal9.

No network, physics advancement, reward, or simulator state mutation. Callers
own the reference clock, previous action, and scalar float32 gait history.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, observation98
from rosclaw_soccer.rsi.ball_aware_inputs import ball_goal9
from rosclaw_soccer.rsi.native_motion_reference import point_velocity_command, reference_inputs70
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


def native_reference_observation177(
    *,
    data: Any,
    layout: NativeVelocityLayout,
    params: VelocityPolicyParameters,
    plan: dict[str, NDArray[Any]],
    frame: int,
    origin: NDArray[np.float64],
    goal: NDArray[np.float64],
    last_action: NDArray[np.float32],
    phase: np.float32,
    torso: int,
) -> tuple[NDArray[np.float32], np.float32]:
    if type(frame) is not int or not 0 <= frame < len(plan["joint"]):
        raise ValueError("in-range explicit reference frame required")
    if origin.shape != (3,) or goal.shape != (3,) or not np.isfinite([origin, goal]).all():
        raise ValueError("finite actual origin and goal required")
    rotation = data.xmat[layout.pelvis].reshape(3, 3)
    _, ref = reference_inputs70(
        root_rotation=rotation,
        anchor_position=data.xpos[torso],
        anchor_rotation=data.xmat[torso].reshape(3, 3),
        joint_position=plan["joint"][frame],
        joint_velocity=plan["joint_velocity"][frame],
        reference_position=plan["torso_position"][frame] + origin,
        reference_quaternion=plan["torso_quaternion"][frame],
        reference_linear_velocity=plan["torso_linear"][frame],
        reference_angular_velocity=plan["torso_angular"][frame],
    )
    cmd = point_velocity_command(
        root_rotation=rotation,
        current_position=data.xpos[layout.pelvis],
        reference_position=plan["pelvis_position"][frame] + origin,
        reference_linear_velocity=plan["pelvis_linear"][frame],
        reference_angular_velocity=plan["pelvis_angular"][frame],
        position_gain=1.0,
    )
    sense = ball_goal9(
        pelvis_rotation=rotation,
        pelvis_position=data.xpos[layout.pelvis],
        pelvis_linear_velocity=data.qvel[layout.rv : layout.rv + 3],
        ball_position=data.xpos[layout.ball],
        ball_linear_velocity=data.qvel[layout.bv : layout.bv + 3],
        target_position=goal,
    )
    q, v = layout.canonical(data)
    obs, next_phase = observation98(
        q, v, layout.angular_velocity_body(data), cmd, last_action, phase, params
    )
    return np.concatenate((obs, ref, sense)), next_phase
