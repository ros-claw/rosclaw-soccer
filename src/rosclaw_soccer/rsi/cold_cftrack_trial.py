"""Fresh declared SIM episodes sharing only stateless NN and immutable models.

Never reset a running body or policy. A never-executed template remains cold;
each returned player owns a newly initialized physical world and private policy.
"""

import numpy as np

from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def validate_cold_ball_xy(ball_xy: tuple[float, float]) -> None:
    if (
        type(ball_xy) is not tuple
        or len(ball_xy) != 2
        or any(type(v) is not float or not np.isfinite(v) for v in ball_xy)
        or not 0.9 <= ball_xy[0] <= 1.1
        or abs(ball_xy[1]) > 0.1
    ):
        raise ValueError("explicit bounded float XY declaration required before new world clock")


def cold_cftrack_trial(
    template: SharedOmniContactPlayer, *, ball_xy: tuple[float, float]
) -> SharedOmniContactPlayer:
    import mujoco

    validate_cold_ball_xy(ball_xy)
    w = template.world
    w._require_current_boundary()
    if (
        type(template) is not SharedOmniContactPlayer
        or len(w.layouts) != 1
        or template.failed
        or template.pending is not None
        or template.motor_plan is not None
        or w.native_steps != 0
        or template.policy.counter_step != 0
        or template.reference_source != "CFgen"
        or np.any(template.policy.obs_history_buffer)
        or np.any(template.policy.action)
        or np.any(w.data.qvel)
    ):
        raise ValueError("never-executed zero-velocity private CFgen template required")
    before = cftrack_state_hash(template.policy)
    policy = copy_cftrack_policy(template.policy)
    data = mujoco.MjData(w.model)
    # Fresh solver state, not the template's forward/solver warmstart cache.
    data.qpos[:] = w.data.qpos
    data.qvel[:] = w.data.qvel
    data.qpos[template.layout.bq : template.layout.bq + 2] = ball_xy
    mujoco.mj_forward(w.model, data)
    world = SharedNativeMotorWorld(
        w.model, data, players=w.parameters, ball_body_name=w.model.body(w.ball).name
    )
    result = SharedOmniContactPlayer(
        world,
        template.prefix,
        policy,
        expected_policy_type=type(template.policy),
        goal=template.policy.goal_pos_override.copy(),
    )
    w._require_current_boundary()
    if cftrack_state_hash(template.policy) != before or w.native_steps != 0:
        raise ValueError("new episode altered its cold donor")
    return result
