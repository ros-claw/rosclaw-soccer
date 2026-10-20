"""Opt-in SIM foot-strike intent: measure every other robot body against ball.

Not enabled for legacy training or goalkeepers/header skills. This new sensor
and reward contract MUST independently qualify before it is used for learning.
"""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from rosclaw_soccer.rsi.identity_kick_credit import IdentityKickCredit
from scripts.rsi_mjlab_ball_aware_backend import (
    BallAwareMotionEnv,
    actual_foot_force_history,
    make_ball_aware_training_cfg,
)


def actual_nonfoot_force_history(env):
    history = env.scene.sensors["ball_nonfoot"].data.force_history
    if (
        history is None
        or history.ndim != 4
        or history.shape[0] != env.num_envs
        or not 1 <= history.shape[1] <= 64
        or history.shape[2:] != (10, 3)
    ):
        raise ValueError("all nonfoot bodies and ten actual substeps required")
    return history.norm(dim=-1).amax(dim=1).flip(1)


def foot_only_kick_reward(env, *, part):
    motion = env.command_manager.get_term("motion")
    if not hasattr(env, "_identity_kick_credit"):
        env._identity_kick_credit = IdentityKickCredit(env.num_envs, env.device)
    ball = env.scene["ball"]
    return env._identity_kick_credit.step(
        step=env.common_step_counter,
        forces=actual_foot_force_history(env),
        expected=motion.kick_leg,
        ball_velocity=env.sim.data.qvel[:, ball.data.indexing.free_joint_v_adr[:3]],
        goal_delta=motion.target_destination_pos
        + env.scene.env_origins
        - ball.data.root_link_pos_w,
        forbidden_forces=actual_nonfoot_force_history(env),
    )[part]


class FootOnlyMotionEnv(BallAwareMotionEnv):
    def _physical_snapshot(self):
        value = super()._physical_snapshot()
        value["nonfoot_force_history"] = actual_nonfoot_force_history(self).clone()
        return value


def make_foot_only_training_cfg(**kwargs):
    cfg = make_ball_aware_training_cfg(football_rewards=True, **kwargs)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name="ball_nonfoot",
            primary=ContactMatch(
                mode="body",
                pattern=".*",
                entity="robot",
                exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
            ),
            secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
            secondary_policy="error",
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=10,
        ),
    )
    for part in ("contact", "wrong_contact", "alignment", "speed"):
        cfg.rewards["identity_" + part].func = foot_only_kick_reward
    # Once-per-episode -300 exceeds the maximum +250 kick-specific sum:
    # +50 contact + five controls * (+30 alignment +10 speed). dt is shared.
    cfg.rewards["identity_forbidden_contact"] = RewardTermCfg(
        func=foot_only_kick_reward, weight=-300.0, params=dict(part="forbidden_contact")
    )
    return cfg
