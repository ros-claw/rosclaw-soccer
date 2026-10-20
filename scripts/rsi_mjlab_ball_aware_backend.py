"""Optional ball-aware177 SIM backend with identity-bound per-foot sensors.

No simulation allocation on import. Original velocity98/reference70, bodies,
motors and ball dynamics stay intact. New observation/reward ABI is explicit.
"""

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from rosclaw_soccer.rsi.identity_kick_credit import IdentityKickCredit
from scripts.rsi_mjlab_motion_backend import ExplicitBoundaryMotionEnv
from scripts.rsi_mjlab_velocity_prior_backend import make_velocity_prior_training_cfg


def ball_goal_observation(env):
    motion = env.command_manager.get_term("motion")
    robot, ball = motion.robot, env.scene["ball"]
    quaternion = robot.data.root_link_quat_w
    robot_v = env.sim.data.qvel[:, robot.data.indexing.free_joint_v_adr[:3]]
    ball_v = env.sim.data.qvel[:, ball.data.indexing.free_joint_v_adr[:3]]
    goal = motion.target_destination_pos + env.scene.env_origins
    return torch.cat(
        tuple(
            quat_apply_inverse(quaternion, value)
            for value in (
                ball.data.root_link_pos_w - robot.data.root_link_pos_w,
                ball_v - robot_v,
                goal - robot.data.root_link_pos_w,
            )
        ),
        dim=1,
    )


def actual_foot_force_history(env):
    values = []
    for side in ("left", "right"):
        data = env.scene.sensors["ball_" + side + "_foot"].data
        history = data.force_history
        if history is None or history.shape != (env.num_envs, 1, 10, 3):
            raise ValueError("complete ten-substep identity-bound foot contact history required")
        values.append(history[:, 0].norm(dim=-1).flip(1))
    return torch.stack(values, dim=-1)


def identity_kick_reward(env, *, part):
    motion = env.command_manager.get_term("motion")
    if not hasattr(env, "_identity_kick_credit"):
        env._identity_kick_credit = IdentityKickCredit(env.num_envs, env.device)
    ball = env.scene["ball"]
    velocity = env.sim.data.qvel[:, ball.data.indexing.free_joint_v_adr[:3]]
    values = env._identity_kick_credit.step(
        step=env.common_step_counter,
        forces=actual_foot_force_history(env),
        expected=motion.kick_leg,
        ball_velocity=velocity,
        goal_delta=motion.target_destination_pos
        + env.scene.env_origins
        - ball.data.root_link_pos_w,
    )
    return values[part]


def intended_foot_proximity(env):
    motion = env.command_manager.get_term("motion")
    robot, ball = motion.robot, env.scene["ball"]
    feet = [robot.body_names.index(side + "_ankle_roll_link") for side in ("left", "right")]
    positions = robot.data.body_link_pos_w[:, feet]
    chosen = positions[torch.arange(env.num_envs, device=env.device), motion.kick_leg.long()]
    distance = (chosen - ball.data.root_link_pos_w).square().sum(dim=1)
    # Shaping only: proximity is never evidence of a physical contact.
    return torch.exp(-distance / 0.3**2)


class BallAwareMotionEnv(ExplicitBoundaryMotionEnv):
    def _reset_idx(self, env_ids=None):
        super()._reset_idx(env_ids)
        credit = getattr(self, "_identity_kick_credit", None)
        if credit is not None:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            credit.reset(env_ids)


def make_ball_aware_training_cfg(*, football_rewards: bool = False, **kwargs):
    if type(football_rewards) is not bool:
        raise ValueError("explicit football reward protocol required")
    cfg = make_velocity_prior_training_cfg(**kwargs)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["current_ball_goal9"] = ObservationTermCfg(
            func=ball_goal_observation
        )
    # Each sensor is bound to ONE physical foot body, not all robot contacts.
    # Full ten-substep history prevents missing brief contacts between controls.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
        ContactSensorCfg(
            name="ball_" + side + "_foot",
            primary=ContactMatch(mode="body", pattern="ball", entity="ball"),
            secondary=ContactMatch(mode="body", pattern=side + "_ankle_roll_link", entity="robot"),
            secondary_policy="error",
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=10,
        )
        for side in ("left", "right")
    )
    if football_rewards:
        cfg.rewards["track_anchor_pos"].weight = 0.0
        cfg.rewards["intended_foot_proximity"] = RewardTermCfg(
            func=intended_foot_proximity, weight=1.0
        )
        for part, weight in (
            ("contact", 50.0),
            ("wrong_contact", -10.0),
            ("alignment", 30.0),
            ("speed", 10.0),
        ):
            cfg.rewards["identity_" + part] = RewardTermCfg(
                func=identity_kick_reward, weight=weight, params=dict(part=part)
            )
    return cfg
