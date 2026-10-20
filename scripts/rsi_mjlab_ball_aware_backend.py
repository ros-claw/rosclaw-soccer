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
    def step(self, action):
        self.terminal_capture = None
        self._capture_terminal = True
        try:
            return super().step(action)
        finally:
            self._capture_terminal = False

    def _physical_snapshot(self):
        return dict(
            qpos_after=self.sim.data.qpos.clone(),
            qvel_after=self.sim.data.qvel.clone(),
            foot_force_history=actual_foot_force_history(self).clone(),
            processed_joint_target=self.action_manager.get_term(
                "joint_pos"
            )._processed_actions.clone(),
            position_target_after_step=self.command_manager.get_term(
                "motion"
            ).robot.data.joint_pos_target.clone(),
            actuator_force_after=self.sim.data.actuator_force.clone(),
            reference_exhausted=self.termination_manager.get_term("reference_exhausted").clone(),
        )

    def physical_step_data(self):
        """Read-only actual final state, including BEFORE-reset terminal worlds."""
        result = self._physical_snapshot()
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminal = getattr(self, "terminal_capture", None)
        if terminal is not None:
            ids = terminal["world_ids"]
            for key in result:
                result[key][ids] = terminal[key]
            mask[ids] = True
        result["terminal_capture_mask"] = mask
        return result

    def _reset_idx(self, env_ids=None):
        if getattr(self, "_capture_terminal", False):
            ids = (
                env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
            )
            if getattr(self, "terminal_capture", None) is not None:
                raise ValueError("one explicit terminal reset batch per physical control required")
            self.terminal_capture = dict(
                world_ids=ids.clone(),
                **{key: value[ids].clone() for key, value in self._physical_snapshot().items()},
            )
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
