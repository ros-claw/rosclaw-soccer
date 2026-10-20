"""Optional MJLab1.2 backend: explicit reset boundaries and fixed action codec.

Import ONLY in the separately qualified training environment, after pinning the
external reference source. No deployment/SDK code is imported. Simulator-only.
"""

from dataclasses import dataclass, fields, replace

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch
from mjlab.utils.lab_api.math import quat_apply_inverse
from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage1_env_cfg
from src.tasks.soccer.mdp.shooter_commands import (
    MultiMotionSoccerCommand,
    MultiMotionSoccerCommandCfg,
)
from src.tasks.soccer.mdp.shooter_obs import motion_anchor_ori_b, motion_anchor_pos_b


class ExplicitBoundaryMotionCommand(MultiMotionSoccerCommand):
    """Motion exhaustion is a terminal boundary, never an in-step teleport."""

    def _compute_soccer_ball_positions(self, env_ids):
        super()._compute_soccer_ball_positions(env_ids)
        # The template's target-height setting describes its smaller ball.
        # Keep our declared .11m sphere on the floor at episode resets only.
        self.soccer_ball_pos[env_ids, 2] = 0.11

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if env_ids.numel() == 0:
            return
        action = self._env.action_manager._terms["joint_pos"]
        if not hasattr(action, "_offset"):
            raise ValueError("fixed default-offset action codec required")
        # Undo the template's sampled-pose action offset. Future inference must
        # use the SAME canonical offset as training, not a hidden reset pose.
        action._offset[env_ids] = self.robot.data.default_joint_pos[env_ids]

    def _update_command(self):
        self.kick_contact_tracker.begin_step()
        # End-of-reference termination is checked by the manager on the next
        # tick; hold the last REFERENCE only. No physical state is overwritten.
        self.time_steps.copy_(torch.minimum(self.time_steps + 1, self.motion_length - 1))
        self._update_target_points_from_sim()
        awarded = self.kick_contact_tracker.get_contact_awarded()
        no_contact = ~awarded
        if torch.any(no_contact):
            self.initial_target_point_pos[no_contact] = self.target_point_pos[no_contact]
        self._compute_relative_transforms()
        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class ExplicitBoundaryMotionCfg(MultiMotionSoccerCommandCfg):
    def build(self, env):
        return ExplicitBoundaryMotionCommand(self, env)


def reference_exhausted(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ExplicitBoundaryMotionCommand):
        raise ValueError("explicit-boundary motion command required")
    return command.time_steps >= command.motion_length - 1


def reference_anchor_linear_velocity_body(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return quat_apply_inverse(command.robot_anchor_quat_w, command.anchor_lin_vel_w)


def standard_ball_spec():
    # Native dimensions are still re-inspected after compilation. No source
    # mesh/weight/data license or entire-world comparability is implied.
    return mujoco.MjSpec.from_string(
        '<mujoco model="soccer_motion_ball"><worldbody><body name="ball">'
        '<freejoint/><inertial pos="0 0 0" mass="0.43" '
        'diaginertia="0.0020812 0.0020812 0.0020812"/>'
        '<geom name="ball_geom" type="sphere" size="0.11" '
        'friction="0.5 0.005 0.001" condim="6" rgba="1 1 1 1"/>'
        "</body></worldbody></mujoco>"
    )


def make_motion_training_cfg(*, motion_dir, num_envs, seed):
    if (
        type(num_envs) is not int
        or not 1 <= num_envs <= 4096
        or type(seed) is not int
        or not 0 <= seed < 2**31
    ):
        raise ValueError("bounded explicit simulation batch and seed required")
    cfg = unitree_g1_stage1_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    original = cfg.commands["motion"]
    if not isinstance(original, MultiMotionSoccerCommandCfg):
        raise ValueError("exact qualified external motion command configuration required")
    values = {f.name: getattr(original, f.name) for f in fields(original)}
    command = ExplicitBoundaryMotionCfg(**values)
    command.motion_dir = str(motion_dir)
    command.motion_origin_offset = (0.0, 0.0, 0.0)
    command.motion_yaw_offset = 0.0
    cfg.commands["motion"] = command
    cfg.terminations["reference_exhausted"] = TerminationTermCfg(
        func=reference_exhausted, time_out=True, params={"command_name": "motion"}
    )
    actor = cfg.observations["actor"]
    # These are declared reference commands and CURRENT measured geometry, not
    # privileged future simulator outcomes. This is a NEW neural ABI trained
    # from scratch; never add these channels to the old 1247 SONIC encoder.
    actor.terms["reference_anchor_position_body"] = ObservationTermCfg(
        func=motion_anchor_pos_b, params={"command_name": "motion"}
    )
    actor.terms["reference_anchor_orientation_body"] = ObservationTermCfg(
        func=motion_anchor_ori_b, params={"command_name": "motion"}
    )
    actor.terms["reference_anchor_linear_velocity_body"] = ObservationTermCfg(
        func=reference_anchor_linear_velocity_body, params={"command_name": "motion"}
    )
    cfg.scene.entities["ball"] = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(3.0, 3.0, 0.13)), spec_fn=standard_ball_spec
    )
    # The reference template's torso subtree excludes both legs. Match the
    # whole robot root so subsequent contact labels cannot silently omit feet.
    cfg.scene.sensors = tuple(
        replace(s, secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"))
        if s.name == "ball_robot_contact"
        else s
        for s in cfg.scene.sensors
    )
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    return cfg
