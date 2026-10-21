# SPDX-License-Identifier: Apache-2.0
# Adapted scheduling: xiaojxkevin/G1-mjlab-soccer at 45e3d1e89ee924a263ff08d810fda5473dc3979d.
# Modified for explicit episode boundaries, fixed codec, observations and ball/contact contracts.
# See third_party/NOTICE.md and third_party/licenses/g1-mjlab-soccer-APACHE-2.0.txt.
"""Optional MJLab1.2 backend: explicit reset boundaries and fixed action codec.

Import ONLY in the separately qualified training environment, after pinning the
external reference source. No deployment/SDK code is imported. Simulator-only.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch
from mjlab.utils.lab_api.math import quat_apply_inverse, sample_uniform
from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage1_env_cfg
from src.tasks.soccer.mdp.shooter_commands import (
    MultiMotionSoccerCommand,
    MultiMotionSoccerCommandCfg,
)
from src.tasks.soccer.mdp.shooter_obs import motion_anchor_ori_b, motion_anchor_pos_b

from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_GLOB, motion_names_for_pattern


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
        defaults = self.robot.data.default_joint_pos
        if defaults is None:
            raise ValueError("canonical action defaults required")
        action._offset[env_ids] = defaults[env_ids]

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


class BalancedInitialMotionCommand(ExplicitBoundaryMotionCommand):
    """Balance sources at the ordinary episode reset, never inside a step."""

    def _uniform_sampling(self, env_ids):
        self.motion_idx[env_ids] = env_ids % self.motion.num_files
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
        self.time_steps[env_ids] = 0


class BodyBoundBallPlacementMixin:
    """New reset protocol: subset anchor index may only index the subset view.

    Legacy upstream code uses this index on full30 data, where torso subset7
    selects right hip pitch full7 instead of torso full15. Never silently change
    old experiment initial conditions; this implementation is explicit opt-in.
    """

    def _compute_soccer_ball_positions(self, env_ids):
        if self.cfg.fixed_ball_pos is not None:
            return super()._compute_soccer_ball_positions(env_ids)
        index = self.motion_anchor_body_index
        full_index = int(self.motion._body_indexes[index])
        if self.robot.body_names[full_index] != self.cfg.anchor_body_name:
            raise ValueError("motion subset anchor is not bound to the named physical body")
        n = len(env_ids)
        source = self.motion_idx[env_ids]
        last = (self.motion_length[env_ids] - 1).clamp(min=0)
        # Correct VIEW: index is a cfg.body_names subset index, not a full index.
        positions = self.motion.body_pos_w
        first_anchor = positions[
            source, torch.zeros(n, dtype=torch.long, device=self.device), index
        ]
        last_anchor = positions[source, last, index]
        vector = last_anchor[:, :2] - first_anchor[:, :2]
        square = torch.sum(vector * vector, dim=-1)
        radius = torch.sqrt(square.clamp(min=1e-12)) + self.curve_radius_offset[env_ids]
        valid = square > 1e-12
        direction = torch.where(
            valid.unsqueeze(-1),
            vector / torch.sqrt(square.clamp(min=1e-12)).unsqueeze(-1),
            torch.tensor([1.0, 0.0], device=self.device).expand(n, -1),
        )
        arc = float(self._target_arc_angle)
        if arc > 0.0:
            angle = torch.atan2(vector[:, 1], vector[:, 0])
            offset = sample_uniform(-arc, arc, (n,), device=self.device)
            angle = torch.where(valid, angle + offset, angle)
            direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        self.soccer_ball_pos[env_ids, :2] = first_anchor[:, :2] + radius[:, None] * direction
        self.soccer_ball_pos[env_ids, 2] = 0.11


class BodyBoundMotionCommand(BodyBoundBallPlacementMixin, ExplicitBoundaryMotionCommand):
    pass


class BodyBoundBalancedMotionCommand(BodyBoundBallPlacementMixin, BalancedInitialMotionCommand):
    pass


@dataclass(kw_only=True)
class BodyBoundMotionCfg(ExplicitBoundaryMotionCfg):
    def build(self, env):
        return BodyBoundMotionCommand(self, env)


@dataclass(kw_only=True)
class BodyBoundBalancedMotionCfg(ExplicitBoundaryMotionCfg):
    def build(self, env):
        return BodyBoundBalancedMotionCommand(self, env)


@dataclass(kw_only=True)
class BalancedInitialMotionCfg(ExplicitBoundaryMotionCfg):
    def build(self, env):
        return BalancedInitialMotionCommand(self, env)


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


class ExplicitBoundaryMotionEnv(ManagerBasedRlEnv):
    def reset(self, **kwargs):
        _, extras = super().reset(**kwargs)
        # Template reset computes relative caches before the simulator forward.
        # Refresh caches AFTER the base reset's forward, without touching bodies
        # or adding a second history sample. Per-step reset is followed by the
        # normal command-manager compute, which already refreshes these caches.
        self.command_manager.get_term("motion")._compute_relative_transforms()
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, extras


def make_motion_training_cfg(
    *,
    motion_dir,
    num_envs,
    seed,
    motion_glob=TRAIN_MOTION_GLOB,
    balanced_initial_sources=False,
    ball_placement_protocol="legacy-subset-full-v0",
):
    if (
        type(num_envs) is not int
        or not 1 <= num_envs <= 4096
        or type(seed) is not int
        or not 0 <= seed < 2**31
    ):
        raise ValueError("bounded explicit simulation batch and seed required")
    expected = motion_names_for_pattern(motion_glob)
    if ball_placement_protocol not in ("legacy-subset-full-v0", "body-bound-v1"):
        raise ValueError("explicit legacy or named-body ball placement protocol required")
    if type(balanced_initial_sources) is not bool or (
        balanced_initial_sources and (motion_glob != TRAIN_MOTION_GLOB or num_envs % 8 != 0)
    ):
        raise ValueError("balanced reset requires an exact multiple of all eight training sources")
    if tuple(p.name for p in sorted(Path(motion_dir).glob(motion_glob))) != expected:
        raise ValueError("exact declared source split required before simulator allocation")
    cfg = unitree_g1_stage1_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    original = cfg.commands["motion"]
    if not isinstance(original, MultiMotionSoccerCommandCfg):
        raise ValueError("exact qualified external motion command configuration required")
    values = {f.name: getattr(original, f.name) for f in fields(original)}
    command_type = (
        BalancedInitialMotionCfg if balanced_initial_sources else ExplicitBoundaryMotionCfg
    )
    if ball_placement_protocol == "body-bound-v1":
        command_type = (
            BodyBoundBalancedMotionCfg if balanced_initial_sources else BodyBoundMotionCfg
        )
    command = command_type(**values)
    command.motion_dir = str(motion_dir)
    command.motion_glob = motion_glob
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
    cfg.observations["critic"].terms["reference_anchor_linear_velocity_body"] = ObservationTermCfg(
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
