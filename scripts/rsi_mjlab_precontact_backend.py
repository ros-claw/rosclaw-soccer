"""Opt-in measured-state contact curriculum. SIM only, unqualified on import.

Restoration happens ONLY at ordinary episode boundaries. No body/ball state
overwrite during active motion. Legacy environment behavior is untouched.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.termination_manager import TerminationTermCfg

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_mjlab_foot_only_backend import FootOnlyMotionEnv, make_foot_only_training_cfg
from scripts.rsi_mjlab_motion_backend import (
    BodyBoundBalancedMotionCfg,
    BodyBoundBalancedMotionCommand,
)


class PrecontactMotionCommand(BodyBoundBalancedMotionCommand):
    def _update_command(self):
        mask = self._env._course_just_reset.clone()
        previous = self.time_steps[mask].clone()
        super()._update_command()
        if mask.any():
            # ManagerBasedRlEnv updates commands AFTER an in-step reset. Its
            # first observation must still represent the restored source frame.
            self.time_steps[mask] = previous
            self._env._restore_initial_solver_seed(torch.nonzero(mask).flatten())
            self._env._course_just_reset[mask] = False
            self._compute_relative_transforms()


@dataclass(kw_only=True)
class PrecontactMotionCfg(BodyBoundBalancedMotionCfg):
    def build(self, env):
        return PrecontactMotionCommand(self, env)


def curriculum_window_exhausted(env):
    return env.episode_length_buf - env._course_start_frame >= 96


def make_precontact_training_cfg(**kwargs):
    if kwargs.get("ball_placement_protocol") != "body-bound-v1":
        raise ValueError("precontact course requires explicit named-body placement protocol")
    if kwargs.get("balanced_initial_sources") is not True:
        raise ValueError("explicit balanced TRAIN8 source course required")
    cfg = make_foot_only_training_cfg(**kwargs)
    original = cfg.commands["motion"]
    values = {f.name: getattr(original, f.name) for f in fields(original)}
    values["resampling_time_range"] = (1e9, 1e9)
    cfg.commands["motion"] = PrecontactMotionCfg(**values)
    # Initial nominal curriculum matches its native source: no scheduled push.
    # Disturbance robustness remains a separate later exam, not inherited here.
    if set(cfg.events) != {"push_robot"}:
        raise ValueError("review changed external event template before course construction")
    cfg.events = {}
    cfg.terminations["curriculum_window"] = TerminationTermCfg(
        func=curriculum_window_exhausted, time_out=True
    )
    return cfg


class PrecontactMotionEnv(FootOnlyMotionEnv):
    def __init__(
        self,
        *args,
        reset_bank: Path,
        reset_bank_hash: str,
        lead_controls: int | None = None,
        **kwargs,
    ):
        if file_hash(reset_bank) != reset_bank_hash:
            raise ValueError("immutable measured TRAIN reset bank required")
        with np.load(reset_bank, allow_pickle=False) as raw:
            bank = {k: raw[k].copy() for k in raw.files}
        expected = {
            "qpos": (256, 43),
            "qvel": (256, 41),
            "qacc_warmstart": (256, 41),
            "previous_action": (256, 29),
            "previous_previous_action": (256, 29),
            "observation": (256, 177),
            "origin": (256, 3),
            "goal_world": (256, 3),
        }
        for key, shape in expected.items():
            if bank[key].shape != shape or not np.isfinite(bank[key]).all():
                raise ValueError("complete finite measured state bank required")
        if not np.array_equal(bank["source_id"], bank["world"] % 8):
            raise ValueError("TRAIN source identity changed")
        super().__init__(*args, **kwargs)
        self._course_bank = {
            k: torch.as_tensor(
                v, device=self.device, dtype=torch.float32 if v.dtype.kind == "f" else torch.long
            )
            for k, v in bank.items()
        }
        self._course_layout = NativeVelocityLayout(
            self.sim.mj_model, robot_prefix="robot/", ball_body_name="ball/ball"
        )
        self._course_reset_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._course_just_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._course_start_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.course_bank_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.set_curriculum_lead(lead_controls)

    def set_curriculum_lead(self, lead: int | None):
        if lead is not None and (type(lead) is not int or lead not in (10, 20, 40, 80)):
            raise ValueError("declared observed lead10/20/40/80 or complete bank required")
        self.course_lead = lead
        self._course_pool = (
            torch.arange(256, device=self.device)
            if lead is None
            else torch.nonzero(self._course_bank["lead_controls"] == lead).flatten()
        )
        if len(self._course_pool) != (256 if lead is None else 64):
            raise ValueError("complete balanced curriculum pool required")

    def _reset_idx(self, env_ids=None):
        ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        if not len(ids):
            return
        super()._reset_idx(ids)
        bank = self._course_bank
        layout = self._course_layout
        index = self._course_pool[(ids + self._course_reset_count[ids]) % len(self._course_pool)]
        self._course_reset_count[ids] += 1
        self.course_bank_index[ids] = index
        origin = self.scene.env_origins[ids]
        shift = origin - bank["origin"][index]
        q = bank["qpos"][index].clone()
        q[:, layout.rq : layout.rq + 3] += shift
        q[:, layout.bq : layout.bq + 3] += shift
        self.sim.data.qpos[ids] = q
        self.sim.data.qvel[ids] = bank["qvel"][index]
        self.sim.data.qacc_warmstart[ids] = bank["qacc_warmstart"][index]
        frame = bank["reference_frame"][index]
        self.sim.data.time[ids] = frame.float() * 0.02
        motion = self.command_manager.get_term("motion")
        motion.motion_idx[ids] = bank["source_id"][index]
        motion.motion_length[ids] = motion.motion.file_lengths[motion.motion_idx[ids]]
        motion.time_steps[ids] = frame
        motion.soccer_ball_pos[ids] = q[:, layout.bq : layout.bq + 3] - origin
        motion.target_point_pos[ids] = motion.soccer_ball_pos[ids]
        motion.initial_target_point_pos[ids] = motion.soccer_ball_pos[ids]
        motion.target_destination_pos[ids] = bank["goal_world"][index] - bank["origin"][index]
        if not torch.equal(motion.kick_leg[ids].long(), bank["expected_foot"][index]):
            raise ValueError("restored role foot differs from source label")
        self.episode_length_buf[ids] = frame
        self._course_start_frame[ids] = frame
        self._course_just_reset[ids] = True
        manager = self.action_manager
        term = manager.get_term("joint_pos")
        if tuple(term.target_names) != G1_DDS_JOINT_NAMES or term.cfg.clip is not None:
            raise ValueError("unclipped exact HG29 action codec required")
        if torch.any(motion.robot.data.encoder_bias[ids] != 0):
            raise ValueError("recorded native curriculum requires zero encoder bias")
        action = bank["previous_action"][index]
        manager.action[ids] = action
        manager.prev_action[ids] = bank["previous_previous_action"][index]
        manager.prev_prev_action[ids] = bank["previous_previous_action"][index]
        term._raw_actions[ids] = action
        term._processed_actions[ids] = action * term._scale[ids] + term._offset[ids]
        motion.robot.set_joint_position_target(
            term._processed_actions[ids], joint_ids=term.target_ids, env_ids=ids
        )

    def reset(self, **kwargs):
        _, extras = super().reset(**kwargs)
        ids = kwargs.get("env_ids")
        if ids is None:
            self._course_just_reset.zero_()
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            self._course_just_reset[ids] = False
        self._restore_initial_solver_seed(ids)
        self.command_manager.get_term("motion")._compute_relative_transforms()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, extras

    def _restore_initial_solver_seed(self, ids):
        self.sim.data.qacc_warmstart[ids] = self._course_bank["qacc_warmstart"][
            self.course_bank_index[ids]
        ]
