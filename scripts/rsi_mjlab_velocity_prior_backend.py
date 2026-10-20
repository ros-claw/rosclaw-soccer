"""Optional SIM backend: qualified velocity98 prefix plus explicit reference70.

Import only after pinning the external MJLab reference and training stack.
No environment allocation, transport, weight loading or hardware on import.
"""

import numpy as np
import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from src.tasks.soccer.mdp.shooter_obs import motion_anchor_ori_b, motion_anchor_pos_b

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.velocity_prior_actor import phase_table
from scripts.rsi_mjlab_motion_backend import make_motion_training_cfg


def velocity_reference_observation(env, *, defaults, command_mode):
    motion = env.command_manager.get_term("motion")
    robot = motion.robot
    names = tuple(robot.joint_names)
    if len(names) != 29 or set(names) != set(G1_DDS_JOINT_NAMES):
        raise ValueError("explicit complete HG29 simulation articulation required")
    ids = [names.index(name) for name in G1_DDS_JOINT_NAMES]
    # The action term's raw vector must itself be HG order; do not silently
    # permute history separately from the actual executed action codec.
    action_term = env.action_manager.get_term("joint_pos")
    if tuple(action_term.target_names) != G1_DDS_JOINT_NAMES:
        raise ValueError("actual action order must match velocity prior HG29 output")
    quaternion = robot.data.root_link_quat_w
    if command_mode == "zero":
        command = torch.zeros((env.num_envs, 3), device=env.device)
    elif command_mode == "reference_velocity":
        linear = quat_apply_inverse(quaternion, motion.anchor_lin_vel_w)
        angular = quat_apply_inverse(quaternion, motion.anchor_ang_vel_w)
        # Exogenous velocity-command range, NOT actuator/state clipping.
        command = torch.stack(
            (
                linear[:, 0].clamp(-0.5, 1.0),
                linear[:, 1].clamp(-0.5, 0.5),
                angular[:, 2].clamp(-1, 1),
            ),
            dim=1,
        )
    else:
        raise ValueError("declared zero or reference_velocity command mode required")
    if not hasattr(env, "_velocity_prior_phase_table"):
        env._velocity_prior_phase_table = torch.from_numpy(phase_table()).to(env.device)
    frames = env.episode_length_buf
    if torch.any(frames < 0) or torch.any(frames >= len(env._velocity_prior_phase_table)):
        raise ValueError("explicit bounded episode phase required; no modulo reset")
    phase = env._velocity_prior_phase_table[frames].double() * (2 * np.pi)
    gait = torch.stack((phase.sin(), phase.cos()), dim=1).float()
    gait[torch.linalg.vector_norm(command, dim=1) < 0.1] = 0
    home = torch.tensor(defaults, dtype=torch.float32, device=env.device)
    prefix = torch.cat(
        (
            robot.data.root_link_ang_vel_b,
            robot.data.projected_gravity_b,
            command,
            gait,
            robot.data.joint_pos[:, ids] - home,
            robot.data.joint_vel[:, ids],
            env.action_manager.action,
        ),
        dim=1,
    )
    suffix = torch.cat(
        (
            motion.joint_pos[:, ids],
            motion.joint_vel[:, ids],
            motion_anchor_pos_b(env, "motion"),
            motion_anchor_ori_b(env, "motion"),
            quat_apply_inverse(motion.robot_anchor_quat_w, motion.anchor_lin_vel_w),
        ),
        dim=1,
    )
    if prefix.shape != (env.num_envs, 98) or suffix.shape != (env.num_envs, 70):
        raise ValueError("actual velocity98 and semantic reference70 required")
    return torch.cat((prefix, suffix), dim=1)


def make_velocity_prior_training_cfg(*, parameters, command_mode, **kwargs):
    if command_mode not in ("zero", "reference_velocity"):
        raise ValueError("explicit prior command mode required")
    cfg = make_motion_training_cfg(**kwargs)
    # The imported training template computes scales from effort/Kp, whereas
    # the qualified deployment codec stores rounded constants. This is an
    # explicit action-codec choice, not a native motor/gain modification.
    cfg.actions["joint_pos"].scale = dict(
        zip(G1_DDS_JOINT_NAMES, parameters.action_scale, strict=True)
    )
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "velocity98_reference70": ObservationTermCfg(
                func=velocity_reference_observation,
                params={"defaults": parameters.defaults, "command_mode": command_mode},
            )
        },
        enable_corruption=False,
        nan_policy="error",
    )
    # Verify these after allocation against actual processed actions. Do not
    # replace gains, limits, geometry or home pose in order to obtain a pass.
    return cfg
