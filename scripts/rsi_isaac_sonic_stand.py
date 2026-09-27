"""SIM_ONLY SONIC-to-Isaac standing integration probe, not policy qualification.

Uses qualified 29-DoF SONIC targets with their native PD gains on an isolated
Isaac Lab G1 asset. The ball is passive; no football contact is requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--g1-usd", required=True, type=Path)
parser.add_argument("--model-root", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--frames", type=int, default=150)
parser.add_argument("--agent-count", type=int, default=1)
parser.add_argument("--forward-command-m-s", type=float, default=0.0)
parser.add_argument("--experimental-high-speed", action="store_true")
parser.add_argument("--lateral-command-m-s", type=float, default=0.0)
parser.add_argument("--lateral-start-frame", type=int)
parser.add_argument("--lateral-end-frame", type=int)
parser.add_argument("--reactive-lateral-command-m-s", type=float, default=0.0)
parser.add_argument("--stop-at-frame", type=int)
parser.add_argument("--brake-command-m-s", type=float, default=0.0)
parser.add_argument("--brake-frames", type=int, default=0)
parser.add_argument("--brake-release-speed-m-s", type=float)
parser.add_argument("--brake-release-window-frames", type=int, default=1)
parser.add_argument("--brake-min-frames", type=int, default=0)
parser.add_argument("--hip-brake-gains")
parser.add_argument("--ankle-brake-gains")
parser.add_argument("--track-ball-contacts", action="store_true")
parser.add_argument("--ball-x-m", type=float, default=2.5)
parser.add_argument("--ball-y-m", type=float, default=0.0)
parser.add_argument("--ball-initial-vx-m-s", type=float, default=0.0)
parser.add_argument("--ball-initial-vy-m-s", type=float, default=0.0)
parser.add_argument("--ball-rolling-start", action="store_true")
parser.add_argument("--right-knee-contact-residual-rad", type=float, default=0.0)
parser.add_argument("--right-hip-pitch-contact-residual-rad", type=float, default=0.0)
parser.add_argument("--right-ankle-pitch-contact-residual-rad", type=float, default=0.0)
parser.add_argument("--right-foot-ik-forward-m", type=float, default=0.0)
parser.add_argument("--right-knee-clearance-m", type=float, default=0.0)
parser.add_argument("--contact-speed-actor", type=Path)
parser.add_argument("--control-mode", choices=("sonic", "frozen_target"), default="sonic")
parser.add_argument(
    "--actuator-mode",
    choices=("sonic_implicit", "official_without_hands"),
    default="sonic_implicit",
)
parser.add_argument(
    "--root-orientation",
    choices=("legacy_numeric_identity", "official_yaw90", "mujoco_aligned"),
    default="mujoco_aligned",
)
parser.add_argument(
    "--observation-root-frame", choices=("raw", "asset_aligned"), default="asset_aligned"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    hip_brake_gains = (
        [0.0] * args.agent_count
        if args.hip_brake_gains is None
        else [float(value) for value in args.hip_brake_gains.split(",")]
    )
    ankle_brake_gains = (
        [0.0] * args.agent_count
        if args.ankle_brake_gains is None
        else [float(value) for value in args.ankle_brake_gains.split(",")]
    )
except ValueError:
    parser.error("finite bounded comma-separated joint brake gains required")
if (
    not args.g1_usd.is_file()
    or args.g1_usd.stat().st_size < 1000
    or not args.model_root.is_dir()
    or not 50 <= args.frames <= 600
    or type(args.agent_count) is not int
    or not 1 <= args.agent_count <= 3
    or not math.isfinite(args.forward_command_m_s)
    or not -0.5 <= args.forward_command_m_s <= 1.5
    or (args.experimental_high_speed and not 0.7 < args.forward_command_m_s <= 1.5)
    or (args.forward_command_m_s > 0.5 and not args.experimental_high_speed)
    or not math.isfinite(args.lateral_command_m_s)
    or not -0.2 <= args.lateral_command_m_s <= 0.2
    or math.hypot(args.forward_command_m_s, args.lateral_command_m_s)
    > (1.5 if args.experimental_high_speed else 0.7)
    or ((args.lateral_start_frame is None) != (args.lateral_end_frame is None))
    or (
        args.lateral_start_frame is not None
        and not 0 <= args.lateral_start_frame < args.lateral_end_frame <= args.frames
    )
    or not math.isfinite(args.reactive_lateral_command_m_s)
    or not -0.2 <= args.reactive_lateral_command_m_s <= 0.2
    or (
        args.reactive_lateral_command_m_s != 0.0
        and (
            args.agent_count != 1
            or not args.track_ball_contacts
            or args.lateral_command_m_s != 0.0
            or args.lateral_start_frame is not None
        )
    )
    or (args.stop_at_frame is not None and not 1 <= args.stop_at_frame < args.frames)
    or not math.isfinite(args.brake_command_m_s)
    or not -0.5 <= args.brake_command_m_s <= 0.0
    or not 0 <= args.brake_frames <= 50
    or (args.stop_at_frame is None and args.brake_frames != 0)
    or (args.brake_frames == 0 and args.brake_command_m_s != 0.0)
    or (args.stop_at_frame is not None and args.stop_at_frame + args.brake_frames >= args.frames)
    or (
        args.brake_release_speed_m_s is not None
        and (
            not math.isfinite(args.brake_release_speed_m_s)
            or not 0.0 <= args.brake_release_speed_m_s <= 0.5
            or args.brake_frames == 0
        )
    )
    or not 1 <= args.brake_release_window_frames <= 20
    or not 0 <= args.brake_min_frames <= args.brake_frames
    or (
        args.brake_release_speed_m_s is None
        and (args.brake_release_window_frames != 1 or args.brake_min_frames != 0)
    )
    or len(hip_brake_gains) != args.agent_count
    or len(ankle_brake_gains) != args.agent_count
    or any(not math.isfinite(value) or abs(value) > 0.35 for value in hip_brake_gains)
    or any(not math.isfinite(value) or abs(value) > 0.35 for value in ankle_brake_gains)
    or (args.stop_at_frame is None and (any(hip_brake_gains) or any(ankle_brake_gains)))
    or (args.track_ball_contacts and args.agent_count != 1)
    or not math.isfinite(args.ball_x_m)
    or not 2.0 <= args.ball_x_m <= 3.0
    or not math.isfinite(args.ball_y_m)
    or not -0.4 <= args.ball_y_m <= 0.4
    or not math.isfinite(args.ball_initial_vx_m_s)
    or not math.isfinite(args.ball_initial_vy_m_s)
    or math.hypot(args.ball_initial_vx_m_s, args.ball_initial_vy_m_s) > 2.0
    or (
        args.ball_rolling_start
        and math.hypot(args.ball_initial_vx_m_s, args.ball_initial_vy_m_s) < 0.01
    )
    or not math.isfinite(args.right_knee_contact_residual_rad)
    or not -0.12 <= args.right_knee_contact_residual_rad <= 0.12
    or not math.isfinite(args.right_hip_pitch_contact_residual_rad)
    or not -0.12 <= args.right_hip_pitch_contact_residual_rad <= 0.12
    or not math.isfinite(args.right_ankle_pitch_contact_residual_rad)
    or not -0.12 <= args.right_ankle_pitch_contact_residual_rad <= 0.12
    or not math.isfinite(args.right_foot_ik_forward_m)
    or not 0.0 <= args.right_foot_ik_forward_m <= 0.15
    or not math.isfinite(args.right_knee_clearance_m)
    or not 0.0 <= args.right_knee_clearance_m <= 0.05
    or (args.right_knee_clearance_m != 0.0 and args.right_foot_ik_forward_m == 0.0)
    or (
        args.right_foot_ik_forward_m != 0.0
        and (
            args.agent_count != 1
            or not args.track_ball_contacts
            or args.contact_speed_actor is not None
        )
    )
    or (
        (
            args.right_knee_contact_residual_rad != 0.0
            or args.right_hip_pitch_contact_residual_rad != 0.0
            or args.right_ankle_pitch_contact_residual_rad != 0.0
        )
        and (args.agent_count != 1 or not args.track_ball_contacts)
    )
    or (
        args.contact_speed_actor is not None
        and (
            not args.contact_speed_actor.is_file()
            or args.agent_count != 1
            or not args.track_ball_contacts
            or args.stop_at_frame is not None
            or args.forward_command_m_s != 0.0
            or args.reactive_lateral_command_m_s != 0.08
            or args.right_knee_contact_residual_rad != 0.12
            or args.right_hip_pitch_contact_residual_rad != 0.0
            or args.right_ankle_pitch_contact_residual_rad != 0.0
        )
    )
    or args.output_dir.exists()
):
    parser.error("bounded frames, qualified local inputs, and a new output directory required")
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES  # noqa: E402
from rosclaw_soccer.providers.g1.sonic_navigation import (  # noqa: E402
    G1SonicNavigation,
    SonicNavigationConfig,
)
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController  # noqa: E402
from rosclaw_soccer.sim.ball_contact_evidence import classify_ball_body_contacts  # noqa: E402
from rosclaw_soccer.sim.ball_rolling_evidence import precontact_rolling_evidence  # noqa: E402
from rosclaw_soccer.sim.contact_speed_actor import choose_contact_speed  # noqa: E402
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json  # noqa: E402
from rosclaw_soccer.sim.foot_target_ik import bounded_foot_target_delta  # noqa: E402
from rosclaw_soccer.sim.isaac_root_bridge import isaac_root_to_mujoco  # noqa: E402
from rosclaw_soccer.sim.joint_target_projection import (  # noqa: E402
    project_modified_joint_targets,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation  # noqa: E402


def main() -> None:
    source_hash = hash_bytes(Path(__file__).read_bytes())
    asset_hash = hash_bytes(args.g1_usd.read_bytes())
    agent_ids = ("blue.playmaker", "blue.striker", "blue.goalkeeper")[: args.agent_count]
    navigations = [
        G1SonicNavigation(
            args.model_root,
            agent_id,
            SonicNavigationConfig(
                maximum_frames=args.frames,
                model_variant="low_latency",
                experimental_maximum_speed_mps=(1.5 if args.experimental_high_speed else None),
            ),
        )
        for agent_id in agent_ids
    ]
    for navigation in navigations:
        navigation.backend.qualification.require_eligible()
    qualification = navigations[0].backend.qualification
    names = tuple(G1_DDS_JOINT_NAMES)
    kp = np.asarray(navigations[0].backend.kp, dtype=np.float64)
    kd = np.asarray(navigations[0].backend.kd, dtype=np.float64)
    effort = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    if len(names) != 29 or kp.shape != (29,) or kd.shape != (29,) or effort.shape != (29,):
        raise ValueError("SONIC joint/gain/effort shape changed")
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=0.002))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    for index in range(args.agent_count):
        sim_utils.create_prim(f"/World/Env{index}", "Xform")
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/Env.*/G1"
    robot_cfg.spawn.usd_path = str(args.g1_usd.resolve())
    if args.actuator_mode == "sonic_implicit":
        robot_cfg.actuators = {
            "sonic": ImplicitActuatorCfg(
                joint_names_expr=list(names),
                effort_limit_sim={
                    name: float(value) for name, value in zip(names, effort, strict=True)
                },
                stiffness={name: float(value) for name, value in zip(names, kp, strict=True)},
                damping={name: float(value) for name, value in zip(names, kd, strict=True)},
                armature={name: 0.01 for name in names},
            )
        }
    else:
        robot_cfg.actuators.pop("hands")
    robot = Articulation(cfg=robot_cfg)
    ball = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/Env0/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.11,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=args.track_ball_contacts,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(args.ball_x_m, args.ball_y_m, 0.13)),
        )
    )
    contact_sensor = None
    contact_body_labels = (
        "left_foot",
        "right_foot",
        "left_ankle_pitch",
        "right_ankle_pitch",
        "left_knee",
        "right_knee",
    )
    if args.track_ball_contacts:
        body_root = "/World/Env0/G1/Geometry/pelvis"
        left_lower = (
            body_root + "/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link"
        )
        right_lower = (
            body_root
            + "/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link"
        )
        contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/Env0/Ball",
                update_period=0.0,
                filter_prim_paths_expr=[
                    left_lower + "/left_ankle_pitch_link/left_ankle_roll_link",
                    right_lower + "/right_ankle_pitch_link/right_ankle_roll_link",
                    left_lower + "/left_ankle_pitch_link",
                    right_lower + "/right_ankle_pitch_link",
                    left_lower,
                    right_lower,
                ],
            )
        )
    sim.reset()
    if robot.num_instances != args.agent_count:
        raise ValueError("Isaac articulation instance count differs from requested agents")
    if set(robot.joint_names) != set(names) or len(robot.joint_names) != 29:
        raise ValueError("Isaac 29-DoF joint names differ from SONIC")
    indices = [robot.joint_names.index(name) for name in names]
    hip_pitch_indices = [
        names.index(name) for name in ("left_hip_pitch_joint", "right_hip_pitch_joint")
    ]
    ankle_pitch_indices = [
        names.index(name) for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
    ]
    right_knee_index = names.index("right_knee_joint")
    right_hip_pitch_index = names.index("right_hip_pitch_joint")
    right_ankle_pitch_index = names.index("right_ankle_pitch_joint")
    contact_kinematic_body_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    )
    contact_kinematic_body_indices = [
        robot.body_names.index(name) for name in contact_kinematic_body_names
    ]
    right_foot_body_index = robot.body_names.index("right_ankle_roll_link")
    left_foot_body_index = robot.body_names.index("left_ankle_roll_link")
    right_knee_body_index = robot.body_names.index("right_knee_link")
    right_leg_indices = (right_hip_pitch_index, right_knee_index, right_ankle_pitch_index)
    right_leg_jacobian_columns = [6 + robot.joint_names.index(names[i]) for i in right_leg_indices]
    if args.right_foot_ik_forward_m != 0.0 and (
        robot.data.body_link_jacobian_w.torch.shape[-1] != 35 or robot.num_base_dofs != 6
    ):
        raise ValueError("floating-base right-foot Jacobian contract changed")
    initial_joint = robot.data.default_joint_pos.torch.clone()
    initial_joint[:, indices] = torch.as_tensor(
        G1SonicRunupController.default_angles, device=sim.device, dtype=initial_joint.dtype
    )
    pose = robot.data.default_root_pose.torch.clone()
    lane_y_m = [2.5 * (index - (args.agent_count - 1) / 2) for index in range(args.agent_count)]
    for index, lane_y in enumerate(lane_y_m):
        pose[index, :3] = torch.tensor((0.0, lane_y, 0.793), device=sim.device)
    root_quat = (
        (1.0, 0.0, 0.0, 0.0)
        if args.root_orientation == "legacy_numeric_identity"
        else (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
        if args.root_orientation == "official_yaw90"
        else (0.0, 0.0, 0.0, 1.0)
    )
    pose[:, 3:7] = torch.tensor(root_quat, device=sim.device)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=initial_joint)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()
    ball.write_root_pose_to_sim_index(root_pose=ball.data.default_root_pose.torch.clone())
    ball_initial_velocity = ball.data.default_root_vel.torch.clone()
    ball_initial_velocity[0, :2] = torch.tensor(
        (args.ball_initial_vx_m_s, args.ball_initial_vy_m_s), device=sim.device
    )
    if args.ball_rolling_start:
        ball_initial_velocity[0, 3:5] = torch.tensor(
            (-args.ball_initial_vy_m_s / 0.11, args.ball_initial_vx_m_s / 0.11),
            device=sim.device,
        )
    ball.write_root_velocity_to_sim_index(root_velocity=ball_initial_velocity)
    ball.reset()
    contact_speed_actor = (
        json.loads(args.contact_speed_actor.read_text(encoding="utf-8"))
        if args.contact_speed_actor is not None
        else None
    )
    if contact_speed_actor is not None and (
        contact_speed_actor["model_hash"] != qualification.qualification_hash
        or contact_speed_actor["asset_hash"] != asset_hash
    ):
        raise ValueError("contact-speed actor model or asset identity mismatch")
    selected_forward_speed_m_s = (
        choose_contact_speed(
            contact_speed_actor,
            observed_ball_x_m=float(ball.data.root_pos_w.torch[0, 0]),
        )
        if contact_speed_actor is not None
        else args.forward_command_m_s
    )
    if contact_sensor is not None:
        contact_sensor.reset()
    qpos_rows = []
    qvel_rows = []
    target_rows = []
    ball_position_rows = []
    ball_observation_position_rows = []
    ball_observation_velocity_rows = []
    ball_observation_angular_velocity_rows = []
    ball_body_contact_force_rows = []
    ball_body_contact_micro_rows = []
    contact_body_position_rows = []
    contact_body_velocity_rows = []
    initial_body_xyz_m: dict[str, dict[str, list[float]]] = {}
    brake_released = [False] * args.agent_count
    brake_release_frame: list[int | None] = [None] * args.agent_count
    root_x_history: list[list[float]] = [[] for _ in agent_ids]
    joint_projection_count = [0] * args.agent_count
    maximum_joint_projection_rad = [0.0] * args.agent_count
    foot_ik_applied_frames = 0
    for frame in range(args.frames):
        ball_observation_position_rows.append(
            ball.data.root_pos_w.torch[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        ball_observation_velocity_rows.append(
            ball.data.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        ball_observation_angular_velocity_rows.append(
            ball.data.root_ang_vel_w.torch[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        contact_body_position_rows.append(
            robot.data.body_pos_w.torch[:, contact_kinematic_body_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        )
        contact_body_velocity_rows.append(
            robot.data.body_lin_vel_w.torch[:, contact_kinematic_body_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        )
        if frame == 0:
            initial_body_xyz_m = {
                agent_id: {
                    name: [float(value) for value in body_positions[body_index]]
                    for body_index, name in enumerate(robot.body_names)
                    if any(part in name.lower() for part in ("pelvis", "ankle", "foot"))
                }
                for agent_id, body_positions in zip(
                    agent_ids, robot.data.body_pos_w.torch.detach().cpu().numpy(), strict=True
                )
            }
        target = robot.data.joint_pos.torch.clone()
        frame_qpos = []
        frame_qvel = []
        frame_targets = []
        for index, (agent_id, navigation) in enumerate(zip(agent_ids, navigations, strict=True)):
            root_pose = (
                robot.data.root_link_pose_w.torch[index].detach().cpu().numpy().astype(np.float64)
            )
            root_velocity = (
                robot.data.root_link_vel_w.torch[index].detach().cpu().numpy().astype(np.float64)
            )
            joint = (
                robot.data.joint_pos.torch[index, indices].detach().cpu().numpy().astype(np.float64)
            )
            velocity = (
                robot.data.joint_vel.torch[index, indices].detach().cpu().numpy().astype(np.float64)
            )
            if args.observation_root_frame == "asset_aligned":
                qpos_root, qvel_root = isaac_root_to_mujoco(
                    pose_xyzw=root_pose,
                    velocity_world=root_velocity,
                    asset_quaternion_xyzw=np.asarray(root_quat),
                )
            else:
                qpos_root, qvel_root = root_pose, root_velocity
            world_qpos = np.concatenate((qpos_root, joint, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
            qvel = np.concatenate((qvel_root, velocity, np.zeros(6)))
            if (
                world_qpos.shape != (43,)
                or qvel.shape != (41,)
                or not np.isfinite(world_qpos).all()
            ):
                raise ValueError("Isaac body observation is invalid")
            policy_qpos = world_qpos.copy()
            policy_qpos[1] -= lane_y_m[index]
            root_x_history[index].append(float(root_pose[0]))
            release_speed = float(root_velocity[0])
            window = args.brake_release_window_frames
            if len(root_x_history[index]) > window:
                release_speed = (root_x_history[index][-1] - root_x_history[index][-window - 1]) / (
                    window * 0.02
                )
            if (
                args.stop_at_frame is not None
                and args.brake_release_speed_m_s is not None
                and frame >= args.stop_at_frame + args.brake_min_frames
                and not brake_released[index]
                and release_speed <= args.brake_release_speed_m_s
            ):
                brake_released[index] = True
                brake_release_frame[index] = frame
            lateral_command = (
                args.lateral_command_m_s
                if (args.stop_at_frame is None or frame < args.stop_at_frame)
                and (
                    args.lateral_start_frame is None
                    or args.lateral_start_frame <= frame < args.lateral_end_frame
                )
                else 0.0
            )
            if args.reactive_lateral_command_m_s != 0.0:
                ball_xyz = ball.data.root_pos_w.torch[0].detach().cpu().numpy()
                ball_gap_m = float(ball_xyz[0] - root_pose[0])
                ball_origin_displacement_m = float(
                    np.linalg.norm(ball_xyz[:2] - np.asarray((args.ball_x_m, args.ball_y_m)))
                )
                if (
                    (args.stop_at_frame is None or frame < args.stop_at_frame)
                    and 0.15 <= ball_gap_m <= 0.70
                    and ball_origin_displacement_m < 0.05
                ):
                    lateral_command = args.reactive_lateral_command_m_s
            obs = TeamMotorObservation(
                agent_id=agent_id,
                frame=frame,
                time_sec=frame * 0.02,
                intent="other",
                prospective_owner=False,
                qpos=tuple(float(value) for value in policy_qpos),
                qvel=tuple(float(value) for value in qvel),
                target_position_m=(0.0, 0.0, 0.0),
                navigation_command=(
                    selected_forward_speed_m_s
                    if args.stop_at_frame is None or frame < args.stop_at_frame
                    else args.brake_command_m_s
                    if frame < args.stop_at_frame + args.brake_frames and not brake_released[index]
                    else 0.0,
                    lateral_command,
                    0.0,
                ),
                navigation_envelope=navigation.navigation_envelope,
            )
            if args.control_mode == "sonic":
                if frame == 0:
                    navigation.start_from_observation(obs)
                try:
                    proposal = navigation.propose(obs)
                except ValueError:
                    print(
                        "RSI_ISAAC_SONIC_PROPOSAL_DIAG="
                        + json.dumps(
                            {
                                "agent_id": agent_id,
                                "frame": frame,
                                "root_z_m": float(root_pose[2]),
                                "observed_quat_wxyz": world_qpos[3:7].tolist(),
                                "max_abs_joint_rad": float(np.max(np.abs(joint))),
                                "max_abs_joint_velocity_rad_s": float(np.max(np.abs(velocity))),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    raise
                proposal_target = np.asarray(proposal.target_rad, dtype=np.float64)
            else:
                proposal_target = np.asarray(
                    G1SonicRunupController.default_angles, dtype=np.float64
                )
            if args.stop_at_frame is not None and frame >= args.stop_at_frame:
                speed_window = min(10, len(root_x_history[index]) - 1)
                measured_speed = (
                    root_x_history[index][-1] - root_x_history[index][-speed_window - 1]
                ) / (speed_window * 0.02)
                bounded_speed = min(0.6, max(0.0, measured_speed))
                ramp = min(1.0, (frame - args.stop_at_frame + 1) / 5.0)
                for joint_index in hip_pitch_indices:
                    proposal_target[joint_index] += hip_brake_gains[index] * bounded_speed * ramp
                for joint_index in ankle_pitch_indices:
                    proposal_target[joint_index] += ankle_brake_gains[index] * bounded_speed * ramp
            if (
                args.stop_at_frame is not None
                and frame >= args.stop_at_frame
                and (hip_brake_gains[index] or ankle_brake_gains[index])
            ):
                limits = (
                    robot.data.joint_pos_limits.torch[index, indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                modified = (hip_pitch_indices if hip_brake_gains[index] else []) + (
                    ankle_pitch_indices if ankle_brake_gains[index] else []
                )
                try:
                    projection = project_modified_joint_targets(
                        target_rad=proposal_target,
                        limits_rad=limits,
                        modified_indices=modified,
                    )
                except ValueError:
                    print(
                        "RSI_ISAAC_JOINT_PROJECTION_DIAG="
                        + json.dumps(
                            {
                                "agent_id": agent_id,
                                "frame": frame,
                                "joints": [
                                    {
                                        "name": names[joint_index],
                                        "target_rad": float(proposal_target[joint_index]),
                                        "lower_rad": float(limits[joint_index, 0]),
                                        "upper_rad": float(limits[joint_index, 1]),
                                    }
                                    for joint_index in modified
                                ],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    raise
                proposal_target = projection.target_rad
                joint_projection_count[index] += projection.projection_count
                maximum_joint_projection_rad[index] = max(
                    maximum_joint_projection_rad[index], projection.maximum_projection_rad
                )
            if (
                args.right_knee_contact_residual_rad != 0.0
                or args.right_hip_pitch_contact_residual_rad != 0.0
                or args.right_ankle_pitch_contact_residual_rad != 0.0
            ):
                ball_xyz = ball.data.root_pos_w.torch[0].detach().cpu().numpy()
                ball_gap_m = float(ball_xyz[0] - root_pose[0])
                ball_origin_displacement_m = float(
                    np.linalg.norm(ball_xyz[:2] - np.asarray((args.ball_x_m, args.ball_y_m)))
                )
                if 0.1 <= ball_gap_m <= 1.0 and ball_origin_displacement_m < 0.05:
                    phase_weight = min(1.0, (1.0 - ball_gap_m) / 0.5)
                    modified_indices = []
                    if args.right_knee_contact_residual_rad != 0.0:
                        proposal_target[right_knee_index] += (
                            args.right_knee_contact_residual_rad * phase_weight
                        )
                        modified_indices.append(right_knee_index)
                    if args.right_hip_pitch_contact_residual_rad != 0.0:
                        proposal_target[right_hip_pitch_index] += (
                            args.right_hip_pitch_contact_residual_rad * phase_weight
                        )
                        modified_indices.append(right_hip_pitch_index)
                    if args.right_ankle_pitch_contact_residual_rad != 0.0:
                        proposal_target[right_ankle_pitch_index] += (
                            args.right_ankle_pitch_contact_residual_rad * phase_weight
                        )
                        modified_indices.append(right_ankle_pitch_index)
                    limits = (
                        robot.data.joint_pos_limits.torch[index, indices]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    projection = project_modified_joint_targets(
                        target_rad=proposal_target,
                        limits_rad=limits,
                        modified_indices=modified_indices,
                    )
                    proposal_target = projection.target_rad
                    joint_projection_count[index] += projection.projection_count
                    maximum_joint_projection_rad[index] = max(
                        maximum_joint_projection_rad[index], projection.maximum_projection_rad
                    )
            if args.right_foot_ik_forward_m != 0.0:
                ball_xyz = ball.data.root_pos_w.torch[0].detach().cpu().numpy()
                body_xyz = robot.data.body_pos_w.torch[index].detach().cpu().numpy()
                foot_xyz = body_xyz[right_foot_body_index]
                ball_gap_m = float(ball_xyz[0] - root_pose[0])
                ball_origin_displacement_m = float(
                    np.linalg.norm(ball_xyz[:2] - np.asarray((args.ball_x_m, args.ball_y_m)))
                )
                if (
                    0.1 <= ball_gap_m <= 0.75
                    and ball_origin_displacement_m < 0.05
                    and foot_xyz[2] > 0.09
                    and body_xyz[left_foot_body_index, 2] < 0.10
                ):
                    jacobian_full = (
                        robot.data.body_link_jacobian_w.torch[index, right_foot_body_index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    jacobian = jacobian_full[:, right_leg_jacobian_columns]
                    knee_jacobian_x = (
                        robot.data.body_link_jacobian_w.torch[index, right_knee_body_index, 0]
                        .detach()
                        .cpu()
                        .numpy()[right_leg_jacobian_columns]
                    )
                    desired_xz = np.asarray(
                        (
                            min(
                                args.right_foot_ik_forward_m,
                                max(0.0, float(ball_xyz[0] - foot_xyz[0]) - 0.05),
                            ),
                            float(np.clip(ball_xyz[2] - foot_xyz[2], -0.04, 0.04)),
                        ),
                        dtype=np.float64,
                    )
                    delta = bounded_foot_target_delta(
                        jacobian[[0, 2]],
                        desired_xz,
                        knee_jacobian_x_m_per_rad=(
                            knee_jacobian_x if args.right_knee_clearance_m != 0.0 else None
                        ),
                        knee_retreat_m=args.right_knee_clearance_m,
                    )
                    for joint_index, joint_delta in zip(right_leg_indices, delta, strict=True):
                        proposal_target[joint_index] += joint_delta
                    limits = (
                        robot.data.joint_pos_limits.torch[index, indices]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    projection = project_modified_joint_targets(
                        target_rad=proposal_target,
                        limits_rad=limits,
                        modified_indices=right_leg_indices,
                    )
                    proposal_target = projection.target_rad
                    joint_projection_count[index] += projection.projection_count
                    maximum_joint_projection_rad[index] = max(
                        maximum_joint_projection_rad[index], projection.maximum_projection_rad
                    )
                    foot_ik_applied_frames += 1
            target[index, indices] = torch.as_tensor(
                proposal_target, device=sim.device, dtype=target.dtype
            )
            frame_qpos.append(world_qpos[:36].copy())
            frame_qvel.append(qvel[:35].copy())
            frame_targets.append(proposal_target)
        foot_force_peak = np.zeros(len(contact_body_labels), dtype=np.float64)
        micro_forces = []
        for _ in range(10):
            robot.set_joint_position_target_index(target=target)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            ball.update(sim.get_physics_dt())
            if contact_sensor is not None:
                contact_sensor.update(sim.get_physics_dt())
                matrix = contact_sensor.data.force_matrix_w
                if matrix is None:
                    raise ValueError("foot-ball filtered contact matrix unavailable")
                force = matrix.torch.detach().cpu().numpy()
                if (
                    force.shape != (1, 1, len(contact_body_labels), 3)
                    or not np.isfinite(force).all()
                ):
                    raise ValueError("foot-ball contact matrix shape or values invalid")
                contact_force_n = np.linalg.norm(force[0, 0], axis=1)
                foot_force_peak = np.maximum(foot_force_peak, contact_force_n)
                micro_forces.append(contact_force_n.copy())
            else:
                micro_forces.append(np.zeros(len(contact_body_labels), dtype=np.float64))
        qpos_rows.append(frame_qpos)
        qvel_rows.append(frame_qvel)
        target_rows.append(frame_targets)
        ball_position_rows.append(
            ball.data.root_pos_w.torch[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        ball_body_contact_force_rows.append(foot_force_peak)
        ball_body_contact_micro_rows.append(micro_forces)
    arrays = {
        "qpos": np.asarray(qpos_rows),
        "qvel": np.asarray(qvel_rows),
        "target": np.asarray(target_rows),
        "ball_position_m": np.asarray(ball_position_rows),
        "ball_observation_position_m": np.asarray(ball_observation_position_rows),
        "ball_observation_velocity_m_s": np.asarray(ball_observation_velocity_rows),
        "ball_observation_angular_velocity_rad_s": np.asarray(
            ball_observation_angular_velocity_rows
        ),
        "ball_body_contact_force_peak_n": np.asarray(ball_body_contact_force_rows),
        "ball_body_contact_force_micro_n": np.asarray(ball_body_contact_micro_rows),
        "contact_body_position_m": np.asarray(contact_body_position_rows),
        "contact_body_velocity_m_s": np.asarray(contact_body_velocity_rows),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite Isaac SONIC trajectory")
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
    heights = arrays["qpos"][:, :, 2]
    contact_summary = classify_ball_body_contacts(arrays["ball_body_contact_force_micro_n"])
    impact_steps = (
        contact_summary.first_foot_microstep,
        contact_summary.first_nonfoot_microstep,
    )
    first_body_impact = min((step for step in impact_steps if step is not None), default=None)
    rolling = precontact_rolling_evidence(
        arrays["ball_observation_position_m"],
        arrays["ball_observation_velocity_m_s"],
        arrays["ball_observation_angular_velocity_rad_s"],
        first_body_impact_microstep=first_body_impact,
    )
    individual = {}
    for index, agent_id in enumerate(agent_ids):
        trajectory = arrays["qpos"][:, index, :]
        displacement = trajectory[:, :2] - trajectory[0, :2]
        stop_x_m = (
            float(trajectory[args.stop_at_frame, 0] - trajectory[0, 0])
            if args.stop_at_frame is not None
            else None
        )
        individual[agent_id] = {
            "min_pelvis_height_m": float(heights[:, index].min()),
            "final_pelvis_height_m": float(heights[-1, index]),
            "forward_displacement_m": float(trajectory[-1, 0] - trajectory[0, 0]),
            "max_displacement_m": float(np.max(np.linalg.norm(displacement, axis=1))),
            "post_stop_displacement_m": (
                float(trajectory[-1, 0] - trajectory[args.stop_at_frame, 0])
                if args.stop_at_frame is not None
                else None
            ),
            "post_stop_peak_backtrack_m": (
                float(np.max(trajectory[args.stop_at_frame :, 0]) - trajectory[-1, 0])
                if args.stop_at_frame is not None
                else None
            ),
            "pre_stop_displacement_m": stop_x_m,
            "brake_release_frame": brake_release_frame[index],
            "joint_projection_count": joint_projection_count[index],
            "maximum_joint_projection_rad": maximum_joint_projection_rad[index],
            "terminal_half_second_speed_m_s": float(
                (trajectory[-1, 0] - trajectory[-min(26, args.frames), 0])
                / (min(25, args.frames - 1) * 0.02)
            ),
            "stand_passed": bool(heights[:, index].min() >= 0.55),
        }
    report = {
        "schema": "rosclaw_soccer.rsi.isaac_sonic_stand.v3",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "asset_hash": asset_hash,
        "model_hash": qualification.qualification_hash,
        "joint_names": list(names),
        "joint_map_hash": hash_json(list(names)),
        "gain_hash": hash_json({"kp": kp.tolist(), "kd": kd.tolist()}),
        "joint_armature_kg_m2": 0.01 if args.actuator_mode == "sonic_implicit" else None,
        "control_mode": args.control_mode,
        "actuator_mode": args.actuator_mode,
        "root_orientation": args.root_orientation,
        "root_quat_xyzw": root_quat,
        "observation_root_frame": args.observation_root_frame,
        "forward_command_m_s": selected_forward_speed_m_s,
        "experimental_high_speed": args.experimental_high_speed,
        "requested_forward_command_m_s": args.forward_command_m_s,
        "contact_speed_actor_hash": (
            contact_speed_actor["actor_hash"] if contact_speed_actor is not None else None
        ),
        "lateral_command_m_s": args.lateral_command_m_s,
        "lateral_start_frame": args.lateral_start_frame,
        "lateral_end_frame": args.lateral_end_frame,
        "reactive_lateral_command_m_s": args.reactive_lateral_command_m_s,
        "stop_at_frame": args.stop_at_frame,
        "brake_command_m_s": args.brake_command_m_s,
        "brake_frames": args.brake_frames,
        "brake_release_speed_m_s": args.brake_release_speed_m_s,
        "brake_release_window_frames": args.brake_release_window_frames,
        "brake_min_frames": args.brake_min_frames,
        "hip_brake_gains": hip_brake_gains,
        "ankle_brake_gains": ankle_brake_gains,
        "agent_count": args.agent_count,
        "agent_ids": agent_ids,
        "lane_y_m": lane_y_m,
        "initial_body_xyz_m": initial_body_xyz_m,
        "individual": individual,
        "frames": args.frames,
        "min_pelvis_height_m": float(heights.min()),
        "final_pelvis_height_m": float(heights[-1].min()),
        "ball_final_height_m": float(ball.data.root_pos_w.torch[0, 2].item()),
        "ball_initial_xyz_m": arrays["ball_position_m"][0].tolist(),
        "ball_final_xyz_m": ball.data.root_pos_w.torch[0].detach().cpu().numpy().tolist(),
        "ball_horizontal_displacement_m": float(
            np.linalg.norm(
                ball.data.root_pos_w.torch[0, :2].detach().cpu().numpy()
                - arrays["ball_position_m"][0, :2]
            )
        ),
        "ball_body_contact_labels": contact_body_labels,
        "contact_kinematic_body_names": contact_kinematic_body_names,
        "ball_body_contact_peak_n": np.max(
            arrays["ball_body_contact_force_peak_n"], axis=0
        ).tolist(),
        "foot_ball_contact_peak_n": np.max(
            arrays["ball_body_contact_force_peak_n"][:, :2], axis=0
        ).tolist(),
        "foot_ball_contact_frames": int(
            np.count_nonzero(np.max(arrays["ball_body_contact_force_peak_n"][:, :2], axis=1) > 1.0)
        ),
        "nonfoot_lower_leg_ball_contact_frames": int(
            np.count_nonzero(np.max(arrays["ball_body_contact_force_peak_n"][:, 2:], axis=1) > 1.0)
        ),
        "first_foot_ball_contact_microstep": contact_summary.first_foot_microstep,
        "first_nonfoot_lower_leg_ball_contact_microstep": contact_summary.first_nonfoot_microstep,
        "foot_first_contact_verified": contact_summary.foot_first,
        "clean_foot_only_contact_verified": contact_summary.clean_foot_only,
        "track_ball_contacts": args.track_ball_contacts,
        "ball_x_m": args.ball_x_m,
        "ball_y_m": args.ball_y_m,
        "ball_initial_vx_m_s": args.ball_initial_vx_m_s,
        "ball_initial_vy_m_s": args.ball_initial_vy_m_s,
        "ball_rolling_start": args.ball_rolling_start,
        "ball_precontact_grounded_frames": rolling.grounded_frames,
        "ball_precontact_mean_slip_m_s": rolling.mean_slip_m_s,
        "ball_precontact_maximum_slip_m_s": rolling.maximum_slip_m_s,
        "right_knee_contact_residual_rad": args.right_knee_contact_residual_rad,
        "right_hip_pitch_contact_residual_rad": args.right_hip_pitch_contact_residual_rad,
        "right_ankle_pitch_contact_residual_rad": args.right_ankle_pitch_contact_residual_rad,
        "right_foot_ik_forward_m": args.right_foot_ik_forward_m,
        "right_knee_clearance_m": args.right_knee_clearance_m,
        "foot_ik_applied_frames": foot_ik_applied_frames,
        "trajectory_hash": hash_bytes((args.output_dir / "trajectory.npz").read_bytes()),
        "stand_passed": bool(heights.min() >= 0.55 and np.isfinite(heights[-1]).all()),
        "trained_actor": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print("RSI_ISAAC_SONIC_STAND=" + json.dumps(report, sort_keys=True), flush=True)
    if hash_bytes(Path(__file__).read_bytes()) != source_hash:
        raise RuntimeError("source changed during Isaac SONIC stand probe")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        chain = []
        current: BaseException | None = exc
        while current is not None and len(chain) < 5:
            chain.append(f"{type(current).__name__}:{current}")
            current = current.__cause__
        print("RSI_ISAAC_SONIC_ERROR=" + " <- ".join(chain), flush=True)
        os._exit(2)
    simulation_app.close()
