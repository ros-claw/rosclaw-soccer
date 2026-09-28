"""SIM_ONLY vector first-touch physics smoke; no learning or policy promotion.

Each lane has its own G1, rolling ball, SONIC state and ball/body contact trace.
The report deliberately counts episodes rather than correlated control frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--g1-usd", required=True, type=Path)
parser.add_argument("--model-root", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--env-count", type=int, default=4)
parser.add_argument("--inference-threads", type=int, choices=range(1, 9))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    not args.g1_usd.is_file()
    or not args.model_root.is_dir()
    or args.output_dir.exists()
    or not 50 <= args.frames <= 400
    or not 2 <= args.env_count <= 16
):
    parser.error("qualified assets, 2-16 environments and new output directory required")
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
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json  # noqa: E402
from rosclaw_soccer.sim.isaac_root_bridge import isaac_root_to_mujoco  # noqa: E402
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation  # noqa: E402


def main() -> None:
    names = tuple(G1_DDS_JOINT_NAMES)
    navigations = [
        G1SonicNavigation(
            args.model_root,
            f"vector.first_touch.{index}",
            SonicNavigationConfig(
                maximum_frames=args.frames,
                model_variant="low_latency",
                experimental_maximum_speed_mps=1.5,
                inference_threads=args.inference_threads,
            ),
        )
        for index in range(args.env_count)
    ]
    for navigation in navigations:
        navigation.backend.qualification.require_eligible()
    kp = np.asarray(navigations[0].backend.kp, dtype=np.float64)
    kd = np.asarray(navigations[0].backend.kd, dtype=np.float64)
    effort = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    if len(names) != 29 or kp.shape != (29,) or kd.shape != (29,):
        raise ValueError("SONIC 29-DoF contract changed")
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=0.002))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    for index in range(args.env_count):
        sim_utils.create_prim(f"/World/Env{index}", "Xform")
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/Env.*/G1"
    robot_cfg.spawn.usd_path = str(args.g1_usd.resolve())
    robot_cfg.actuators = {
        "sonic": ImplicitActuatorCfg(
            joint_names_expr=list(names),
            effort_limit_sim={name: float(v) for name, v in zip(names, effort, strict=True)},
            stiffness={name: float(v) for name, v in zip(names, kp, strict=True)},
            damping={name: float(v) for name, v in zip(names, kd, strict=True)},
            armature={name: 0.01 for name in names},
        )
    }
    robot = Articulation(cfg=robot_cfg)
    ball = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/Env.*/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.11,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, 0.13)),
        )
    )
    contacts = []
    for index in range(args.env_count):
        root = f"/World/Env{index}/G1/Geometry/pelvis"
        left = root + "/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link"
        right = (
            root + "/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link"
        )
        contacts.append(
            ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/Env{index}/Ball",
                    update_period=0.0,
                    filter_prim_paths_expr=[
                        left + "/left_ankle_pitch_link/left_ankle_roll_link",
                        right + "/right_ankle_pitch_link/right_ankle_roll_link",
                        left + "/left_ankle_pitch_link",
                        right + "/right_ankle_pitch_link",
                        left,
                        right,
                    ],
                )
            )
        )
    sim.reset()
    if robot.num_instances != args.env_count or ball.num_instances != args.env_count:
        raise ValueError("G1/ball environment count mismatch")
    if set(robot.joint_names) != set(names):
        raise ValueError("Isaac and SONIC joint names differ")
    indices = [robot.joint_names.index(name) for name in names]
    # At 6 s the ball can deflect >2 m sideways; 8 m lane spacing prevents
    # a struck ball from contaminating a neighboring G1's episode.
    lanes = np.arange(args.env_count, dtype=np.float64) * 8.0
    courses = [
        (
            2.4 if (i // 4) % 2 == 0 else 2.6,
            -0.1 if (i // 2) % 2 == 0 else 0.1,
            -0.5 if i % 2 == 0 else 0.5,
        )
        for i in range(args.env_count)
    ]
    pose = robot.data.default_root_pose.torch.clone()
    initial_joint = robot.data.default_joint_pos.torch.clone()
    initial_joint[:, indices] = torch.as_tensor(
        G1SonicRunupController.default_angles, device=sim.device, dtype=initial_joint.dtype
    )
    for i, lane_y in enumerate(lanes):
        pose[i, :3] = torch.tensor((0.0, lane_y, 0.793), device=sim.device)
    pose[:, 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=sim.device)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=initial_joint)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()
    ball_pose = ball.data.default_root_pose.torch.clone()
    ball_velocity = ball.data.default_root_vel.torch.clone()
    for i, (x, y, vx) in enumerate(courses):
        ball_pose[i, :3] = torch.tensor((x, lanes[i] + y, 0.13), device=sim.device)
        ball_velocity[i, 0] = vx
        ball_velocity[i, 4] = vx / 0.11
    ball.write_root_pose_to_sim_index(root_pose=ball_pose)
    ball.write_root_velocity_to_sim_index(root_velocity=ball_velocity)
    ball.reset()
    for contact in contacts:
        contact.reset()
    positions = []
    contact_forces = []
    angular_velocities = []
    minimum_pelvis = np.full(args.env_count, np.inf)
    pelvis_index = robot.body_names.index("pelvis")
    for frame in range(args.frames):
        target = robot.data.joint_pos.torch.clone()
        for i, navigation in enumerate(navigations):
            root_pose = (
                robot.data.root_link_pose_w.torch[i].detach().cpu().numpy().astype(np.float64)
            )
            root_velocity = (
                robot.data.root_link_vel_w.torch[i].detach().cpu().numpy().astype(np.float64)
            )
            joint = robot.data.joint_pos.torch[i, indices].detach().cpu().numpy().astype(np.float64)
            velocity = (
                robot.data.joint_vel.torch[i, indices].detach().cpu().numpy().astype(np.float64)
            )
            qroot, vroot = isaac_root_to_mujoco(
                pose_xyzw=root_pose,
                velocity_world=root_velocity,
                asset_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
            )
            qroot[1] -= lanes[i]
            qpos = np.concatenate((qroot, joint, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
            qvel = np.concatenate((vroot, velocity, np.zeros(6)))
            obs = TeamMotorObservation(
                agent_id=f"vector.first_touch.{i}",
                frame=frame,
                time_sec=frame * 0.02,
                intent="other",
                prospective_owner=False,
                qpos=tuple(float(v) for v in qpos),
                qvel=tuple(float(v) for v in qvel),
                target_position_m=(0.0, 0.0, 0.0),
                navigation_command=(1.4, 0.0, 0.0),
                navigation_envelope=navigation.navigation_envelope,
            )
            if frame == 0:
                navigation.start_from_observation(obs)
            proposal = navigation.propose(obs)
            target[i, indices] = torch.as_tensor(proposal.target_rad, device=sim.device)
        frame_forces_gpu = torch.zeros((args.env_count, 6), device=sim.device)
        for _ in range(10):
            robot.set_joint_position_target_index(target=target)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            ball.update(sim.get_physics_dt())
            for i, contact in enumerate(contacts):
                contact.update(sim.get_physics_dt())
                matrix = contact.data.force_matrix_w
                if matrix is None:
                    raise ValueError("ball/body filtered contact matrix unavailable")
                force = matrix.torch
                if force.shape != (1, 1, 6, 3):
                    raise ValueError(f"unexpected lane {i} force matrix {force.shape}")
                frame_forces_gpu[i] = torch.maximum(
                    frame_forces_gpu[i], torch.linalg.vector_norm(force[0, 0], dim=-1)
                )
        positions.append(ball.data.root_pos_w.torch.detach().cpu().numpy().copy())
        angular_velocities.append(ball.data.root_ang_vel_w.torch.detach().cpu().numpy().copy())
        contact_forces.append(frame_forces_gpu.detach().cpu().numpy().copy())
        pelvis_z = robot.data.body_pos_w.torch[:, pelvis_index, 2].detach().cpu().numpy()
        minimum_pelvis = np.minimum(minimum_pelvis, pelvis_z)
    positions_arr = np.asarray(positions)
    angular_arr = np.asarray(angular_velocities)
    forces_arr = np.asarray(contact_forces)
    if (
        not np.isfinite(positions_arr).all()
        or not np.isfinite(angular_arr).all()
        or not np.isfinite(forces_arr).all()
    ):
        raise ValueError("nonfinite vector trajectory")
    if np.max(np.abs(positions_arr[:, :, 1] - lanes[None, :])) >= 4.0:
        raise ValueError("ball escaped its isolated training lane")
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "trace.npz",
        ball_position_m=positions_arr,
        ball_angular_velocity_rad_s=angular_arr,
        ball_body_contact_force_peak_n=forces_arr,
    )
    trace_hash = hash_bytes((args.output_dir / "trace.npz").read_bytes())
    entries = []
    for i, (x, y, vx) in enumerate(courses):
        active = np.flatnonzero(np.max(forces_arr[:, i], axis=1) > 1.0)
        entries.append(
            {
                "environment": i,
                "course": {"ball_x_m": x, "ball_y_local_m": y, "ball_vx_m_s": vx},
                "lane_y_m": float(lanes[i]),
                "minimum_pelvis_z_m": float(minimum_pelvis[i]),
                "first_contact_frame": int(active[0]) if len(active) else None,
                "contact_body_indices": np.flatnonzero(
                    np.max(forces_arr[:, i], axis=0) > 1.0
                ).tolist(),
                "ball_final_local_xyz_m": [
                    float(positions_arr[-1, i, 0]),
                    float(positions_arr[-1, i, 1] - lanes[i]),
                    float(positions_arr[-1, i, 2]),
                ],
            }
        )
    report = {
        "schema": "rsi_isaac_vector_first_touch_smoke_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "source_hash": hash_bytes(Path(__file__).read_bytes()),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "sonic_qualification_hash": navigations[0].backend.qualification.qualification_hash,
        "trace_hash": trace_hash,
        "frames": args.frames,
        "inference_threads": args.inference_threads,
        "environments": entries,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_ISAAC_VECTOR_FIRST_TOUCH=" + json.dumps(report, sort_keys=True), flush=True)


try:
    main()
finally:
    simulation_app.close()
