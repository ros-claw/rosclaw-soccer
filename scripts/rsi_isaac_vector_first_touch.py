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
parser.add_argument("--onnx-graph-encoder-layout", action="store_true")
parser.add_argument("--reset-replay", action="store_true")
parser.add_argument("--second-reset-replay", action="store_true")
parser.add_argument("--candidate-actions", type=Path)
parser.add_argument("--parent-report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    not args.g1_usd.is_file()
    or not args.model_root.is_dir()
    or args.output_dir.exists()
    or not 50 <= args.frames <= 400
    or not 2 <= args.env_count <= 16
    or (args.second_reset_replay and not args.reset_replay)
    or ((args.candidate_actions is None) != (args.parent_report is None))
    or (args.candidate_actions is not None and not args.candidate_actions.is_file())
    or (args.parent_report is not None and not args.parent_report.is_file())
    or (args.candidate_actions is not None and args.reset_replay)
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
from rosclaw_soccer.rsi.first_touch_candidate import (  # noqa: E402
    JOINT_NAMES,
    load_first_touch_candidate,
    project_residual_target,
)
from rosclaw_soccer.rsi.vector_first_touch_evidence import (  # noqa: E402
    audit_vector_first_touch,
)
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
                onnx_graph_encoder_layout=args.onnx_graph_encoder_layout,
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
    candidate = None
    if args.candidate_actions is not None and args.parent_report is not None:
        parent_audit = audit_vector_first_touch(args.parent_report.parent)
        parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
        if (
            parent_audit["source_report_hash"] != parent["report_hash"]
            or parent["asset_hash"] != hash_bytes(args.g1_usd.read_bytes())
            or parent["sonic_qualification_hash"]
            != navigations[0].backend.qualification.qualification_hash
            or parent.get("onnx_graph_encoder_layout", False) is not args.onnx_graph_encoder_layout
            or len(parent["environments"]) != len(courses)
            or [
                (
                    row["course"]["ball_x_m"],
                    row["course"]["ball_y_local_m"],
                    row["course"]["ball_vx_m_s"],
                )
                for row in parent["environments"]
            ]
            != courses
        ):
            raise ValueError("candidate Parent physics, asset or foundation differs")
        candidate = load_first_touch_candidate(
            args.candidate_actions,
            expected_courses=tuple(courses),
            parent_report_hash=parent["report_hash"],
        )
        print("RSI_ISAAC_CANDIDATE_READY=" + candidate.candidate_hash, flush=True)
    candidate_joint_indices = [robot.joint_names.index(name) for name in JOINT_NAMES]
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
    pelvis_index = robot.body_names.index("pelvis")

    def rollout(current_navigations: list[G1SonicNavigation]):
        positions = []
        contact_forces = []
        angular_velocities = []
        robot_root_observations = []
        robot_root_velocity_observations = []
        robot_joint_observations = []
        robot_joint_velocity_observations = []
        robot_target_observations = []
        applied_frames = np.zeros(args.env_count, dtype=np.int64)
        projection_counts = np.zeros(args.env_count, dtype=np.int64)
        contact_seen = np.zeros(args.env_count, dtype=np.bool_)
        minimum_pelvis = np.full(args.env_count, np.inf)
        for frame in range(args.frames):
            robot_root_observations.append(
                robot.data.root_link_pose_w.torch.detach().cpu().numpy().copy()
            )
            robot_root_velocity_observations.append(
                robot.data.root_link_vel_w.torch.detach().cpu().numpy().copy()
            )
            robot_joint_observations.append(
                robot.data.joint_pos.torch[:, indices].detach().cpu().numpy().copy()
            )
            robot_joint_velocity_observations.append(
                robot.data.joint_vel.torch[:, indices].detach().cpu().numpy().copy()
            )
            target = robot.data.joint_pos.torch.clone()
            for i, navigation in enumerate(current_navigations):
                root_pose = (
                    robot.data.root_link_pose_w.torch[i].detach().cpu().numpy().astype(np.float64)
                )
                root_velocity = (
                    robot.data.root_link_vel_w.torch[i].detach().cpu().numpy().astype(np.float64)
                )
                joint = (
                    robot.data.joint_pos.torch[i, indices].detach().cpu().numpy().astype(np.float64)
                )
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
            baseline_target = target.clone()
            if candidate is not None:
                ball_xyz = ball.data.root_pos_w.torch.detach().cpu().numpy()
                root_xyz = robot.data.root_link_pose_w.torch[:, :3].detach().cpu().numpy()
                limits = robot.data.joint_pos_limits.torch.detach().cpu().numpy()
                for index, residuals in enumerate(candidate.actions_rad):
                    gap = float(ball_xyz[index, 0] - root_xyz[index, 0])
                    lateral_gap = float(ball_xyz[index, 1] - root_xyz[index, 1])
                    if not contact_seen[index] and 0.15 <= gap <= 1.0 and abs(lateral_gap) <= 0.5:
                        weight = min(1.0, max(0.0, (1.0 - gap) / 0.5))
                        for joint_index, residual in zip(
                            candidate_joint_indices, residuals, strict=True
                        ):
                            value, projected = project_residual_target(
                                float(target[index, joint_index]),
                                residual * weight,
                                float(limits[index, joint_index, 0]),
                                float(limits[index, joint_index, 1]),
                            )
                            target[index, joint_index] = value
                            projection_counts[index] += projected
                        applied_frames[index] += 1
            robot_target_observations.append(target[:, indices].detach().cpu().numpy().copy())
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
                    if (
                        candidate is not None
                        and not contact_seen[i]
                        and bool(torch.any(torch.linalg.vector_norm(force[0, 0], dim=-1) > 1.0))
                    ):
                        contact_seen[i] = True
                        target[i, candidate_joint_indices] = baseline_target[
                            i, candidate_joint_indices
                        ]
            positions.append(ball.data.root_pos_w.torch.detach().cpu().numpy().copy())
            angular_velocities.append(ball.data.root_ang_vel_w.torch.detach().cpu().numpy().copy())
            contact_forces.append(frame_forces_gpu.detach().cpu().numpy().copy())
            pelvis_z = robot.data.body_pos_w.torch[:, pelvis_index, 2].detach().cpu().numpy()
            minimum_pelvis = np.minimum(minimum_pelvis, pelvis_z)
        return (
            np.asarray(positions),
            np.asarray(angular_velocities),
            np.asarray(contact_forces),
            minimum_pelvis,
            np.asarray(robot_root_observations),
            np.asarray(robot_root_velocity_observations),
            np.asarray(robot_joint_observations),
            np.asarray(robot_joint_velocity_observations),
            np.asarray(robot_target_observations),
            applied_frames,
            projection_counts,
        )

    (
        positions_arr,
        angular_arr,
        forces_arr,
        minimum_pelvis,
        root_observations,
        root_velocity_observations,
        joint_observations,
        joint_velocity_observations,
        target_observations,
        applied_frames,
        projection_counts,
    ) = rollout(navigations)
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
        "schema": (
            "rsi_isaac_vector_first_touch_candidate_execution_v2"
            if candidate is not None
            else "rsi_isaac_vector_first_touch_smoke_v1"
        ),
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
    if args.onnx_graph_encoder_layout:
        report["onnx_graph_encoder_layout"] = True
    if candidate is not None:
        report["parent_report_hash"] = candidate.parent_report_hash
        report["candidate_hash"] = candidate.candidate_hash
        report["candidate_action_joint_names"] = list(JOINT_NAMES)
        report["candidate_actions_rad"] = [list(row) for row in candidate.actions_rad]
        report["candidate_actions_applied_frames"] = applied_frames.tolist()
        report["candidate_action_projection_count"] = projection_counts.tolist()
        report["trained_actor"] = False
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_ISAAC_VECTOR_FIRST_TOUCH=" + json.dumps(report, sort_keys=True), flush=True)
    if args.reset_replay:
        sim.reset(soft=False)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(
            root_velocity=robot.data.default_root_vel.torch.clone()
        )
        robot.write_joint_position_to_sim_index(position=initial_joint)
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        robot.reset()
        ball.write_root_pose_to_sim_index(root_pose=ball_pose)
        ball.write_root_velocity_to_sim_index(root_velocity=ball_velocity)
        ball.reset()
        for contact in contacts:
            contact.reset()
        sim.forward()
        replay_navigations = [
            G1SonicNavigation(
                args.model_root,
                f"vector.first_touch.{index}",
                SonicNavigationConfig(
                    maximum_frames=args.frames,
                    model_variant="low_latency",
                    experimental_maximum_speed_mps=1.5,
                    inference_threads=args.inference_threads,
                    onnx_graph_encoder_layout=args.onnx_graph_encoder_layout,
                ),
            )
            for index in range(args.env_count)
        ]
        if [item.contract_hash for item in replay_navigations] != [
            item.contract_hash for item in navigations
        ]:
            raise ValueError("reset changed the private SONIC contract")
        (
            replay_position,
            replay_spin,
            replay_force,
            replay_pelvis,
            replay_root_observations,
            replay_root_velocity_observations,
            replay_joint_observations,
            replay_joint_velocity_observations,
            replay_target_observations,
            _replay_applied_frames,
            _replay_projection_counts,
        ) = rollout(replay_navigations)
        replay_path = args.output_dir / "reset_replay.npz"
        np.savez_compressed(
            replay_path,
            ball_position_m=replay_position,
            ball_angular_velocity_rad_s=replay_spin,
            ball_body_contact_force_peak_n=replay_force,
        )
        state_probe_path = args.output_dir / "reset_state_probe.npz"
        np.savez_compressed(
            state_probe_path,
            root_before=root_observations,
            root_after=replay_root_observations,
            root_velocity_before=root_velocity_observations,
            root_velocity_after=replay_root_velocity_observations,
            joint_before=joint_observations,
            joint_after=replay_joint_observations,
            joint_velocity_before=joint_velocity_observations,
            joint_velocity_after=replay_joint_velocity_observations,
            target_before=target_observations,
            target_after=replay_target_observations,
        )
        first_frames = []
        replay_first_frames = []
        body_classes_equal = True
        precontact_max_diff_m = 0.0
        for index in range(args.env_count):
            first = np.flatnonzero(np.max(forces_arr[:, index], axis=1) > 1.0)
            replay_first = np.flatnonzero(np.max(replay_force[:, index], axis=1) > 1.0)
            first_frames.append(int(first[0]) if len(first) else None)
            replay_first_frames.append(int(replay_first[0]) if len(replay_first) else None)
            body_classes_equal &= bool(
                np.array_equal(
                    np.max(forces_arr[:, index], axis=0) > 1.0,
                    np.max(replay_force[:, index], axis=0) > 1.0,
                )
            )
            cutoff = min(
                first[0] if len(first) else args.frames,
                replay_first[0] if len(replay_first) else args.frames,
            )
            if cutoff:
                precontact_max_diff_m = max(
                    precontact_max_diff_m,
                    float(
                        np.max(
                            np.abs(positions_arr[:cutoff, index] - replay_position[:cutoff, index])
                        )
                    ),
                )
        reset_verified = bool(
            body_classes_equal
            and first_frames == replay_first_frames
            and precontact_max_diff_m < 0.005
            and float(np.min(replay_pelvis)) >= 0.65
            and np.isfinite(replay_position).all()
            and np.isfinite(replay_spin).all()
            and np.isfinite(replay_force).all()
        )
        reset_report = {
            "schema": "rsi_isaac_vector_first_touch_reset_replay_v1",
            "activation_ceiling": "SIM_ONLY",
            "learning_authorized": False,
            "promotion_authorized": False,
            "source_report_hash": report["report_hash"],
            "replay_trace_hash": hash_bytes(replay_path.read_bytes()),
            "state_probe_hash": hash_bytes(state_probe_path.read_bytes()),
            "source_hash": hash_bytes(Path(__file__).read_bytes()),
            "reset_strategy": "physx_full_reset_then_explicit_state_restore",
            "reset_verified": reset_verified,
            "first_contact_frames": first_frames,
            "replay_first_contact_frames": replay_first_frames,
            "body_classes_equal": body_classes_equal,
            "precontact_max_ball_position_diff_m": precontact_max_diff_m,
            "full_ball_position_max_diff_m": float(np.max(np.abs(positions_arr - replay_position))),
            "minimum_replay_pelvis_z_m": float(np.min(replay_pelvis)),
            "initial_root_max_diff_m": float(
                np.max(np.abs(root_observations[0] - replay_root_observations[0]))
            ),
            "initial_root_velocity_max_diff_m_s": float(
                np.max(np.abs(root_velocity_observations[0] - replay_root_velocity_observations[0]))
            ),
            "initial_joint_max_diff_rad": float(
                np.max(np.abs(joint_observations[0] - replay_joint_observations[0]))
            ),
            "initial_joint_velocity_max_diff_rad_s": float(
                np.max(
                    np.abs(joint_velocity_observations[0] - replay_joint_velocity_observations[0])
                )
            ),
            "initial_target_max_diff_rad": float(
                np.max(np.abs(target_observations[0] - replay_target_observations[0]))
            ),
        }
        reset_report["report_hash"] = hash_json(reset_report)
        (args.output_dir / "reset_report.json").write_text(
            json.dumps(reset_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("RSI_ISAAC_RESET_REPLAY=" + json.dumps(reset_report, sort_keys=True), flush=True)
        if args.second_reset_replay:
            sim.reset(soft=False)
            robot.write_root_pose_to_sim_index(root_pose=pose)
            robot.write_root_velocity_to_sim_index(
                root_velocity=robot.data.default_root_vel.torch.clone()
            )
            robot.write_joint_position_to_sim_index(position=initial_joint)
            robot.write_joint_velocity_to_sim_index(
                velocity=robot.data.default_joint_vel.torch.clone()
            )
            robot.reset()
            ball.write_root_pose_to_sim_index(root_pose=ball_pose)
            ball.write_root_velocity_to_sim_index(root_velocity=ball_velocity)
            ball.reset()
            for contact in contacts:
                contact.reset()
            sim.forward()
            second_navigations = [
                G1SonicNavigation(
                    args.model_root,
                    f"vector.first_touch.{index}",
                    SonicNavigationConfig(
                        maximum_frames=args.frames,
                        model_variant="low_latency",
                        experimental_maximum_speed_mps=1.5,
                        inference_threads=args.inference_threads,
                        onnx_graph_encoder_layout=args.onnx_graph_encoder_layout,
                    ),
                )
                for index in range(args.env_count)
            ]
            (
                second_position,
                second_spin,
                second_force,
                second_pelvis,
                second_root,
                second_root_velocity,
                second_joint,
                second_joint_velocity,
                second_target,
                _second_applied_frames,
                _second_projection_counts,
            ) = rollout(second_navigations)
            second_trace_path = args.output_dir / "second_reset_replay.npz"
            np.savez_compressed(
                second_trace_path,
                ball_position_m=second_position,
                ball_angular_velocity_rad_s=second_spin,
                ball_body_contact_force_peak_n=second_force,
            )
            second_state_path = args.output_dir / "second_reset_state_probe.npz"
            np.savez_compressed(
                second_state_path,
                root=second_root,
                root_velocity=second_root_velocity,
                joint=second_joint,
                joint_velocity=second_joint_velocity,
                target=second_target,
            )
            second_frames = [
                int(active[0]) if len(active) else None
                for index in range(args.env_count)
                for active in [np.flatnonzero(np.max(second_force[:, index], axis=1) > 1.0)]
            ]
            second_bodies = [
                np.flatnonzero(np.max(second_force[:, index], axis=0) > 1.0).tolist()
                for index in range(args.env_count)
            ]
            replay_bodies = [
                np.flatnonzero(np.max(replay_force[:, index], axis=0) > 1.0).tolist()
                for index in range(args.env_count)
            ]
            second_report = {
                "schema": "rsi_isaac_vector_first_touch_second_reset_v1",
                "activation_ceiling": "SIM_ONLY",
                "learning_authorized": False,
                "promotion_authorized": False,
                "first_reset_report_hash": reset_report["report_hash"],
                "second_trace_hash": hash_bytes(second_trace_path.read_bytes()),
                "second_state_hash": hash_bytes(second_state_path.read_bytes()),
                "first_reset_contact_frames": replay_first_frames,
                "second_reset_contact_frames": second_frames,
                "first_reset_contact_bodies": replay_bodies,
                "second_reset_contact_bodies": second_bodies,
                "first_second_ball_max_diff_m": float(
                    np.max(np.abs(replay_position - second_position))
                ),
                "first_second_root_max_diff_m": float(
                    np.max(np.abs(replay_root_observations - second_root))
                ),
                "second_minimum_pelvis_z_m": float(np.min(second_pelvis)),
            }
            second_report["report_hash"] = hash_json(second_report)
            (args.output_dir / "second_reset_report.json").write_text(
                json.dumps(second_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print("RSI_ISAAC_SECOND_RESET=" + json.dumps(second_report, sort_keys=True), flush=True)
        if not reset_verified:
            raise ValueError("independent physics/SONIC reset replay failed")


try:
    main()
except Exception as exc:
    print(f"RSI_ISAAC_VECTOR_FAILURE={type(exc).__name__}:{exc}", flush=True)
    raise
finally:
    simulation_app.close()
