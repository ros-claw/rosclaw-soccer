"""SIM_ONLY vector first-touch physics smoke; no learning or policy promotion.

Each lane has its own G1, rolling ball, SONIC state and ball/body contact trace.
The report deliberately counts episodes rather than correlated control frames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--g1-usd", required=True, type=Path)
parser.add_argument("--model-root", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--env-count", type=int, default=4)
parser.add_argument("--training-course-seed", type=int)
parser.add_argument("--single-course-lane", type=int)
parser.add_argument("--navigation-speed-mps", type=float, default=1.4)
parser.add_argument("--planner-seed", type=int, default=30300)
parser.add_argument("--near-ball-gap-m", type=float)
parser.add_argument("--near-ball-speed-mps", type=float)
parser.add_argument("--near-ball-incoming-only", action="store_true")
parser.add_argument("--record-body-trace", action="store_true")
parser.add_argument("--record-foot-geometry", action="store_true")
parser.add_argument("--inference-threads", type=int, choices=range(1, 9))
parser.add_argument("--onnx-graph-encoder-layout", action="store_true")
parser.add_argument("--torch-batch-shadow", action="store_true")
parser.add_argument("--torch-batch-drive", action="store_true")
parser.add_argument("--torch-batch-plan-only", action="store_true")
parser.add_argument("--reset-replay", action="store_true")
parser.add_argument("--second-reset-replay", action="store_true")
parser.add_argument("--candidate-actions", type=Path)
parser.add_argument("--temporal-policy-actions", type=Path)
parser.add_argument("--late-swing-policy", type=Path)
parser.add_argument("--revalidate-swing-side", action="store_true")
parser.add_argument("--late-swing-lateral-cap-m", type=float, default=0.05)
parser.add_argument("--support-knee-retract-m", type=float, default=0.0)
parser.add_argument("--support-knee-lane", type=int)
parser.add_argument("--temporal-followthrough-frames", type=int, default=0)
parser.add_argument("--parent-report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    not args.g1_usd.is_file()
    or not args.model_root.is_dir()
    or args.output_dir.exists()
    or not 50 <= args.frames <= 400
    or not 1 <= args.env_count <= 16
    or (
        (args.env_count == 1) != (args.single_course_lane is not None)
        or (
            args.single_course_lane is not None
            and (
                args.training_course_seed is None
                or not 0 <= args.single_course_lane < 16
                or args.candidate_actions is not None
                or args.temporal_policy_actions is not None
            )
        )
    )
    or not 0.8 <= args.navigation_speed_mps <= 1.5
    or not 0 <= args.planner_seed <= 2**31 - 3000
    or (
        args.planner_seed != 30300
        and (args.candidate_actions is not None or args.temporal_policy_actions is not None)
    )
    or ((args.near_ball_gap_m is None) != (args.near_ball_speed_mps is None))
    or (args.near_ball_gap_m is not None and not 0.6 <= args.near_ball_gap_m <= 1.6)
    or (args.near_ball_speed_mps is not None and not 0.8 <= args.near_ball_speed_mps <= 1.5)
    or (args.near_ball_incoming_only and args.near_ball_gap_m is None)
    or (args.second_reset_replay and not args.reset_replay)
    or sum(
        value is not None
        for value in (args.candidate_actions, args.temporal_policy_actions, args.late_swing_policy)
    )
    > 1
    or (
        any(
            value is not None
            for value in (
                args.candidate_actions,
                args.temporal_policy_actions,
                args.late_swing_policy,
            )
        )
        != (args.parent_report is not None)
    )
    or (args.candidate_actions is not None and not args.candidate_actions.is_file())
    or (args.temporal_policy_actions is not None and not args.temporal_policy_actions.is_file())
    or (args.late_swing_policy is not None and not args.late_swing_policy.is_file())
    or (
        args.late_swing_policy is not None
        and (
            args.frames != 300
            or args.env_count not in (1, 16)
            or args.training_course_seed is None
            or not args.record_body_trace
            or not args.record_foot_geometry
            or not args.torch_batch_plan_only
            or args.reset_replay
        )
    )
    or not 0 <= args.temporal_followthrough_frames <= 30
    or (args.temporal_followthrough_frames > 0 and args.temporal_policy_actions is None)
    or (args.record_foot_geometry and not args.record_body_trace)
    or args.support_knee_retract_m not in (0.0, 0.04, 0.08)
    or (args.support_knee_retract_m != 0.0 and args.late_swing_policy is None)
    or (args.revalidate_swing_side and args.late_swing_policy is None)
    or args.late_swing_lateral_cap_m not in (0.05, 0.10, 0.15)
    or (args.late_swing_lateral_cap_m != 0.05 and args.late_swing_policy is None)
    or (args.late_swing_lateral_cap_m != 0.05 and args.env_count != 1)
    or (
        args.support_knee_lane is not None
        and (args.support_knee_retract_m == 0.0 or not 0 <= args.support_knee_lane < args.env_count)
    )
    or (args.parent_report is not None and not args.parent_report.is_file())
    or (
        any(
            value is not None
            for value in (
                args.candidate_actions,
                args.temporal_policy_actions,
                args.late_swing_policy,
            )
        )
        and args.reset_replay
    )
    or (
        args.temporal_policy_actions is not None
        and (not args.record_body_trace or not args.torch_batch_plan_only or args.env_count != 16)
    )
    or (args.torch_batch_drive and not args.torch_batch_shadow)
    or (args.torch_batch_plan_only and (args.torch_batch_shadow or args.torch_batch_drive))
    or (
        args.training_course_seed is not None
        and (
            args.env_count not in (1, 16)
            or args.reset_replay
            or not 0 <= args.training_course_seed < 2**32
        )
    )
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
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch  # noqa: E402
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker  # noqa: E402
from rosclaw_soccer.rsi import contact_time_phase_features as time_phase_module  # noqa: E402
from rosclaw_soccer.rsi import late_swing_memory as late_swing_module  # noqa: E402
from rosclaw_soccer.rsi import taskspace_gate_memory as taskspace_gate_module  # noqa: E402
from rosclaw_soccer.rsi.contact_time_phase_features import (  # noqa: E402
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.first_touch_candidate import (  # noqa: E402
    JOINT_NAMES,
    load_first_touch_candidate,
    project_residual_target,
)
from rosclaw_soccer.rsi.first_touch_course_catalog import (  # noqa: E402
    sample_training_courses,
    static_development_courses,
)
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor  # noqa: E402
from rosclaw_soccer.rsi.support_knee_evidence import (  # noqa: E402
    audit_support_knee_action_trace,
)
from rosclaw_soccer.rsi.support_knee_nullspace import (  # noqa: E402
    support_knee_nullspace_delta,
)
from rosclaw_soccer.rsi.taskspace_gate_memory import select_taskspace_gate  # noqa: E402
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES  # noqa: E402
from rosclaw_soccer.rsi.taskspace_swing_probe import (  # noqa: E402
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)
from rosclaw_soccer.rsi.temporal_first_touch_policy import (  # noqa: E402
    JOINT_NAMES as TEMPORAL_JOINT_NAMES,
)
from rosclaw_soccer.rsi.temporal_first_touch_policy import (  # noqa: E402  # noqa: E402
    followthrough_residual,
    temporal_residual,
)
from rosclaw_soccer.rsi.temporal_first_touch_policy import (  # noqa: E402
    load_candidate as load_temporal_candidate,
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
                planner_seed=args.planner_seed,
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
    full_courses = (
        sample_training_courses(args.training_course_seed, 16)
        if args.training_course_seed is not None
        else None
    )
    courses = list(
        (full_courses[args.single_course_lane],)
        if args.single_course_lane is not None and full_courses is not None
        else static_development_courses(args.env_count)
        if full_courses is None
        else full_courses
    )
    candidate = None
    temporal_candidate = None
    late_actor = None
    if (
        any(
            value is not None
            for value in (
                args.candidate_actions,
                args.temporal_policy_actions,
                args.late_swing_policy,
            )
        )
    ) and args.parent_report is not None:
        parent_audit = audit_vector_first_touch(args.parent_report.parent)
        parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
        if (
            parent_audit["source_report_hash"] != parent["report_hash"]
            or parent["asset_hash"] != hash_bytes(args.g1_usd.read_bytes())
            or parent["sonic_qualification_hash"]
            != navigations[0].backend.qualification.qualification_hash
            or parent.get("onnx_graph_encoder_layout", False) is not args.onnx_graph_encoder_layout
            or parent.get("torch_batch_plan_only", False) is not args.torch_batch_plan_only
            or parent.get("navigation_speed_mps", 1.4) != args.navigation_speed_mps
            or parent.get("near_ball_gap_m") != args.near_ball_gap_m
            or parent.get("near_ball_speed_mps") != args.near_ball_speed_mps
            or parent.get("near_ball_incoming_only", False) is not args.near_ball_incoming_only
            or parent.get("training_course_seed") != args.training_course_seed
            or parent.get("single_course_lane") != args.single_course_lane
            or (
                args.training_course_seed is not None
                and parent.get("course_catalog_hash") != hash_json(full_courses)
            )
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
        if args.candidate_actions is not None:
            candidate = load_first_touch_candidate(
                args.candidate_actions,
                expected_courses=tuple(courses),
                parent_report_hash=parent["report_hash"],
            )
            print("RSI_ISAAC_CANDIDATE_READY=" + candidate.candidate_hash, flush=True)
        elif args.temporal_policy_actions is not None:
            temporal_candidate = load_temporal_candidate(
                args.temporal_policy_actions,
                expected_courses=tuple(courses),
                parent_report_hash=parent["report_hash"],
            )
            print(
                "RSI_ISAAC_TEMPORAL_CANDIDATE_READY=" + temporal_candidate.candidate_hash,
                flush=True,
            )
        else:
            late_actor = load_late_swing_actor(args.late_swing_policy)
            if (
                late_actor["policy_source_hash"]
                != hash_bytes(Path(taskspace_gate_module.__file__).read_bytes())
                or late_actor["feature_source_hash"]
                != hash_bytes(Path(time_phase_module.__file__).read_bytes())
                or late_actor["loader_source_hash"]
                != hash_bytes(Path(late_swing_module.__file__).read_bytes())
            ):
                raise ValueError("late-swing actor source changed")
            print("RSI_ISAAC_LATE_SWING_READY=" + late_actor["actor_hash"], flush=True)
    candidate_joint_indices = [robot.joint_names.index(name) for name in JOINT_NAMES]
    temporal_joint_indices = [robot.joint_names.index(name) for name in TEMPORAL_JOINT_NAMES]
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
    foot_geometry_body_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    )
    foot_geometry_indices = (
        [robot.body_names.index(name) for name in foot_geometry_body_names]
        if args.record_foot_geometry
        else []
    )
    leg_joint_indices = (
        tuple(tuple(robot.joint_names.index(name) for name in row) for row in LEG_NAMES)
        if late_actor is not None
        else ()
    )

    def rollout(current_navigations: list[G1SonicNavigation]):
        positions = []
        contact_forces = []
        angular_velocities = []
        robot_root_observations = []
        robot_root_velocity_observations = []
        robot_joint_observations = []
        robot_joint_velocity_observations = []
        robot_target_observations = []
        command_speed_observations = []
        temporal_ball_position_observations = []
        temporal_ball_velocity_observations = []
        temporal_baseline_target_observations = []
        temporal_residual_observations = []
        foot_geometry_position_observations = []
        foot_geometry_velocity_observations = []
        swing_foot_positions = []
        swing_linear_jacobians = []
        support_knee_jacobians = []
        support_applied_residuals = []
        swing_selected_sides = []
        swing_applied_residuals = []
        swing_baseline_targets = []
        swing_executed_targets = []
        swing_side = np.full(args.env_count, -1, dtype=np.int64)
        swing_contact_frame = np.full(args.env_count, -1, dtype=np.int64)
        swing_contact_delta = np.zeros((args.env_count, 6), dtype=np.float64)
        selected_taskspace_mask = np.zeros(args.env_count, dtype=np.bool_)
        gate_features = None
        applied_frames = np.zeros(args.env_count, dtype=np.int64)
        projection_counts = np.zeros(args.env_count, dtype=np.int64)
        contact_seen = np.zeros(args.env_count, dtype=np.bool_)
        first_contact_frames = np.full(args.env_count, -1, dtype=np.int64)
        contact_residuals = np.zeros((args.env_count, len(TEMPORAL_JOINT_NAMES)))
        minimum_pelvis = np.full(args.env_count, np.inf)
        batch_tracker = None
        batch_target_max_difference = 0.0
        batch_model = (
            FrozenSonicG1Torch(args.model_root, variant="low_latency", device=str(sim.device))
            if args.torch_batch_shadow or args.torch_batch_plan_only
            else None
        )
        for frame in range(args.frames):
            ball_xyz_frame = (
                ball.data.root_pos_w.torch.detach().cpu().numpy()
                if args.near_ball_gap_m is not None
                or temporal_candidate is not None
                or args.record_foot_geometry
                else None
            )
            ball_velocity_frame = (
                ball.data.root_lin_vel_w.torch.detach().cpu().numpy()
                if temporal_candidate is not None or args.record_foot_geometry
                else None
            )
            ball_vx_frame = (
                ball.data.root_lin_vel_w.torch[:, 0].detach().cpu().numpy()
                if args.near_ball_incoming_only
                else None
            )
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
            if args.record_foot_geometry:
                foot_geometry_position_observations.append(
                    robot.data.body_pos_w.torch[:, foot_geometry_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                foot_geometry_velocity_observations.append(
                    robot.data.body_lin_vel_w.torch[:, foot_geometry_indices]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
            target = robot.data.joint_pos.torch.clone()
            frame_command_speeds = np.full(args.env_count, args.navigation_speed_mps)
            qpos_rows = []
            qvel_rows = []
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
                qpos_rows.append(qpos)
                qvel_rows.append(qvel)
                command_speed = args.navigation_speed_mps
                if (
                    args.near_ball_gap_m is not None
                    and ball_xyz_frame is not None
                    and not contact_seen[i]
                ):
                    gap_m = float(ball_xyz_frame[i, 0] - root_pose[0])
                    lateral_gap_m = float(ball_xyz_frame[i, 1] - root_pose[1])
                    if (
                        0.15 <= gap_m <= args.near_ball_gap_m
                        and abs(lateral_gap_m) <= 0.5
                        and (ball_vx_frame is None or ball_vx_frame[i] < -0.05)
                    ):
                        command_speed = args.near_ball_speed_mps
                obs = TeamMotorObservation(
                    agent_id=f"vector.first_touch.{i}",
                    frame=frame,
                    time_sec=frame * 0.02,
                    intent="other",
                    prospective_owner=False,
                    qpos=tuple(float(v) for v in qpos),
                    qvel=tuple(float(v) for v in qvel),
                    target_position_m=(0.0, 0.0, 0.0),
                    navigation_command=(command_speed, 0.0, 0.0),
                    navigation_envelope=navigation.navigation_envelope,
                )
                frame_command_speeds[i] = command_speed
                if frame == 0:
                    navigation.start_from_observation(obs)
                if args.torch_batch_plan_only:
                    navigation.prepare_batched_proposal(obs)
                else:
                    proposal = navigation.propose(obs)
                    target[i, indices] = torch.as_tensor(proposal.target_rad, device=sim.device)
            if batch_model is not None:
                qpos_batch = np.asarray(qpos_rows)
                qvel_batch = np.asarray(qvel_rows)
                if batch_tracker is None:
                    batch_tracker = BatchedSonicTracker(
                        batch_model,
                        np.stack(
                            [navigation.backend.reference for navigation in current_navigations]
                        ),
                        low_latency_legacy_encoder_layout=not args.onnx_graph_encoder_layout,
                    )
                    batch_tracker.reset(qpos_batch, qvel_batch)
                else:
                    batch_tracker.observe(qpos_batch, qvel_batch)
                    if frame % current_navigations[0].config.replan_frames == 0:
                        batch_tracker.refresh_unexecuted_reference(
                            frame,
                            np.stack(
                                [navigation.backend.reference for navigation in current_navigations]
                            ),
                            unchanged_lookahead_frames=current_navigations[
                                0
                            ].config.lookahead_frames,
                        )
                batch_target = batch_tracker.update(frame, qpos_batch, qvel_batch)
                if args.torch_batch_plan_only:
                    for i, navigation in enumerate(current_navigations):
                        proposal = navigation.commit_batched_action(
                            batch_tracker.action[i].detach().cpu().numpy()
                        )
                        target[i, indices] = torch.as_tensor(proposal.target_rad, device=sim.device)
                difference = float(
                    torch.max(torch.abs(batch_target - target[:, indices])).detach().cpu()
                )
                batch_target_max_difference = max(batch_target_max_difference, difference)
                if difference > 1e-3:
                    raise ValueError("batched SONIC target differs from verified navigation target")
                if args.torch_batch_drive:
                    target[:, indices] = batch_target
            baseline_target = target.clone()
            if late_actor is not None and frame == 30:
                local_root = robot_root_observations[-1].copy()
                local_ball = ball_xyz_frame.copy()
                local_root[:, 1] -= lanes
                local_ball[:, 1] -= lanes
                raw = current_context(
                    local_root,
                    robot_root_velocity_observations[-1],
                    local_ball,
                    ball_velocity_frame,
                )
                predicted = predict_contact_time(
                    raw, np.asarray(late_actor["contact_time_weights"])
                )
                gate_features = gait_phase_features(raw, predicted)
                selected_taskspace_mask = select_taskspace_gate(
                    gate_features,
                    np.asarray(late_actor["memory_features"]),
                    np.asarray(late_actor["memory_clean"]),
                    np.asarray(late_actor["memory_reward"]),
                    np.asarray(late_actor["memory_groups"]),
                    neighbors=late_actor["neighbors"],
                    confidence=late_actor["confidence"],
                    baseline_clean_ceiling=late_actor["baseline_clean_ceiling"],
                )
                selected_taskspace_mask &= np.asarray([course[2] < 0 for course in courses])
            frame_temporal_residual = np.zeros((args.env_count, len(TEMPORAL_JOINT_NAMES)))
            if temporal_candidate is not None or args.record_foot_geometry:
                temporal_ball_position_observations.append(ball_xyz_frame.copy())
                temporal_ball_velocity_observations.append(ball_velocity_frame.copy())
            if temporal_candidate is not None:
                temporal_baseline_target_observations.append(
                    baseline_target[:, indices].detach().cpu().numpy().copy()
                )
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
            elif temporal_candidate is not None:
                root_xyz = robot.data.root_link_pose_w.torch[:, :3].detach().cpu().numpy()
                limits = robot.data.joint_pos_limits.torch.detach().cpu().numpy()
                for index, weights in enumerate(temporal_candidate.weights_per_course):
                    if contact_seen[index]:
                        if args.temporal_followthrough_frames == 0:
                            continue
                        residuals = followthrough_residual(
                            contact_residuals[index],
                            elapsed_frames=frame - int(first_contact_frames[index]),
                            followthrough_frames=args.temporal_followthrough_frames,
                        )
                    else:
                        residuals = temporal_residual(
                            weights,
                            ball_relative_xyz_m=tuple(
                                float(value) for value in ball_xyz_frame[index] - root_xyz[index]
                            ),
                            ball_vx_m_s=float(ball_velocity_frame[index, 0]),
                            joint_position_rad=robot_joint_observations[-1][index],
                            joint_velocity_rad_s=robot_joint_velocity_observations[-1][index],
                        )
                    if not np.any(residuals):
                        if not contact_seen[index]:
                            contact_residuals[index] = 0.0
                        continue
                    for output_index, (joint_index, residual) in enumerate(
                        zip(temporal_joint_indices, residuals, strict=True)
                    ):
                        value, projected = project_residual_target(
                            float(target[index, joint_index]),
                            float(residual),
                            float(limits[index, joint_index, 0]),
                            float(limits[index, joint_index, 1]),
                        )
                        frame_temporal_residual[index, output_index] = value - float(
                            baseline_target[index, joint_index]
                        )
                        target[index, joint_index] = value
                        projection_counts[index] += projected
                    if not contact_seen[index]:
                        contact_residuals[index] = frame_temporal_residual[index]
                    applied_frames[index] += 1
            if late_actor is not None:
                feet = (
                    robot.data.body_link_pos_w.torch[:, foot_geometry_indices[:2]]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                full_jacobian = robot.data.body_link_jacobian_w.torch.detach().cpu().numpy()
                jacobians = np.stack(
                    [
                        np.take(
                            full_jacobian[:, body_index, :3, :],
                            np.asarray(leg_joint_indices[side]) + 6,
                            axis=-1,
                        )
                        for side, body_index in enumerate(foot_geometry_indices[:2])
                    ],
                    axis=1,
                )
                if jacobians.shape != (args.env_count, 2, 3, 6):
                    raise ValueError("full-episode foot Jacobian shape changed")
                knee_jacobians = np.stack(
                    [
                        np.take(
                            full_jacobian[:, body_index, 0, :],
                            np.asarray(leg_joint_indices[side]) + 6,
                            axis=-1,
                        )
                        for side, body_index in enumerate(foot_geometry_indices[2:])
                    ],
                    axis=1,
                )
                if knee_jacobians.shape != (args.env_count, 2, 6):
                    raise ValueError("full-episode knee Jacobian shape changed")
                limits = robot.data.joint_pos_limits.torch.detach().cpu().numpy()
                baseline = baseline_target.detach().cpu().numpy()
                frame_swing_residual = np.zeros((args.env_count, len(robot.joint_names)))
                frame_support_residual = np.zeros((args.env_count, len(robot.joint_names)))
                for lane in range(args.env_count):
                    if not selected_taskspace_mask[lane]:
                        continue
                    if swing_contact_frame[lane] < 0:
                        swing_side[lane] = choose_swing_side(
                            feet[lane],
                            ball_xyz_frame[lane],
                            int(swing_side[lane]),
                            acquisition_max_gap_m=0.55,
                            revalidate_swing_side=args.revalidate_swing_side,
                        )
                    side = int(swing_side[lane])
                    if side < 0:
                        continue
                    joint_ids = list(leg_joint_indices[side])
                    if swing_contact_frame[lane] >= 0:
                        delta = release_joint_delta(
                            swing_contact_delta[lane], frame - int(swing_contact_frame[lane])
                        )
                    else:
                        delta = swing_joint_delta(
                            feet[lane, side],
                            ball_xyz_frame[lane],
                            jacobians[lane, side],
                            baseline[lane, joint_ids],
                            limits[lane, joint_ids],
                            forward_cap_m=0.08,
                            lateral_cap_m=args.late_swing_lateral_cap_m,
                            vertical_offset_m=0.04,
                        )
                    target[lane, joint_ids] = torch.as_tensor(
                        baseline[lane, joint_ids] + delta, device=sim.device, dtype=torch.float32
                    )
                    frame_swing_residual[lane, joint_ids] = (
                        target[lane, joint_ids].detach().cpu().numpy() - baseline[lane, joint_ids]
                    )
                    if (
                        args.support_knee_retract_m
                        and swing_contact_frame[lane] < 0
                        and (args.support_knee_lane is None or lane == args.support_knee_lane)
                    ):
                        support = 1 - side
                        support_ids = list(leg_joint_indices[support])
                        support_action = support_knee_nullspace_delta(
                            jacobians[lane, support],
                            knee_jacobians[lane, support],
                            baseline[lane, support_ids],
                            limits[lane, support_ids],
                            retract_m=args.support_knee_retract_m,
                            support_foot_grounded=bool(feet[lane, support, 2] < 0.10),
                            swing_foot_airborne=bool(
                                feet[lane, side, 2] - feet[lane, support, 2] >= 0.02
                            ),
                        )
                        if not support_action.abstained:
                            target[lane, support_ids] = torch.as_tensor(
                                baseline[lane, support_ids]
                                + np.asarray(support_action.joint_delta_rad),
                                device=sim.device,
                                dtype=torch.float32,
                            )
                            frame_support_residual[lane, support_ids] = (
                                target[lane, support_ids].detach().cpu().numpy()
                                - baseline[lane, support_ids]
                            )
                            frame_swing_residual[lane, support_ids] = frame_support_residual[
                                lane, support_ids
                            ]
                    applied_frames[lane] += int(np.any(np.abs(frame_swing_residual[lane]) > 1e-6))
                swing_foot_positions.append(feet)
                swing_linear_jacobians.append(jacobians)
                support_knee_jacobians.append(knee_jacobians)
                support_applied_residuals.append(frame_support_residual)
                swing_selected_sides.append(swing_side.copy())
                swing_applied_residuals.append(frame_swing_residual)
                swing_baseline_targets.append(baseline)
                swing_executed_targets.append(target.detach().cpu().numpy().copy())
            robot_target_observations.append(target[:, indices].detach().cpu().numpy().copy())
            command_speed_observations.append(frame_command_speeds)
            if temporal_candidate is not None:
                temporal_residual_observations.append(frame_temporal_residual)
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
                        (candidate is not None or temporal_candidate is not None)
                        and not contact_seen[i]
                        and bool(torch.any(torch.linalg.vector_norm(force[0, 0], dim=-1) > 1.0))
                    ):
                        contact_seen[i] = True
                        first_contact_frames[i] = frame
                        active_joint_indices = (
                            candidate_joint_indices
                            if candidate is not None
                            else temporal_joint_indices
                        )
                        if not (
                            temporal_candidate is not None and args.temporal_followthrough_frames
                        ):
                            target[i, active_joint_indices] = baseline_target[
                                i, active_joint_indices
                            ]
            positions.append(ball.data.root_pos_w.torch.detach().cpu().numpy().copy())
            angular_velocities.append(ball.data.root_ang_vel_w.torch.detach().cpu().numpy().copy())
            contact_forces.append(frame_forces_gpu.detach().cpu().numpy().copy())
            if late_actor is not None:
                contact_now = contact_forces[-1]
                last_delta = swing_applied_residuals[-1]
                for lane in range(args.env_count):
                    if swing_contact_frame[lane] < 0 and np.any(contact_now[lane] > 1.0):
                        swing_contact_frame[lane] = frame
                        if swing_side[lane] >= 0:
                            joint_ids = list(leg_joint_indices[int(swing_side[lane])])
                            swing_contact_delta[lane] = last_delta[lane, joint_ids]
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
            np.asarray(command_speed_observations),
            np.asarray(temporal_ball_position_observations),
            np.asarray(temporal_ball_velocity_observations),
            np.asarray(temporal_baseline_target_observations),
            np.asarray(temporal_residual_observations),
            np.asarray(foot_geometry_position_observations),
            np.asarray(foot_geometry_velocity_observations),
            applied_frames,
            projection_counts,
            batch_target_max_difference,
            (
                np.asarray(swing_foot_positions),
                np.asarray(swing_linear_jacobians),
                np.asarray(swing_selected_sides),
                np.asarray(swing_applied_residuals),
                np.asarray(swing_baseline_targets),
                np.asarray(swing_executed_targets),
                selected_taskspace_mask,
                gate_features,
                np.asarray(support_knee_jacobians),
                np.asarray(support_applied_residuals),
            ),
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
        command_speed_observations,
        temporal_ball_position_observations,
        temporal_ball_velocity_observations,
        temporal_baseline_target_observations,
        temporal_residual_observations,
        foot_geometry_position_observations,
        foot_geometry_velocity_observations,
        applied_frames,
        projection_counts,
        batch_target_max_difference,
        swing_data,
    ) = rollout(navigations)
    if (
        not np.isfinite(positions_arr).all()
        or not np.isfinite(angular_arr).all()
        or not np.isfinite(forces_arr).all()
    ):
        raise ValueError("nonfinite vector trajectory")
    lateral_excursion = np.max(np.abs(positions_arr[:, :, 1] - lanes[None, :]), axis=0)
    if late_actor is not None:
        root_xyz = root_observations[:, :, :3]
        if args.env_count == 1:
            minimum_cross_robot_distance = None
            minimum_cross_ball_distance = None
            isolated = True
        else:
            cross_robot_distance = np.linalg.norm(
                positions_arr[:, :, None, :] - root_xyz[:, None, :, :], axis=-1
            )
            cross_ball_distance = np.linalg.norm(
                positions_arr[:, :, None, :] - positions_arr[:, None, :, :], axis=-1
            )
            diagonal = np.eye(args.env_count, dtype=np.bool_)
            cross_robot_distance[:, diagonal] = np.inf
            cross_ball_distance[:, diagonal] = np.inf
            minimum_cross_robot_distance = float(np.min(cross_robot_distance))
            minimum_cross_ball_distance = float(np.min(cross_ball_distance))
            isolated = bool(
                np.max(lateral_excursion) < 6.0
                and minimum_cross_robot_distance > 2.0
                and minimum_cross_ball_distance > 1.0
            )
    else:
        minimum_cross_robot_distance = None
        minimum_cross_ball_distance = None
        isolated = bool(args.env_count == 1 or np.max(lateral_excursion) < 4.0)
    if not isolated:
        args.output_dir.mkdir(parents=True)
        failure_path = args.output_dir / "lane_escape_trace.npz"
        np.savez_compressed(
            failure_path,
            ball_position_m=positions_arr,
            ball_angular_velocity_rad_s=angular_arr,
            ball_body_contact_force_peak_n=forces_arr,
            root_pose_xyzw_m=root_observations,
            joint_target_rad=target_observations,
            selected_taskspace_mask=swing_data[6],
        )
        failure = {
            "schema": "rsi_isaac_vector_first_touch_lane_escape_v1",
            "activation_ceiling": "SIM_ONLY",
            "runner_source_hash": hash_bytes(Path(__file__).read_bytes()),
            "parent_report_hash": parent["report_hash"] if args.parent_report else None,
            "late_swing_actor_hash": late_actor["actor_hash"] if late_actor else None,
            "trace_hash": hash_bytes(failure_path.read_bytes()),
            "lateral_excursion_m": lateral_excursion.tolist(),
            "escaped_lanes": np.flatnonzero(lateral_excursion >= 4.0).tolist(),
            "minimum_cross_robot_distance_m": minimum_cross_robot_distance,
            "minimum_cross_ball_distance_m": minimum_cross_ball_distance,
            "promotion_authorized": False,
        }
        failure["report_hash"] = hash_json(failure)
        (args.output_dir / "lane_escape_report.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise ValueError("ball escaped its isolated training lane; diagnostic trace preserved")
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "trace.npz",
        ball_position_m=positions_arr,
        ball_angular_velocity_rad_s=angular_arr,
        ball_body_contact_force_peak_n=forces_arr,
    )
    trace_hash = hash_bytes((args.output_dir / "trace.npz").read_bytes())
    body_trace_hash = None
    if args.record_body_trace:
        body_trace_path = args.output_dir / "body_trace.npz"
        body_record = {
            "root_pose_xyzw_m": root_observations,
            "root_velocity_world": root_velocity_observations,
            "joint_position_rad": joint_observations,
            "joint_velocity_rad_s": joint_velocity_observations,
            "joint_target_rad": target_observations,
            "navigation_speed_mps": command_speed_observations,
        }
        if args.record_foot_geometry:
            body_record.update(
                ball_position_before_step_m=temporal_ball_position_observations,
                ball_linear_velocity_before_step_m_s=temporal_ball_velocity_observations,
                foot_geometry_position_before_step_m=np.asarray(
                    foot_geometry_position_observations
                ),
                foot_geometry_velocity_before_step_m_s=np.asarray(
                    foot_geometry_velocity_observations
                ),
            )
        if temporal_candidate is not None:
            body_record.update(
                ball_position_before_step_m=temporal_ball_position_observations,
                ball_linear_velocity_before_step_m_s=temporal_ball_velocity_observations,
                baseline_joint_target_rad=temporal_baseline_target_observations,
                applied_residual_rad=temporal_residual_observations,
            )
        np.savez_compressed(body_trace_path, **body_record)
        body_trace_hash = hash_bytes(body_trace_path.read_bytes())
    swing_trace_hash = None
    if late_actor is not None:
        (
            feet,
            jacobians,
            sides,
            residuals,
            baseline,
            executed,
            selected_mask,
            gate_features,
            knee_jacobians,
            support_residuals,
        ) = swing_data
        if gate_features is None or not np.isfinite(gate_features).all():
            raise ValueError("late-swing gate was not evaluated")
        ball_local = np.asarray(temporal_ball_position_observations).copy()
        ball_local[:, :, 1] -= lanes[None, :]
        action_path = args.output_dir / "late_swing_action_trace.npz"
        np.savez_compressed(
            action_path,
            pre_step_foot_link_position_w=feet,
            pre_step_foot_linear_jacobian_w=jacobians,
            taskspace_selected_side=sides,
            applied_taskspace_joint_delta_rad=residuals,
            baseline_taskspace_joint_target_rad=baseline,
            executed_taskspace_joint_target_rad=executed,
            taskspace_joint_limits_rad=robot.data.joint_pos_limits.torch.detach().cpu().numpy(),
            pre_step_ball_position_local_m=ball_local,
            observed_ball_body_contact_force_peak_n=forces_arr,
            predicted_baseline_joint_target_rad=baseline[:, :, indices],
            frame30_gate_features=gate_features,
            pre_step_knee_x_jacobian_w=knee_jacobians,
            applied_support_knee_joint_delta_rad=support_residuals,
        )
        swing_trace_hash = hash_bytes(action_path.read_bytes())
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
            else "rsi_isaac_vector_first_touch_temporal_candidate_v1"
            if temporal_candidate is not None
            else "rsi_isaac_vector_first_touch_late_swing_v1"
            if late_actor is not None
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
        "navigation_speed_mps": args.navigation_speed_mps,
        "inference_threads": args.inference_threads,
        "environments": entries,
    }
    if args.planner_seed != 30300:
        report["planner_seed"] = args.planner_seed
    if args.torch_batch_shadow:
        report["torch_batch_shadow"] = True
        report["torch_batch_drive"] = args.torch_batch_drive
        report["torch_batch_max_target_difference_rad"] = batch_target_max_difference
    if args.torch_batch_plan_only:
        report["torch_batch_plan_only"] = True
        report["torch_batch_max_internal_target_difference_rad"] = batch_target_max_difference
    if args.training_course_seed is not None:
        report["training_course_seed"] = args.training_course_seed
        report["course_catalog_hash"] = hash_json(full_courses)
    if args.single_course_lane is not None:
        report["single_course_lane"] = args.single_course_lane
        report["single_instance_max_lateral_excursion_m"] = float(np.max(lateral_excursion))
    if args.near_ball_gap_m is not None:
        report["near_ball_gap_m"] = args.near_ball_gap_m
        report["near_ball_speed_mps"] = args.near_ball_speed_mps
        report["near_ball_incoming_only"] = args.near_ball_incoming_only
    if body_trace_hash is not None:
        report["body_trace_hash"] = body_trace_hash
    if args.record_foot_geometry:
        report["foot_geometry_body_names"] = list(foot_geometry_body_names)
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
    elif temporal_candidate is not None:
        report["parent_report_hash"] = temporal_candidate.parent_report_hash
        report["candidate_hash"] = temporal_candidate.candidate_hash
        report["temporal_policy_joint_names"] = list(TEMPORAL_JOINT_NAMES)
        report["temporal_policy_applied_frames"] = applied_frames.tolist()
        report["temporal_policy_projection_count"] = projection_counts.tolist()
        report["temporal_followthrough_frames"] = args.temporal_followthrough_frames
        report["trained_actor"] = False
    elif late_actor is not None:
        report.update(
            parent_report_hash=parent["report_hash"],
            late_swing_actor_hash=late_actor["actor_hash"],
            taskspace_probe_hash=hash_bytes(
                Path(choose_swing_side.__code__.co_filename).read_bytes()
            ),
            late_swing_action_trace_hash=swing_trace_hash,
            selected_taskspace_mask=swing_data[6].tolist(),
            taskspace_forward_m=0.08,
            taskspace_lateral_cap_m=args.late_swing_lateral_cap_m,
            taskspace_vertical_offset_m=0.0,
            taskspace_acquisition_max_gap_m=0.95,
            taskspace_revalidate_swing_side=args.revalidate_swing_side,
            taskspace_leg_joint_names=[list(row) for row in LEG_NAMES],
            taskspace_joint_order=list(robot.joint_names),
            taskspace_applied_frames=applied_frames.tolist(),
            support_knee_retract_m=args.support_knee_retract_m,
            support_knee_lane=args.support_knee_lane,
            minimum_cross_robot_distance_m=minimum_cross_robot_distance,
            minimum_cross_ball_distance_m=minimum_cross_ball_distance,
            trained_actor=args.late_swing_lateral_cap_m == 0.05,
        )
        if args.late_swing_lateral_cap_m != 0.05:
            report["late_swing_action_override"] = (
                f"diagnostic_lateral_cap_{args.late_swing_lateral_cap_m:.2f}_m"
            )
        if args.support_knee_retract_m:
            with np.load(action_path) as action_replay:
                report["support_knee_action_audit"] = audit_support_knee_action_trace(
                    action_replay,
                    report,
                    frames=args.frames,
                    count=args.env_count,
                )
            report["support_knee_helper_hash"] = hash_bytes(
                Path(support_knee_nullspace_delta.__code__.co_filename).read_bytes()
            )
            report["support_knee_audit_source_hash"] = hash_bytes(
                Path(audit_support_knee_action_trace.__code__.co_filename).read_bytes()
            )
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
                    planner_seed=args.planner_seed,
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
            _replay_command_speed_observations,
            _replay_temporal_ball_position_observations,
            _replay_temporal_ball_velocity_observations,
            _replay_temporal_baseline_target_observations,
            _replay_temporal_residual_observations,
            _replay_foot_geometry_position_observations,
            _replay_foot_geometry_velocity_observations,
            _replay_applied_frames,
            _replay_projection_counts,
            _replay_batch_target_max_difference,
            _replay_swing_data,
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
                        planner_seed=args.planner_seed,
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
                _second_command_speed_observations,
                _second_temporal_ball_position_observations,
                _second_temporal_ball_velocity_observations,
                _second_temporal_baseline_target_observations,
                _second_temporal_residual_observations,
                _second_foot_geometry_position_observations,
                _second_foot_geometry_velocity_observations,
                _second_applied_frames,
                _second_projection_counts,
                _second_batch_target_max_difference,
                _second_swing_data,
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
    os._exit(1)
else:
    simulation_app.close()
