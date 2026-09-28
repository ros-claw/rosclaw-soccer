"""SIM_ONLY replay-equivalence gate for authenticated precontact G1 snapshots.

The recorded future parent joint targets are privileged diagnostic input. This
script neither trains a controller nor authorizes candidate promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot-bank", required=True, type=Path)
parser.add_argument("--g1-usd", required=True, type=Path)
parser.add_argument("--model-root", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--sample-count", type=int, default=16)
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--knee-extension-probe", action="store_true")
parser.add_argument("--closed-loop-sonic", action="store_true")
parser.add_argument("--shared-candidate", type=Path)
parser.add_argument("--phase-target-frames", type=float)
parser.add_argument("--contextual-phase-policy", type=Path)
parser.add_argument("--local-phase-policy", type=Path)
parser.add_argument("--phase-recovery-frames", type=int, choices=(12, 20, 30))
parser.add_argument("--taskspace-forward-m", type=float, choices=(0.08, 0.16))
parser.add_argument("--taskspace-lateral-cap-m", type=float, choices=(0.05, 0.10), default=0.05)
parser.add_argument(
    "--taskspace-vertical-offset-m", type=float, choices=(-0.04, 0.0, 0.04), default=0.0
)
parser.add_argument(
    "--taskspace-acquisition-max-gap-m", type=float, choices=(0.35, 0.55, 0.95), default=0.95
)
parser.add_argument("--taskspace-gate-policy", type=Path)
parser.add_argument("--taskspace-family-policy", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    not args.snapshot_bank.is_dir()
    or not args.g1_usd.is_file()
    or not args.model_root.is_dir()
    or args.output_dir.exists()
    or not 2 <= args.sample_count <= 16
    or args.start_index < 0
    or (args.knee_extension_probe and args.shared_candidate is not None)
    or (args.shared_candidate is not None and not args.shared_candidate.is_file())
    or (args.shared_candidate is not None and not args.closed_loop_sonic)
    or (args.phase_target_frames is not None and not args.closed_loop_sonic)
    or (
        args.phase_target_frames is not None
        and (args.knee_extension_probe or args.shared_candidate)
    )
    or (args.contextual_phase_policy is not None and not args.contextual_phase_policy.is_file())
    or (args.contextual_phase_policy is not None and not args.closed_loop_sonic)
    or (
        args.contextual_phase_policy is not None
        and (
            args.phase_target_frames is not None
            or args.knee_extension_probe
            or args.shared_candidate
        )
    )
    or (args.local_phase_policy is not None and not args.local_phase_policy.is_file())
    or (args.local_phase_policy is not None and not args.closed_loop_sonic)
    or (args.taskspace_forward_m is not None and args.local_phase_policy is None)
    or (args.taskspace_gate_policy is not None and not args.taskspace_gate_policy.is_file())
    or (args.taskspace_family_policy is not None and not args.taskspace_family_policy.is_file())
    or (args.taskspace_family_policy is not None and args.taskspace_gate_policy is not None)
    or (args.taskspace_family_policy is not None and args.taskspace_forward_m != 0.08)
    or (args.taskspace_family_policy is not None and args.taskspace_acquisition_max_gap_m != 0.95)
    or (
        args.taskspace_family_policy is not None
        and (args.taskspace_lateral_cap_m != 0.05 or args.taskspace_vertical_offset_m != 0.0)
    )
    or (args.taskspace_gate_policy is not None and args.taskspace_forward_m != 0.08)
    or (args.taskspace_gate_policy is not None and args.taskspace_acquisition_max_gap_m != 0.95)
    or (args.taskspace_forward_m is None and args.taskspace_acquisition_max_gap_m != 0.95)
    or (
        args.taskspace_gate_policy is not None
        and (args.taskspace_lateral_cap_m != 0.05 or args.taskspace_vertical_offset_m != 0.0)
    )
    or (
        args.taskspace_forward_m is None
        and (args.taskspace_lateral_cap_m != 0.05 or args.taskspace_vertical_offset_m != 0.0)
    )
    or (args.taskspace_forward_m is not None and args.phase_recovery_frames is not None)
    or (
        args.phase_recovery_frames is not None
        and args.local_phase_policy is None
        and args.phase_target_frames is None
        and args.contextual_phase_policy is None
    )
    or (
        args.local_phase_policy is not None
        and (
            args.phase_target_frames is not None
            or args.contextual_phase_policy is not None
            or args.knee_extension_probe
            or args.shared_candidate
        )
    )
):
    parser.error("qualified snapshot bank, assets and fresh 2-16 lane output required")
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
from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch  # noqa: E402
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker  # noqa: E402
from rosclaw_soccer.rsi import baseline_retention_phase as guarded_phase_module  # noqa: E402
from rosclaw_soccer.rsi import contact_time_phase_features as time_phase_module  # noqa: E402
from rosclaw_soccer.rsi import contextual_phase_policy as phase_policy_module  # noqa: E402
from rosclaw_soccer.rsi import local_phase_memory as local_phase_module  # noqa: E402
from rosclaw_soccer.rsi import taskspace_family_memory as taskspace_family_module  # noqa: E402
from rosclaw_soccer.rsi import taskspace_gate_memory as taskspace_gate_module  # noqa: E402
from rosclaw_soccer.rsi.baseline_retention_phase import (  # noqa: E402
    load_guarded_phase_actor,
    select_guarded_phase,
)
from rosclaw_soccer.rsi.contact_time_phase_features import (  # noqa: E402
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.contextual_phase_policy import (  # noqa: E402
    context_features,
    load_phase_actor,
    select_actions,
)
from rosclaw_soccer.rsi.first_touch_candidate import project_residual_target  # noqa: E402
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank  # noqa: E402
from rosclaw_soccer.rsi.local_phase_memory import (  # noqa: E402
    load_local_phase_actor,
    select_local_phase,
)
from rosclaw_soccer.rsi.snapshot_shared_temporal_policy import load_candidate  # noqa: E402
from rosclaw_soccer.rsi.sonic_phase_probe import (  # noqa: E402
    phase_offset_frames,
    recovered_phase_offset_frames,
)
from rosclaw_soccer.rsi.taskspace_family_memory import (  # noqa: E402
    load_taskspace_family_actor,
    select_taskspace_family,
)
from rosclaw_soccer.rsi.taskspace_gate_memory import (  # noqa: E402
    load_taskspace_gate_actor,
    select_taskspace_gate,
)
from rosclaw_soccer.rsi.taskspace_swing_probe import (  # noqa: E402
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)
from rosclaw_soccer.rsi.temporal_first_touch_policy import (  # noqa: E402
    JOINT_NAMES as PROBE_JOINT_NAMES,
)
from rosclaw_soccer.rsi.temporal_first_touch_policy import (  # noqa: E402
    knee_extension_probe_weights,
    temporal_residual,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json  # noqa: E402
from rosclaw_soccer.sim.isaac_root_bridge import isaac_root_to_mujoco  # noqa: E402
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation  # noqa: E402


def _canonical_state(
    root: np.ndarray, velocity: np.ndarray, joint: np.ndarray, dq: np.ndarray, lane_y: float
) -> tuple[np.ndarray, np.ndarray]:
    qroot, vroot = isaac_root_to_mujoco(
        pose_xyzw=root.astype(np.float64),
        velocity_world=velocity.astype(np.float64),
        asset_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
    )
    qroot[1] -= lane_y
    return (
        np.concatenate((qroot, joint, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))),
        np.concatenate((vroot, dq, np.zeros(6))),
    )


def _observation(
    navigation: G1SonicNavigation, frame: int, qpos: np.ndarray, qvel: np.ndarray
) -> TeamMotorObservation:
    return TeamMotorObservation(
        agent_id=navigation.agent_id,
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


def main() -> None:
    if args.phase_target_frames is not None:
        phase_offset_frames(0, args.phase_target_frames)
    bank_audit = audit_snapshot_bank(args.snapshot_bank)
    manifest = json.loads((args.snapshot_bank / "manifest.json").read_text(encoding="utf-8"))
    shared_candidate_hash = None
    shared_weights = None
    if args.shared_candidate is not None:
        shared_candidate_hash, shared_weights = load_candidate(
            args.shared_candidate, bank_manifest_hash=bank_audit["manifest_hash"]
        )
    stop = args.start_index + args.sample_count
    if stop > manifest["snapshot_count"]:
        raise ValueError("snapshot selection exceeds authenticated bank")
    with np.load(args.snapshot_bank / "snapshots.npz", allow_pickle=False) as archive:
        snapshots = {key: archive[key][args.start_index : stop].copy() for key in archive.files}
    contextual_phase_hash = None
    local_phase_hash = None
    selected_phase_targets = None
    if args.contextual_phase_policy is not None:
        contextual_phase_hash, phase_weights = load_phase_actor(args.contextual_phase_policy)
        policy_manifest = json.loads(args.contextual_phase_policy.read_text(encoding="utf-8"))
        if policy_manifest.get("policy_source_hash") != hash_bytes(
            Path(phase_policy_module.__file__).read_bytes()
        ):
            raise ValueError("contextual phase actor source changed")
        selected_phase_targets = select_actions(
            context_features(
                snapshots["root_pose_local_xyzw_m"],
                snapshots["ball_position_local_m"],
                snapshots["ball_linear_velocity_m_s"],
                snapshots["foot_geometry_position_local_m"],
            ),
            phase_weights,
        )
    if args.local_phase_policy is not None:
        actor_schema = json.loads(args.local_phase_policy.read_text(encoding="utf-8")).get("schema")
        guarded = actor_schema == "rsi_baseline_retention_phase_actor_v5"
        local_actor = (
            load_guarded_phase_actor(args.local_phase_policy)
            if guarded
            else load_local_phase_actor(args.local_phase_policy)
        )
        source_module = guarded_phase_module if guarded else local_phase_module
        if (
            local_actor.get("holdout_open_authorized") is not True
            or local_actor.get("policy_source_hash")
            != hash_bytes(Path(source_module.__file__).read_bytes())
            or local_actor.get("feature_source_hash")
            != hash_bytes(Path(time_phase_module.__file__).read_bytes())
        ):
            raise ValueError("local phase actor lacks frozen source or development gate")
        local_phase_hash = local_actor["actor_hash"]
        raw_context = current_context(
            snapshots["root_pose_local_xyzw_m"],
            snapshots["root_velocity_world"],
            snapshots["ball_position_local_m"],
            snapshots["ball_linear_velocity_m_s"],
        )
        predicted_time = predict_contact_time(
            raw_context, np.asarray(local_actor["contact_time_weights"])
        )
        select = select_guarded_phase if guarded else select_local_phase
        parameters = (
            {"baseline_clean_ceiling": local_actor["baseline_clean_ceiling"]} if guarded else {}
        )
        selected_phase_targets = select(
            gait_phase_features(raw_context, predicted_time),
            np.asarray(local_actor["memory_features"]),
            np.asarray(local_actor["memory_clean"]),
            np.asarray(local_actor["memory_reward"]),
            np.asarray(local_actor["memory_groups"]),
            neighbors=local_actor["neighbors"],
            confidence=local_actor["confidence"],
            **parameters,
        )
    taskspace_gate_hash = None
    taskspace_family_hash = None
    selected_taskspace_mask = np.ones(args.sample_count, dtype=np.bool_)
    selected_taskspace_actions = np.zeros(args.sample_count, dtype=np.int64)
    if args.taskspace_gate_policy is not None:
        gate_actor = load_taskspace_gate_actor(args.taskspace_gate_policy)
        if (
            gate_actor.get("holdout_open_authorized") is not True
            or gate_actor.get("frozen_phase_actor_hash") != local_phase_hash
            or gate_actor.get("policy_source_hash")
            != hash_bytes(Path(taskspace_gate_module.__file__).read_bytes())
            or gate_actor.get("feature_source_hash")
            != hash_bytes(Path(time_phase_module.__file__).read_bytes())
        ):
            raise ValueError("task-space gate lacks frozen source or development authorization")
        taskspace_gate_hash = gate_actor["actor_hash"]
        gate_raw = current_context(
            snapshots["root_pose_local_xyzw_m"],
            snapshots["root_velocity_world"],
            snapshots["ball_position_local_m"],
            snapshots["ball_linear_velocity_m_s"],
        )
        gate_time = predict_contact_time(gate_raw, np.asarray(gate_actor["contact_time_weights"]))
        selected_taskspace_mask = select_taskspace_gate(
            gait_phase_features(gate_raw, gate_time),
            np.asarray(gate_actor["memory_features"]),
            np.asarray(gate_actor["memory_clean"]),
            np.asarray(gate_actor["memory_reward"]),
            np.asarray(gate_actor["memory_groups"]),
            neighbors=gate_actor["neighbors"],
            confidence=gate_actor["confidence"],
            baseline_clean_ceiling=gate_actor["baseline_clean_ceiling"],
        )
    if args.taskspace_family_policy is not None:
        family_actor = load_taskspace_family_actor(args.taskspace_family_policy)
        if (
            family_actor.get("frozen_phase_actor_hash") != local_phase_hash
            or family_actor.get("policy_source_hash")
            != hash_bytes(Path(taskspace_family_module.__file__).read_bytes())
            or family_actor.get("feature_source_hash")
            != hash_bytes(Path(time_phase_module.__file__).read_bytes())
        ):
            raise ValueError("task-space family lacks frozen source or parent")
        taskspace_family_hash = family_actor["actor_hash"]
        family_raw = current_context(
            snapshots["root_pose_local_xyzw_m"],
            snapshots["root_velocity_world"],
            snapshots["ball_position_local_m"],
            snapshots["ball_linear_velocity_m_s"],
        )
        family_time = predict_contact_time(
            family_raw, np.asarray(family_actor["contact_time_weights"])
        )
        selected_taskspace_actions = select_taskspace_family(
            gait_phase_features(family_raw, family_time),
            np.asarray(family_actor["memory_features"]),
            np.asarray(family_actor["memory_clean"]),
            np.asarray(family_actor["memory_reward"]),
            np.asarray(family_actor["memory_groups"]),
            neighbors=family_actor["neighbors"],
            confidence=family_actor["confidence"],
        )
        selected_taskspace_mask = selected_taskspace_actions != 0
    if manifest["source_identity"][1] != hash_bytes(args.g1_usd.read_bytes()):
        raise ValueError("snapshot G1 asset hash changed")
    names = tuple(G1_DDS_JOINT_NAMES)
    navigation = G1SonicNavigation(
        args.model_root,
        "snapshot.replay.foundation",
        SonicNavigationConfig(
            maximum_frames=300, model_variant="low_latency", experimental_maximum_speed_mps=1.5
        ),
    )
    navigation.backend.qualification.require_eligible()
    if (
        manifest["source_identity"][2] != navigation.backend.qualification.qualification_hash
        or len(names) != 29
    ):
        raise ValueError("snapshot frozen foundation qualification changed")
    kp = np.asarray(navigation.backend.kp, dtype=np.float64)
    kd = np.asarray(navigation.backend.kd, dtype=np.float64)
    effort = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    if kp.shape != (29,) or kd.shape != (29,) or effort.shape != (29,):
        raise ValueError("invalid SONIC G1 actuator contract")
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=0.002))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    for index in range(args.sample_count):
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
    for index in range(args.sample_count):
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
    if robot.num_instances != args.sample_count or ball.num_instances != args.sample_count:
        raise ValueError("snapshot replay G1/ball count mismatch")
    indices = [robot.joint_names.index(name) for name in names]
    lanes = np.arange(args.sample_count, dtype=np.float64) * 8.0
    root_pose = snapshots["root_pose_local_xyzw_m"].copy()
    root_pose[:, 1] += lanes
    ball_position = snapshots["ball_position_local_m"].copy()
    ball_position[:, 1] += lanes
    robot_pose = torch.as_tensor(root_pose, device=sim.device, dtype=torch.float32)
    robot_velocity = torch.as_tensor(
        snapshots["root_velocity_world"], device=sim.device, dtype=torch.float32
    )
    joint_position = robot.data.default_joint_pos.torch.clone()
    joint_velocity = robot.data.default_joint_vel.torch.clone()
    joint_position[:, indices] = torch.as_tensor(
        snapshots["joint_position_rad"], device=sim.device, dtype=torch.float32
    )
    joint_velocity[:, indices] = torch.as_tensor(
        snapshots["joint_velocity_rad_s"], device=sim.device, dtype=torch.float32
    )
    robot.write_root_pose_to_sim_index(root_pose=robot_pose)
    robot.write_joint_position_to_sim_index(position=joint_position)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
    robot.reset()
    sim.forward()
    # Isaac's root-velocity writer takes center-of-mass velocity, whereas the
    # authenticated body trace records root *link* velocity. Convert at the
    # restored pose before stepping, rather than silently shifting momentum.
    com_offset = robot.data.root_com_pose_w.torch[:, :3] - robot.data.root_link_pose_w.torch[:, :3]
    com_velocity = robot_velocity.clone()
    com_velocity[:, :3] += torch.linalg.cross(robot_velocity[:, 3:], com_offset)
    robot.write_root_velocity_to_sim_index(root_velocity=com_velocity)
    ball_pose = ball.data.default_root_pose.torch.clone()
    ball_pose[:, :3] = torch.as_tensor(ball_position, device=sim.device, dtype=torch.float32)
    ball_velocity = torch.as_tensor(
        np.concatenate(
            (
                snapshots["ball_linear_velocity_m_s"],
                snapshots["ball_angular_velocity_rad_s"],
            ),
            axis=1,
        ),
        device=sim.device,
        dtype=torch.float32,
    )
    ball.write_root_pose_to_sim_index(root_pose=ball_pose)
    ball.write_root_velocity_to_sim_index(root_velocity=ball_velocity)
    ball.reset()
    for contact in contacts:
        contact.reset()
    sim.forward()
    initial_state_error = {
        "root_pose_m": np.max(
            np.abs(
                robot.data.root_link_pose_w.torch.detach().cpu().numpy() - robot_pose.cpu().numpy()
            ),
            axis=1,
        ).tolist(),
        "root_velocity_m_s": np.max(
            np.abs(
                robot.data.root_link_vel_w.torch.detach().cpu().numpy()
                - snapshots["root_velocity_world"]
            ),
            axis=1,
        ).tolist(),
        "joint_position_rad": np.max(
            np.abs(
                robot.data.joint_pos.torch[:, indices].detach().cpu().numpy()
                - snapshots["joint_position_rad"]
            ),
            axis=1,
        ).tolist(),
        "joint_velocity_rad_s": np.max(
            np.abs(
                robot.data.joint_vel.torch[:, indices].detach().cpu().numpy()
                - snapshots["joint_velocity_rad_s"]
            ),
            axis=1,
        ).tolist(),
        "ball_position_m": np.max(
            np.abs(ball.data.root_pos_w.torch.detach().cpu().numpy() - ball_position), axis=1
        ).tolist(),
        "ball_velocity_m_s": np.max(
            np.abs(
                np.concatenate(
                    (
                        ball.data.root_lin_vel_w.torch.detach().cpu().numpy(),
                        ball.data.root_ang_vel_w.torch.detach().cpu().numpy(),
                    ),
                    axis=1,
                )
                - ball_velocity.cpu().numpy()
            ),
            axis=1,
        ).tolist(),
    }
    initial_root_pose_local = robot.data.root_link_pose_w.torch.detach().cpu().numpy().copy()
    initial_root_pose_local[:, 1] -= lanes
    initial_ball_position_local = ball.data.root_pos_w.torch.detach().cpu().numpy().copy()
    initial_ball_position_local[:, 1] -= lanes
    initial_root_velocity = robot.data.root_link_vel_w.torch.detach().cpu().numpy().copy()
    initial_joint_position = robot.data.joint_pos.torch[:, indices].detach().cpu().numpy().copy()
    initial_joint_velocity = robot.data.joint_vel.torch[:, indices].detach().cpu().numpy().copy()
    initial_ball_velocity = np.concatenate(
        (
            ball.data.root_lin_vel_w.torch.detach().cpu().numpy(),
            ball.data.root_ang_vel_w.torch.detach().cpu().numpy(),
        ),
        axis=1,
    )
    closed_navigations: list[G1SonicNavigation] = []
    batch_tracker = None
    warmup_max_target_error = 0.0
    if args.closed_loop_sonic:
        if manifest.get("fixed_start_frame") is None or manifest["fixed_start_frame"] < 1:
            raise ValueError("closed-loop SONIC requires one common absolute snapshot frame")
        source_data = {}
        lane_sources = []
        for row in manifest["snapshots"][args.start_index : stop]:
            folder = Path(row["source_folder"])
            if folder not in source_data:
                source_report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                if (
                    source_report.get("navigation_speed_mps") != 1.4
                    or source_report.get("torch_batch_plan_only") is not True
                    or source_report.get("planner_seed", 30300) != 30300
                ):
                    raise ValueError("source SONIC controller contract is not replayable")
                with np.load(folder / "body_trace.npz", allow_pickle=False) as record:
                    source_data[folder] = (
                        source_report,
                        {key: record[key] for key in record.files},
                    )
            lane_sources.append((source_data[folder], row["lane"]))
            closed_navigations.append(
                G1SonicNavigation(
                    args.model_root,
                    f"vector.first_touch.{row['lane']}",
                    SonicNavigationConfig(
                        maximum_frames=300,
                        planner_seed=30300,
                        model_variant="low_latency",
                        experimental_maximum_speed_mps=1.5,
                        inference_threads=1,
                    ),
                )
            )
        batch_model = FrozenSonicG1Torch(
            args.model_root, variant="low_latency", device=str(sim.device)
        )
        for frame in range(manifest["fixed_start_frame"]):
            qpos_rows = []
            qvel_rows = []
            for lane, ((source_report, source_body), source_lane) in enumerate(lane_sources):
                qpos, qvel = _canonical_state(
                    source_body["root_pose_xyzw_m"][frame, source_lane],
                    source_body["root_velocity_world"][frame, source_lane],
                    source_body["joint_position_rad"][frame, source_lane],
                    source_body["joint_velocity_rad_s"][frame, source_lane],
                    float(source_report["environments"][source_lane]["lane_y_m"]),
                )
                qpos_rows.append(qpos)
                qvel_rows.append(qvel)
                observation = _observation(closed_navigations[lane], frame, qpos, qvel)
                if frame == 0:
                    closed_navigations[lane].start_from_observation(observation)
                closed_navigations[lane].prepare_batched_proposal(observation)
            qpos_batch = np.asarray(qpos_rows)
            qvel_batch = np.asarray(qvel_rows)
            if batch_tracker is None:
                batch_tracker = BatchedSonicTracker(
                    batch_model,
                    np.stack([nav.backend.reference for nav in closed_navigations]),
                    low_latency_legacy_encoder_layout=True,
                )
                batch_tracker.reset(qpos_batch, qvel_batch)
            else:
                batch_tracker.observe(qpos_batch, qvel_batch)
                if frame % closed_navigations[0].config.replan_frames == 0:
                    batch_tracker.refresh_unexecuted_reference(
                        frame,
                        np.stack([nav.backend.reference for nav in closed_navigations]),
                        unchanged_lookahead_frames=closed_navigations[0].config.lookahead_frames,
                    )
            batch_tracker.update(frame, qpos_batch, qvel_batch)
            for lane, ((_, source_body), source_lane) in enumerate(lane_sources):
                proposal = closed_navigations[lane].commit_batched_action(
                    batch_tracker.action[lane].detach().cpu().numpy()
                )
                warmup_max_target_error = max(
                    warmup_max_target_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(proposal.target_rad)
                                - source_body["joint_target_rad"][frame, source_lane]
                            )
                        )
                    ),
                )
        if warmup_max_target_error > 1e-3:
            raise ValueError("SONIC controller history warmup differs from audited source")
    observed_ball = []
    observed_root = []
    observed_force = []
    pre_step_root = []
    pre_step_joint = []
    pre_step_joint_velocity = []
    pre_step_ball = []
    pre_step_ball_velocity = []
    applied_residual = []
    predicted_baseline_target = []
    applied_phase_offsets = []
    max_parent_target_error_at_snapshot = 0.0
    probe_applied_frames = np.zeros(args.sample_count, dtype=np.int64)
    contact_seen = np.zeros(args.sample_count, dtype=np.bool_)
    phase_first_contact_state = np.full(args.sample_count, -1, dtype=np.int64)
    swing_side = np.full(args.sample_count, -1, dtype=np.int64)
    swing_contact_frame = np.full(args.sample_count, -1, dtype=np.int64)
    swing_contact_delta = np.zeros((args.sample_count, 6), dtype=np.float64)
    swing_foot_positions = []
    swing_linear_jacobians = []
    swing_selected_sides = []
    swing_applied_residuals = []
    swing_baseline_targets = []
    swing_executed_targets = []
    probe_weights = (
        shared_weights if shared_weights is not None else knee_extension_probe_weights()[0]
    )
    probe_joint_indices = [robot.joint_names.index(name) for name in PROBE_JOINT_NAMES]
    probe_joint_limits = (
        robot.data.joint_pos_limits.torch[:, probe_joint_indices].detach().cpu().numpy().copy()
    )
    leg_joint_names = tuple(
        tuple(
            f"{side}_{part}_joint"
            for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        )
        for side in ("left", "right")
    )
    leg_joint_indices = tuple(
        tuple(robot.joint_names.index(name) for name in row) for row in leg_joint_names
    )
    foot_body_indices = tuple(
        robot.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")
    )
    if args.taskspace_forward_m is not None and (
        len(set(leg_joint_indices[0] + leg_joint_indices[1])) != 12
        or robot.data.body_link_jacobian_w.torch.shape[-1] != len(robot.joint_names) + 6
    ):
        raise ValueError("task-space G1 Jacobian or leg joint contract invalid")
    for frame in range(manifest["window_frames"]):
        root_before = robot.data.root_link_pose_w.torch.detach().cpu().numpy().copy()
        joint_before = robot.data.joint_pos.torch[:, indices].detach().cpu().numpy().copy()
        joint_velocity_before = robot.data.joint_vel.torch[:, indices].detach().cpu().numpy().copy()
        ball_before = ball.data.root_pos_w.torch.detach().cpu().numpy().copy()
        ball_velocity_before = ball.data.root_lin_vel_w.torch.detach().cpu().numpy().copy()
        root_before[:, 1] -= lanes
        ball_before[:, 1] -= lanes
        pre_step_root.append(root_before)
        pre_step_joint.append(joint_before)
        pre_step_joint_velocity.append(joint_velocity_before)
        pre_step_ball.append(ball_before)
        pre_step_ball_velocity.append(ball_velocity_before)
        target = robot.data.joint_pos.torch.clone()
        if args.closed_loop_sonic:
            if batch_tracker is None:
                raise RuntimeError("SONIC closed-loop warmup did not initialize")
            absolute_frame = manifest["fixed_start_frame"] + frame
            qpos_rows = []
            qvel_rows = []
            for lane, navigation in enumerate(closed_navigations):
                qpos, qvel = _canonical_state(
                    root_before[lane],
                    robot.data.root_link_vel_w.torch[lane].detach().cpu().numpy(),
                    joint_before[lane],
                    joint_velocity_before[lane],
                    0.0,
                )
                qpos_rows.append(qpos)
                qvel_rows.append(qvel)
                navigation.prepare_batched_proposal(
                    _observation(navigation, absolute_frame, qpos, qvel)
                )
            qpos_batch = np.asarray(qpos_rows)
            qvel_batch = np.asarray(qvel_rows)
            batch_tracker.observe(qpos_batch, qvel_batch)
            if absolute_frame % closed_navigations[0].config.replan_frames == 0:
                batch_tracker.refresh_unexecuted_reference(
                    absolute_frame,
                    np.stack([nav.backend.reference for nav in closed_navigations]),
                    unchanged_lookahead_frames=closed_navigations[0].config.lookahead_frames,
                )
            if args.phase_target_frames is None and selected_phase_targets is None:
                batch_tracker.update(absolute_frame, qpos_batch, qvel_batch)
            else:
                targets = (
                    selected_phase_targets
                    if selected_phase_targets is not None
                    else np.full(args.sample_count, args.phase_target_frames)
                )
                phase_offsets = np.asarray(
                    [
                        (
                            phase_offset_frames(frame, float(value))
                            if args.phase_recovery_frames is None
                            else recovered_phase_offset_frames(
                                frame,
                                float(value),
                                first_foot_contact_frame=(
                                    int(phase_first_contact_state[lane])
                                    if phase_first_contact_state[lane] >= 0
                                    else None
                                ),
                                recovery_frames=args.phase_recovery_frames,
                            )
                        )
                        for lane, value in enumerate(targets)
                    ]
                )
                batch_tracker.update(
                    absolute_frame,
                    qpos_batch,
                    qvel_batch,
                    phase_offsets_frames=phase_offsets,
                )
                applied_phase_offsets.append(phase_offsets)
            for lane, navigation in enumerate(closed_navigations):
                proposal = navigation.commit_batched_action(
                    batch_tracker.action[lane].detach().cpu().numpy()
                )
                target[lane, indices] = torch.as_tensor(
                    proposal.target_rad, device=sim.device, dtype=torch.float32
                )
            if frame == 0:
                max_parent_target_error_at_snapshot = float(
                    np.max(
                        np.abs(
                            target[:, indices].detach().cpu().numpy()
                            - snapshots["privileged_parent_joint_targets_rad"][:, 0]
                        )
                    )
                )
                if max_parent_target_error_at_snapshot > 1e-3:
                    raise ValueError("restored SONIC target differs from authenticated parent")
        else:
            target[:, indices] = torch.as_tensor(
                snapshots["privileged_parent_joint_targets_rad"][:, frame],
                device=sim.device,
                dtype=torch.float32,
            )
        predicted_baseline_target.append(target[:, indices].detach().cpu().numpy().copy())
        base_target = target.clone()
        frame_residual = np.zeros((args.sample_count, len(PROBE_JOINT_NAMES)))
        if args.knee_extension_probe or shared_weights is not None:
            limits = robot.data.joint_pos_limits.torch.detach().cpu().numpy()
            for lane in range(args.sample_count):
                if contact_seen[lane]:
                    continue
                residual = temporal_residual(
                    probe_weights,
                    ball_relative_xyz_m=tuple(
                        float(v) for v in ball_before[lane] - root_before[lane, :3]
                    ),
                    ball_vx_m_s=float(ball_velocity_before[lane, 0]),
                    joint_position_rad=joint_before[lane],
                    joint_velocity_rad_s=joint_velocity_before[lane],
                )
                if not np.any(residual):
                    continue
                for output_index, (joint_index, value) in enumerate(
                    zip(probe_joint_indices, residual, strict=True)
                ):
                    projected_target, _projected = project_residual_target(
                        float(base_target[lane, joint_index]),
                        float(value),
                        float(limits[lane, joint_index, 0]),
                        float(limits[lane, joint_index, 1]),
                    )
                    target[lane, joint_index] = projected_target
                    frame_residual[lane, output_index] = projected_target - float(
                        base_target[lane, joint_index]
                    )
                probe_applied_frames[lane] += 1
        if args.taskspace_forward_m is not None:
            feet = (
                robot.data.body_link_pos_w.torch[:, foot_body_indices].detach().cpu().numpy().copy()
            )
            full_jacobian = robot.data.body_link_jacobian_w.torch.detach().cpu().numpy()
            jacobians = np.stack(
                [
                    np.take(
                        full_jacobian[:, body_index, :3, :],
                        np.asarray(leg_joint_indices[side]) + 6,
                        axis=-1,
                    )
                    for side, body_index in enumerate(foot_body_indices)
                ],
                axis=1,
            )
            if jacobians.shape != (args.sample_count, 2, 3, 6):
                raise ValueError("task-space swing Jacobian shape changed")
            limits = robot.data.joint_pos_limits.torch.detach().cpu().numpy()
            baseline = base_target.detach().cpu().numpy()
            frame_swing_residual = np.zeros((args.sample_count, len(robot.joint_names)))
            for lane in range(args.sample_count):
                if not selected_taskspace_mask[lane]:
                    continue
                if swing_contact_frame[lane] < 0:
                    swing_side[lane] = choose_swing_side(
                        feet[lane],
                        ball.data.root_pos_w.torch[lane].detach().cpu().numpy(),
                        int(swing_side[lane]),
                        acquisition_max_gap_m=args.taskspace_acquisition_max_gap_m,
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
                        ball.data.root_pos_w.torch[lane].detach().cpu().numpy(),
                        jacobians[lane, side],
                        baseline[lane, joint_ids],
                        limits[lane, joint_ids],
                        forward_cap_m=args.taskspace_forward_m,
                        lateral_cap_m=(
                            0.10
                            if selected_taskspace_actions[lane] == 2
                            else args.taskspace_lateral_cap_m
                        ),
                        vertical_offset_m=(
                            0.04
                            if selected_taskspace_actions[lane] == 1
                            else args.taskspace_vertical_offset_m
                        ),
                    )
                target[lane, joint_ids] = torch.as_tensor(
                    baseline[lane, joint_ids] + delta, device=sim.device, dtype=torch.float32
                )
                frame_swing_residual[lane, joint_ids] = (
                    target[lane, joint_ids].detach().cpu().numpy() - baseline[lane, joint_ids]
                )
            swing_foot_positions.append(feet)
            swing_linear_jacobians.append(jacobians)
            swing_selected_sides.append(swing_side.copy())
            swing_applied_residuals.append(frame_swing_residual)
            swing_baseline_targets.append(baseline)
            swing_executed_targets.append(target.detach().cpu().numpy().copy())
        applied_residual.append(frame_residual)
        peak_force = torch.zeros((args.sample_count, 6), device=sim.device)
        for _ in range(10):
            robot.set_joint_position_target_index(target=target)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            ball.update(sim.get_physics_dt())
            for lane, contact in enumerate(contacts):
                contact.update(sim.get_physics_dt())
                matrix = contact.data.force_matrix_w
                if matrix is None or matrix.torch.shape != (1, 1, 6, 3):
                    raise ValueError("snapshot replay contact filter invalid")
                peak_force[lane] = torch.maximum(
                    peak_force[lane], torch.linalg.vector_norm(matrix.torch[0, 0], dim=-1)
                )
                if (
                    (args.knee_extension_probe or shared_weights is not None)
                    and not contact_seen[lane]
                    and bool(torch.any(torch.linalg.vector_norm(matrix.torch[0, 0], dim=-1) > 1.0))
                ):
                    contact_seen[lane] = True
                    target[lane, probe_joint_indices] = base_target[lane, probe_joint_indices]
        if args.phase_recovery_frames is not None:
            contact_now = peak_force.detach().cpu().numpy()
            for lane in range(args.sample_count):
                if phase_first_contact_state[lane] != -1:
                    continue
                bodies = set(np.flatnonzero(contact_now[lane] > 1.0).tolist())
                if bodies:
                    phase_first_contact_state[lane] = frame if bodies <= {0, 1} else -2
        if args.taskspace_forward_m is not None:
            contact_now = peak_force.detach().cpu().numpy()
            last_delta = swing_applied_residuals[-1]
            for lane in range(args.sample_count):
                if swing_contact_frame[lane] >= 0:
                    continue
                if np.any(contact_now[lane] > 1.0):
                    swing_contact_frame[lane] = frame
                    if swing_side[lane] >= 0:
                        joint_ids = list(leg_joint_indices[int(swing_side[lane])])
                        swing_contact_delta[lane] = last_delta[lane, joint_ids]
        observed_ball.append(ball.data.root_pos_w.torch.detach().cpu().numpy().copy())
        observed_root.append(robot.data.root_link_pose_w.torch.detach().cpu().numpy().copy())
        observed_force.append(peak_force.detach().cpu().numpy().copy())
    observed_ball_arr = np.asarray(observed_ball)
    observed_root_arr = np.asarray(observed_root)
    observed_force_arr = np.asarray(observed_force)
    observed_ball_arr[:, :, 1] -= lanes[None, :]
    observed_root_arr[:, :, 1] -= lanes[None, :]
    reference_ball = snapshots["reference_ball_position_local_m"].transpose(1, 0, 2)
    reference_root = snapshots["reference_root_pose_local_xyzw_m"].transpose(1, 0, 2)
    reference_force = snapshots["reference_contact_force_n"].transpose(1, 0, 2)
    precontact = manifest["lead_frames"]
    ball_error = np.linalg.norm(observed_ball_arr - reference_ball, axis=2)
    root_error = np.linalg.norm(observed_root_arr[:-1, :, :3] - reference_root[1:, :, :3], axis=2)
    rows = []
    for lane in range(args.sample_count):
        source = manifest["snapshots"][args.start_index + lane]
        observed = np.flatnonzero(np.max(observed_force_arr[:, lane], axis=1) > 1.0)
        expected = np.flatnonzero(np.max(reference_force[:, lane], axis=1) > 1.0)
        observed_first = int(observed[0]) if len(observed) else None
        expected_first = int(expected[0]) if len(expected) else None
        observed_bodies = np.flatnonzero(np.max(observed_force_arr[:, lane], axis=0) > 1.0).tolist()
        expected_bodies = np.flatnonzero(np.max(reference_force[:, lane], axis=0) > 1.0).tolist()
        rows.append(
            {
                "source_report_hash": source["source_report_hash"],
                "source_lane": source["lane"],
                "observed_first_contact_offset": observed_first,
                "reference_first_contact_offset": expected_first,
                "observed_contact_body_indices": observed_bodies,
                "reference_contact_body_indices": expected_bodies,
                "precontact_max_ball_position_error_m": float(
                    np.max(ball_error[:precontact, lane])
                ),
                "precontact_max_root_position_error_m": float(
                    np.max(root_error[:precontact, lane])
                ),
            }
        )
    args.output_dir.mkdir(parents=True)
    trace_path = args.output_dir / "replay.npz"
    trace_values = dict(
        initial_root_pose_local_xyzw_m=initial_root_pose_local,
        initial_root_velocity_world=initial_root_velocity,
        initial_joint_position_rad=initial_joint_position,
        initial_joint_velocity_rad_s=initial_joint_velocity,
        initial_ball_position_local_m=initial_ball_position_local,
        initial_ball_velocity_world=initial_ball_velocity,
        pre_step_root_pose_local_xyzw_m=np.asarray(pre_step_root),
        pre_step_joint_position_rad=np.asarray(pre_step_joint),
        pre_step_joint_velocity_rad_s=np.asarray(pre_step_joint_velocity),
        pre_step_ball_position_local_m=np.asarray(pre_step_ball),
        pre_step_ball_linear_velocity_m_s=np.asarray(pre_step_ball_velocity),
        applied_probe_residual_rad=np.asarray(applied_residual),
        predicted_baseline_joint_target_rad=np.asarray(predicted_baseline_target),
        probe_joint_position_limits_rad=probe_joint_limits,
        observed_ball_position_local_m=observed_ball_arr,
        observed_root_pose_local_xyzw_m=observed_root_arr,
        observed_ball_body_contact_force_peak_n=observed_force_arr,
    )
    if args.phase_target_frames is not None or selected_phase_targets is not None:
        trace_values["applied_sonic_phase_offset_frames"] = np.asarray(applied_phase_offsets)
    if args.taskspace_forward_m is not None:
        trace_values.update(
            pre_step_foot_link_position_w=np.asarray(swing_foot_positions),
            pre_step_foot_linear_jacobian_w=np.asarray(swing_linear_jacobians),
            taskspace_selected_side=np.asarray(swing_selected_sides),
            applied_taskspace_joint_delta_rad=np.asarray(swing_applied_residuals),
            baseline_taskspace_joint_target_rad=np.asarray(swing_baseline_targets),
            executed_taskspace_joint_target_rad=np.asarray(swing_executed_targets),
            taskspace_joint_limits_rad=robot.data.joint_pos_limits.torch.detach().cpu().numpy(),
        )
    np.savez_compressed(trace_path, **trace_values)
    report = {
        "schema": "rsi_isaac_first_touch_snapshot_replay_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "privileged_future_targets_diagnostic_only": True,
        "snapshot_bank_manifest_hash": bank_audit["manifest_hash"],
        "runner_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "foundation_qualification_hash": navigation.backend.qualification.qualification_hash,
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "start_index": args.start_index,
        "sample_count": args.sample_count,
        "window_frames": manifest["window_frames"],
        "knee_extension_probe": args.knee_extension_probe,
        "shared_candidate_hash": shared_candidate_hash,
        "phase_target_frames": args.phase_target_frames,
        "phase_ramp_frames": 20
        if args.phase_target_frames is not None or selected_phase_targets is not None
        else None,
        "phase_recovery_frames": args.phase_recovery_frames,
        "taskspace_forward_m": args.taskspace_forward_m,
        "taskspace_lateral_cap_m": args.taskspace_lateral_cap_m,
        "taskspace_vertical_offset_m": args.taskspace_vertical_offset_m,
        "taskspace_acquisition_max_gap_m": args.taskspace_acquisition_max_gap_m,
        "taskspace_leg_joint_names": [list(row) for row in leg_joint_names]
        if args.taskspace_forward_m is not None
        else None,
        "taskspace_joint_order": list(robot.joint_names)
        if args.taskspace_forward_m is not None
        else None,
        "taskspace_gate_actor_hash": taskspace_gate_hash,
        "taskspace_family_actor_hash": taskspace_family_hash,
        "selected_taskspace_actions": selected_taskspace_actions.tolist()
        if taskspace_family_hash is not None
        else None,
        "selected_taskspace_mask": selected_taskspace_mask.tolist()
        if args.taskspace_forward_m is not None
        else None,
        "contextual_phase_actor_hash": contextual_phase_hash,
        "local_phase_actor_hash": local_phase_hash,
        "selected_phase_targets_frames": (
            selected_phase_targets.tolist() if selected_phase_targets is not None else None
        ),
        "probe_joint_limits_recorded": True,
        "closed_loop_sonic": args.closed_loop_sonic,
        "warmup_max_target_error_rad": warmup_max_target_error,
        "max_parent_target_error_at_snapshot_rad": max_parent_target_error_at_snapshot,
        "probe_applied_frames": probe_applied_frames.tolist(),
        "initial_state_max_absolute_error": initial_state_error,
        "rows": rows,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_ISAAC_SNAPSHOT_REPLAY="
        + json.dumps(
            {
                "report_hash": report["report_hash"],
                "sample_count": args.sample_count,
                "max_precontact_ball_error_m": float(np.max(ball_error[:precontact])),
                "max_precontact_root_error_m": float(np.max(root_error[:precontact])),
            }
        ),
        flush=True,
    )


try:
    main()
except Exception as exc:
    print("RSI_ISAAC_SNAPSHOT_REPLAY_FAILURE=" + repr(exc), flush=True)
    traceback.print_exc()
    # Kit may convert an uncaught Python exception into process exit 0. A
    # failed physical replay must never look successful to orchestration.
    os._exit(1)
else:
    simulation_app.close()
