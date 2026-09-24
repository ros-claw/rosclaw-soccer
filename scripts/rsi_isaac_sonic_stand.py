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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    not args.g1_usd.is_file()
    or args.g1_usd.stat().st_size < 1000
    or not args.model_root.is_dir()
    or not 50 <= args.frames <= 300
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
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES  # noqa: E402
from rosclaw_soccer.providers.g1.sonic_navigation import (  # noqa: E402
    G1SonicNavigation,
    SonicNavigationConfig,
)
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController  # noqa: E402
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json  # noqa: E402
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation  # noqa: E402


def main() -> None:
    source_hash = hash_bytes(Path(__file__).read_bytes())
    asset_hash = hash_bytes(args.g1_usd.read_bytes())
    navigation = G1SonicNavigation(
        args.model_root,
        "blue.playmaker",
        SonicNavigationConfig(maximum_frames=args.frames, model_variant="low_latency"),
    )
    qualification = navigation.backend.qualification
    qualification.require_eligible()
    names = tuple(G1_DDS_JOINT_NAMES)
    kp = np.asarray(navigation.backend.kp, dtype=np.float64)
    kd = np.asarray(navigation.backend.kd, dtype=np.float64)
    effort = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    if len(names) != 29 or kp.shape != (29,) or kd.shape != (29,) or effort.shape != (29,):
        raise ValueError("SONIC joint/gain/effort shape changed")
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=0.002))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    sim_utils.create_prim("/World/Env0", "Xform")
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/Env.*/G1"
    robot_cfg.spawn.usd_path = str(args.g1_usd.resolve())
    robot_cfg.actuators = {
        "sonic": ImplicitActuatorCfg(
            joint_names_expr=list(names),
            effort_limit_sim={
                name: float(value) for name, value in zip(names, effort, strict=True)
            },
            stiffness={name: float(value) for name, value in zip(names, kp, strict=True)},
            damping={name: float(value) for name, value in zip(names, kd, strict=True)},
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
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, 0.13)),
        )
    )
    sim.reset()
    if set(robot.joint_names) != set(names) or len(robot.joint_names) != 29:
        raise ValueError("Isaac 29-DoF joint names differ from SONIC")
    indices = [robot.joint_names.index(name) for name in names]
    initial_joint = robot.data.default_joint_pos.torch.clone()
    initial_joint[0, indices] = torch.as_tensor(
        G1SonicRunupController.default_angles, device=sim.device, dtype=initial_joint.dtype
    )
    pose = robot.data.default_root_pose.torch.clone()
    pose[0, :3] = torch.tensor((0.0, 0.0, 0.793), device=sim.device)
    pose[0, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=sim.device)
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=initial_joint)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()
    ball.write_root_pose_to_sim_index(root_pose=ball.data.default_root_pose.torch.clone())
    ball.reset()
    qpos_rows = []
    qvel_rows = []
    target_rows = []
    for frame in range(args.frames):
        root = robot.data.root_state_w.torch[0].detach().cpu().numpy().astype(np.float64)
        joint = robot.data.joint_pos.torch[0, indices].detach().cpu().numpy().astype(np.float64)
        velocity = robot.data.joint_vel.torch[0, indices].detach().cpu().numpy().astype(np.float64)
        qpos = np.concatenate((root[:7], joint, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
        qvel = np.concatenate((root[7:13], velocity, np.zeros(6)))
        if qpos.shape != (43,) or qvel.shape != (41,) or not np.isfinite(qpos).all():
            raise ValueError("Isaac body observation is invalid")
        obs = TeamMotorObservation(
            agent_id="blue.playmaker",
            frame=frame,
            time_sec=frame * 0.02,
            intent="other",
            prospective_owner=False,
            qpos=tuple(float(value) for value in qpos),
            qvel=tuple(float(value) for value in qvel),
            target_position_m=(0.0, 0.0, 0.0),
            navigation_command=(0.0, 0.0, 0.0),
            navigation_envelope=navigation.navigation_envelope,
        )
        if frame == 0:
            navigation.start_from_observation(obs)
        proposal = navigation.propose(obs)
        target = robot.data.joint_pos.torch.clone()
        target[0, indices] = torch.as_tensor(
            proposal.target_rad, device=sim.device, dtype=target.dtype
        )
        for _ in range(10):
            robot.set_joint_position_target_index(target=target)
            robot.write_data_to_sim()
            ball.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            ball.update(sim.get_physics_dt())
        qpos_rows.append(qpos[:36].copy())
        qvel_rows.append(qvel[:35].copy())
        target_rows.append(np.asarray(proposal.target_rad, dtype=np.float64))
    arrays = {
        "qpos": np.asarray(qpos_rows),
        "qvel": np.asarray(qvel_rows),
        "target": np.asarray(target_rows),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite Isaac SONIC trajectory")
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
    heights = arrays["qpos"][:, 2]
    report = {
        "schema": "rosclaw_soccer.rsi.isaac_sonic_stand.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "asset_hash": asset_hash,
        "model_hash": qualification.qualification_hash,
        "joint_names": list(names),
        "joint_map_hash": hash_json(list(names)),
        "gain_hash": hash_json({"kp": kp.tolist(), "kd": kd.tolist()}),
        "frames": args.frames,
        "min_pelvis_height_m": float(heights.min()),
        "final_pelvis_height_m": float(heights[-1]),
        "max_displacement_m": float(np.max(np.linalg.norm(arrays["qpos"][:, :2], axis=1))),
        "ball_final_height_m": float(ball.data.root_pos_w.torch[0, 2].item()),
        "trajectory_hash": hash_bytes((args.output_dir / "trajectory.npz").read_bytes()),
        "stand_passed": bool(heights.min() >= 0.55 and math.isfinite(float(heights[-1]))),
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
        print(f"RSI_ISAAC_SONIC_ERROR={type(exc).__name__}:{exc}", flush=True)
        os._exit(2)
    simulation_app.close()
