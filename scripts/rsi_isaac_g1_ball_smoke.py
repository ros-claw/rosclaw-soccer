"""SIM_ONLY G1+ball asset/actuation smoke for the future contact gym.

This checks simulator integration only; it does not prove standing skill,
learning, task success, safety transfer, or R1 eligibility.
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
asset_group = parser.add_mutually_exclusive_group(required=True)
asset_group.add_argument("--g1-usd", type=Path)
asset_group.add_argument("--official-g1", action="store_true")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num-envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.steps <= 1000 or not 1 <= args.num_envs <= 16:
    parser.error("bounded steps and environments are required")
if args.g1_usd is not None and (not args.g1_usd.is_file() or args.g1_usd.stat().st_size < 1000):
    parser.error("a non-pointer local G1 USD is required")
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402


def main() -> None:
    print("RSI_ISAAC_G1_STAGE=build", flush=True)
    from pxr import Usd, UsdPhysics

    if args.g1_usd is not None:
        asset_stage = Usd.Stage.Open(str(args.g1_usd.resolve()))
        asset_roots = [
            str(prim.GetPath())
            for prim in asset_stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        print(
            "RSI_ISAAC_G1_ASSET="
            + json.dumps(
                {
                    "source": "local",
                    "default_prim": str(asset_stage.GetDefaultPrim().GetPath()),
                    "articulation_roots": asset_roots,
                    "prim_count": sum(1 for _ in asset_stage.Traverse()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if len(asset_roots) != 1:
            raise RuntimeError("local G1 USD has no unique articulation root")
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1.0 / 120.0))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    origins = torch.zeros((args.num_envs, 3), device=sim.device)
    for index in range(args.num_envs):
        origins[index, 0] = (index % 4) * 4.0
        origins[index, 1] = (index // 4) * 4.0
        sim_utils.create_prim(
            f"/World/Env{index}",
            "Xform",
            translation=(float(origins[index, 0]), float(origins[index, 1]), 0.0),
        )
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/Env.*/G1"
    if args.g1_usd is not None:
        robot_cfg.spawn.usd_path = str(args.g1_usd.resolve())
        robot_cfg.actuators.pop("hands")  # SONIC's 29-DoF URDF has no finger joints.
    else:
        print(
            "RSI_ISAAC_G1_ASSET="
            + json.dumps({"source": "isaaclab_official", "usd_path": robot_cfg.spawn.usd_path}),
            flush=True,
        )
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 0.0, 0.13)),
        )
    )
    sim.reset()
    print("RSI_ISAAC_G1_STAGE=reset", flush=True)
    robot_pose = robot.data.default_root_pose.torch.clone()
    robot_pose[:, :3] += origins
    robot.write_root_pose_to_sim_index(root_pose=robot_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()
    ball_pose = ball.data.default_root_pose.torch.clone()
    ball_pose[:, :3] += origins
    ball.write_root_pose_to_sim_index(root_pose=ball_pose)
    ball.reset()
    hip_index = robot.joint_names.index("left_hip_pitch_joint")
    targets = robot.data.default_joint_pos.torch.clone()
    targets[:, hip_index] += torch.linspace(-0.25, 0.25, args.num_envs, device=sim.device)
    for _ in range(args.steps):
        robot.set_joint_position_target_index(target=targets)
        robot.write_data_to_sim()
        ball.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        ball.update(sim.get_physics_dt())
    root = robot.data.root_pos_w.torch
    joints = robot.data.joint_pos.torch
    result = {
        "schema": "rosclaw_soccer.rsi.isaac_g1_ball_smoke.v1",
        "activation_ceiling": "SIM_ONLY",
        "asset_source": "local" if args.g1_usd is not None else "isaaclab_official",
        "steps": args.steps,
        "joint_count": int(joints.shape[1]),
        "num_envs": robot.num_instances,
        "min_root_height_m": float(root[:, 2].min().item()),
        "max_root_height_m": float(root[:, 2].max().item()),
        "min_ball_height_m": float(ball.data.root_pos_w.torch[:, 2].min().item()),
        "left_hip_target_spread_rad": float(
            (targets[:, hip_index].max() - targets[:, hip_index].min()).item()
        ),
        "left_hip_final_spread_rad": float(
            (joints[:, hip_index].max() - joints[:, hip_index].min()).item()
        ),
        "finite": bool(torch.isfinite(root).all() and torch.isfinite(joints).all()),
        "task_success": False,
        "trained_actor": False,
    }
    print("RSI_ISAAC_G1_BALL_SMOKE=" + json.dumps(result, sort_keys=True))
    expected_joint_count = 29 if args.g1_usd is not None else 43
    if (
        result["joint_count"] != expected_joint_count
        or robot.num_instances != args.num_envs
        or not result["finite"]
        or args.num_envs >= 2
        and result["left_hip_final_spread_rad"] <= 0.03
    ):
        raise RuntimeError("G1 asset joint mapping or simulation finite check failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RSI_ISAAC_G1_ERROR={type(exc).__name__}:{exc}", flush=True)
        os._exit(2)  # Kit's close() exits 0 and would hide the failed smoke.
    simulation_app.close()
