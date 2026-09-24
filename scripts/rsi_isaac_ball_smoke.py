"""Bounded Isaac Lab SIM_ONLY ball-contact smoke, not an RSI-M0 training task.

Run through Isaac Sim's python.sh with Isaac Lab source on PYTHONPATH. The
script creates only simulation prims and never opens a robot/ROS transport.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.num_envs <= 128 or not 1 <= args.steps <= 1000:
    parser.error("bounded num-envs and steps required")
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def main() -> None:
    radius = 0.11
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=1.0 / 120.0))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    for index in range(args.num_envs):
        sim_utils.create_prim(
            f"/World/Env{index}",
            "Xform",
            translation=(float(index % 8) * 4, float(index // 8) * 4, 0.0),
        )
    cfg = RigidObjectCfg(
        prim_path="/World/Env.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, radius + 0.02)),
    )
    ball = RigidObject(cfg=cfg)
    sim.reset()
    pose = ball.data.default_root_pose.torch.clone()
    for index in range(args.num_envs):
        pose[index, 0] += (index % 8) * 4
        pose[index, 1] += (index // 8) * 4
    ball.write_root_pose_to_sim_index(root_pose=pose)
    velocity = ball.data.default_root_vel.torch.clone()
    velocity[:, 0] = 2.0
    velocity[:, 1] = torch.linspace(-0.5, 0.5, args.num_envs, device=sim.device)
    ball.write_root_velocity_to_sim_index(root_velocity=velocity)
    ball.reset()
    for _ in range(args.steps):
        ball.write_data_to_sim()
        sim.step()
        ball.update(sim.get_physics_dt())
    positions = ball.data.root_pos_w.torch
    linear_velocity = ball.data.root_lin_vel_w.torch
    angular_velocity = ball.data.root_ang_vel_w.torch
    displacement = positions[:, 0] - pose[:, 0]
    result = {
        "schema": "rosclaw_soccer.rsi.isaac_ball_smoke.v1",
        "activation_ceiling": "SIM_ONLY",
        "num_envs": ball.num_instances,
        "steps": args.steps,
        "min_ball_height_m": float(positions[:, 2].min().item()),
        "max_ball_height_m": float(positions[:, 2].max().item()),
        "mean_final_x_velocity_mps": float(linear_velocity[:, 0].mean().item()),
        "min_forward_displacement_m": float(displacement.min().item()),
        "mean_angular_speed_rad_s": float(
            torch.linalg.vector_norm(angular_velocity, dim=1).mean().item()
        ),
        "finite": bool(torch.isfinite(positions).all() and torch.isfinite(linear_velocity).all()),
    }
    print("RSI_ISAAC_BALL_SMOKE=" + json.dumps(result, sort_keys=True))
    if (
        not result["finite"]
        or ball.num_instances != args.num_envs
        or result["min_forward_displacement_m"] <= 0.5
        or not radius - 0.03 <= result["min_ball_height_m"] <= radius + 0.03
        or result["mean_angular_speed_rad_s"] <= 0.1
    ):
        raise RuntimeError("Isaac ball smoke failed contact, rolling or finite check")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RSI_ISAAC_BALL_ERROR={type(exc).__name__}:{exc}", flush=True)
        os._exit(2)  # Kit's close() exits 0 and would hide the failed smoke.
    simulation_app.close()
