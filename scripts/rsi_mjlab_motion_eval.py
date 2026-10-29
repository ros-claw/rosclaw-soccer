"""First-episode frozen G1 motion tracking evaluation, simulation only."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest, tracked_reference
from scripts.rsi_native_fresh_common import clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", required=True, type=Path)
    p.add_argument("--reference-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-hash", required=True)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    clean(a.expected_head)
    root = a.reference_root.resolve()
    if a.output.resolve().is_relative_to(Path.cwd()) or a.output.resolve().is_relative_to(root):
        raise ValueError("external output required")
    if file_digest(a.protocol) != a.protocol_hash or file_digest(a.checkpoint) != a.checkpoint_hash:
        raise ValueError("exact protocol and checkpoint hashes required")
    pins = tracked_reference(root, a.reference_head)
    motions = sorted((root / "src/assets/soccer/motions/shooter").glob("*.npz"))
    if len(motions) != 13 or any(str(path) not in pins for path in motions):
        raise ValueError("thirteen pinned reference motions required")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg
    from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage1_env_cfg

    from rosclaw_soccer.training.continuous_competitive_match_growth import _write_trajectory

    cfg = unitree_g1_stage1_env_cfg()
    cfg.scene.num_envs, cfg.seed = 16, 20261030
    cfg.commands["motion"].motion_dir = str(motions[0].parent)
    cfg.commands["motion"].sampling_mode = "uniform"
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    agent = unitree_g1_soccer_ppo_runner_cfg()
    agent.logger, agent.upload_model = "tensorboard", False
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            reference_head=a.reference_head,
            reference_files=pins,
            checkpoint_hash=a.checkpoint_hash,
            protocol_hash=a.protocol_hash,
            worlds=16,
            max_controls=150,
            training_motions=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    try:
        torch.manual_seed(20261030)
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        if env.action_manager.total_action_dim != 29:
            raise ValueError("29 joint-position actions required")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(
            wrapped, asdict(agent), str(a.output / "logs"), device="cuda:0"
        )
        checkpoint = torch.load(a.checkpoint, map_location="cuda:0", weights_only=True)
        runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        runner.alg.actor.eval()
        torch.manual_seed(20261030)
        obs, _ = wrapped.reset()
        robot = env.scene["robot"].data
        motion = env.command_manager.get_term("motion")

        def tensor(value):
            return value.detach().cpu().numpy().copy()

        trace = dict(
            initial_actor_obs=tensor(obs["actor"]),
            initial_root_pose=tensor(robot.root_link_pose_w),
            initial_root_velocity=tensor(robot.root_link_vel_w),
            initial_joint_pos=tensor(robot.joint_pos),
            initial_joint_vel=tensor(robot.joint_vel),
            initial_motion_id=tensor(motion.motion_idx),
            initial_motion_step=tensor(motion.time_steps),
        )
        rows = {
            name: []
            for name in (
                "valid_before",
                "done",
                "reward",
                "root_pose",
                "joint_pos",
                "joint_vel",
                "motion_id",
                "motion_step",
                "error_body_pos",
                "error_joint_pos",
            )
        }
        alive = torch.ones(16, dtype=torch.bool, device="cuda:0")
        with torch.no_grad():
            for _ in range(150):
                action = runner.alg.actor(obs)
                obs, reward, done, _ = wrapped.step(action)
                rows["valid_before"].append(tensor(alive))
                rows["done"].append(tensor(done))
                rows["reward"].append(tensor(reward))
                rows["root_pose"].append(tensor(robot.root_link_pose_w))
                rows["joint_pos"].append(tensor(robot.joint_pos))
                rows["joint_vel"].append(tensor(robot.joint_vel))
                rows["motion_id"].append(tensor(motion.motion_idx))
                rows["motion_step"].append(tensor(motion.time_steps))
                rows["error_body_pos"].append(tensor(motion.metrics["error_body_pos"]))
                rows["error_joint_pos"].append(tensor(motion.metrics["error_joint_pos"]))
                alive &= ~done.bool()
        trace.update({k: np.stack(v) for k, v in rows.items()})
        artifact = _write_trajectory(a.output / "trajectory.npz", trace)
        if (
            file_digest(a.protocol) != a.protocol_hash
            or file_digest(a.checkpoint) != a.checkpoint_hash
            or tracked_reference(root, a.reference_head) != pins
        ):
            raise ValueError("pinned input changed")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="STAGE1_FROZEN_MOTION_EVALUATION_ONLY",
                source_head=a.expected_head,
                declaration_hash=file_digest(a.output / "declaration.json"),
                checkpoint_hash=a.checkpoint_hash,
                artifact=artifact,
                world_steps=16 * 150,
                first_episode_survivors=int(alive.sum().item()),
                hardware_authorized=False,
                promotion_authorized=False,
                soccer_verified=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error_type=type(error).__name__, error=str(error), hardware_authorized=False),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
