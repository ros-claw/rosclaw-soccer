"""One-GPU Stage II MLP transfer and PPO physics smoke, SIM only."""

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
        raise ValueError("thirteen pinned motion references required")
    sys.path.insert(0, str(root))
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from rsl_rl.models.mlp_model import MLPModel
    from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg
    from src.tasks.soccer.config.g1.training_env_cfgs import (
        unitree_g1_stage1_env_cfg,
        unitree_g1_stage2_env_cfg,
    )
    from tensordict import TensorDict

    from scripts.rsi_mjlab_mlp_transfer import expand_mlp_state

    first, cfg = unitree_g1_stage1_env_cfg(), unitree_g1_stage2_env_cfg()
    for group in ("actor", "critic"):
        common = first.observations[group].terms
        extended = cfg.observations[group].terms
        if list(extended) != [*common, "target_point_pos", "target_destination_pos"]:
            raise ValueError("expected stage observation ordering required")
        if any(asdict(common[k]) != asdict(extended[k]) for k in common):
            raise ValueError("common observation term config mismatch")
    if asdict(first.actions["joint_pos"]) != asdict(cfg.actions["joint_pos"]):
        raise ValueError("joint action config mismatch")
    cfg.scene.num_envs, cfg.seed = 16, 20261029
    cfg.commands["motion"].motion_dir = str(motions[0].parent)
    agent = unitree_g1_soccer_ppo_runner_cfg()
    agent.logger, agent.upload_model, agent.num_steps_per_env = "tensorboard", False, 24
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            reference_head=a.reference_head,
            reference_files=pins,
            checkpoint_hash=a.checkpoint_hash,
            protocol_hash=a.protocol_hash,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    try:
        torch.manual_seed(20261029)
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        if env.action_manager.total_action_dim != 29:
            raise ValueError("expected 29 joint-position actions")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(
            wrapped, asdict(agent), str(a.output / "logs"), device="cuda:0"
        )
        obs = wrapped.get_observations()
        checkpoint = torch.load(a.checkpoint, map_location="cuda:0", weights_only=True)
        for group, old_dim, new_dim in (("actor", 154, 160), ("critic", 292, 298)):
            model = getattr(runner.alg, group)
            if obs[group].shape != (16, new_dim):
                raise ValueError("unexpected actual observation dimensions")
            source = checkpoint[f"{group}_state_dict"]
            state = expand_mlp_state(
                source,
                model.state_dict(),
                old_features=tuple(map(str, range(old_dim))),
                new_features=tuple(map(str, range(new_dim))),
            )
            model.load_state_dict(state, strict=True)
        old_obs = TensorDict({"actor": obs["actor"][:, :154]}, batch_size=[16])
        source_actor = MLPModel(
            old_obs,
            {"actor": ["actor"]},
            "actor",
            29,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=dict(
                class_name="GaussianDistribution", init_std=1.0, std_type="scalar"
            ),
        ).to("cuda:0")
        source_actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        with torch.no_grad():
            difference = (source_actor(old_obs) - runner.alg.actor(obs)).abs().max().item()
        if difference > 1e-6:
            raise ValueError("expanded policy differs on identical actual observation prefix")
        before = {
            name: value.detach().clone() for name, value in runner.alg.actor.named_parameters()
        }
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        torch.cuda.synchronize()
        changed = [
            name
            for name, value in runner.alg.actor.named_parameters()
            if not torch.equal(before[name], value)
        ]
        if not changed or env._sim_step_counter != 192:
            raise ValueError("actual physics and changed actor parameters required")
        dest = a.output / "candidate.pt"
        runner.save(str(dest))
        if (
            tracked_reference(root, a.reference_head) != pins
            or file_digest(a.checkpoint) != a.checkpoint_hash
            or file_digest(a.protocol) != a.protocol_hash
        ):
            raise ValueError("pinned input changed")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="STAGE2_MLP_TRANSFER_PHYSICS_SMOKE_ONLY",
                source_head=a.expected_head,
                reference_head=a.reference_head,
                declaration_hash=file_digest(a.output / "declaration.json"),
                parent_checkpoint_hash=a.checkpoint_hash,
                candidate_hash=file_digest(dest),
                actual_observation_max_action_difference=difference,
                changed_actor_parameters=changed,
                native_steps=env._sim_step_counter,
                world_steps=16 * 24 * 2,
                kick_skill_proven=False,
                team_game_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
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
