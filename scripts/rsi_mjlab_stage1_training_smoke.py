"""Bounded SIM-only G1 MJWarp motion-tracking PPO integration smoke."""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_fresh_common import clean


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def tracked_reference(root, head):
    if subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip() != head or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("clean exact reference checkout required")
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--", "src"]).split(
        b"\0"
    )
    files = [root / name.decode() for name in names if name]
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("ordinary tracked reference source/assets required")
    return {str(path): file_digest(path) for path in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--reference-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--measure-normalizer-drift", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    root = a.reference_root.resolve()
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.output.resolve().is_relative_to(root)
        or file_digest(a.protocol) != a.protocol_hash
    ):
        raise ValueError("unused external output and exact predeclared protocol required")
    pins = tracked_reference(root, a.reference_head)
    motions = sorted((root / "src/assets/soccer/motions/shooter").glob("*.npz"))
    if len(motions) != 13 or any(str(path) not in pins for path in motions):
        raise ValueError("all thirteen tracked reference motions required")
    sys.path.insert(0, str(root))
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg
    from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage1_env_cfg

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one explicitly isolated CUDA device required")
    cfg = unitree_g1_stage1_env_cfg()
    cfg.scene.num_envs = 16
    cfg.seed = 20261029
    cfg.commands["motion"].motion_dir = str(motions[0].parent)
    agent = unitree_g1_soccer_ppo_runner_cfg()
    agent.seed = 20261029
    agent.num_steps_per_env = 24
    agent.max_iterations = 2
    # Local-only logs; the reference default WandB adapter is incompatible
    # with the installed WandB Settings API and is unnecessary for this smoke.
    agent.logger = "tensorboard"
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            reference_head=a.reference_head,
            source_head=a.expected_head,
            reference_files=pins,
            protocol_hash=a.protocol_hash,
            device=torch.cuda.get_device_name(0),
            environments=16,
            steps_per_iteration=24,
            iterations=2,
            measure_normalizer_drift=a.measure_normalizer_drift,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        if env.action_manager.total_action_dim != 29:
            raise ValueError("actual 29-joint action surface required")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(
            wrapped, asdict(agent), str(a.output / "logs"), device="cuda:0"
        )
        drift = []
        if a.measure_normalizer_drift:
            original_update = runner.alg.update

            def update_with_drift_measurement():
                storage = runner.alg.storage
                old_mean, old_std = storage.distribution_params
                with torch.no_grad():
                    observations = storage.observations.flatten(0, 1)
                    mean = runner.alg.actor(observations).reshape_as(old_mean)
                    std = runner.alg.actor.distribution.std_param.expand_as(mean)
                    old_dist = torch.distributions.Normal(old_mean, old_std)
                    new_dist = torch.distributions.Normal(mean, std)
                    kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)
                    logp = new_dist.log_prob(storage.actions).sum(-1)
                    delta = (logp - storage.actions_log_prob.squeeze(-1)).abs()
                    if not torch.isfinite(kl).all() or not torch.isfinite(delta).all():
                        raise ValueError("finite within-rollout distribution metrics required")
                    drift.append(
                        dict(
                            samples=kl.numel(),
                            mean_kl=kl.mean().item(),
                            p95_kl=torch.quantile(kl.flatten(), 0.95).item(),
                            max_kl=kl.max().item(),
                            mean_abs_logp_delta=delta.mean().item(),
                            p95_abs_logp_delta=torch.quantile(delta.flatten(), 0.95).item(),
                            max_abs_logp_delta=delta.max().item(),
                        )
                    )
                return original_update()

            runner.alg.update = update_with_drift_measurement
        policy = runner.alg.get_policy()
        before = {name: value.detach().cpu().clone() for name, value in policy.named_parameters()}
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        torch.cuda.synchronize()
        after = {name: value.detach().cpu() for name, value in policy.named_parameters()}
        if before.keys() != after.keys() or any(
            not torch.isfinite(v).all() for v in after.values()
        ):
            raise ValueError("finite actor parameters required")
        changed = [name for name in before if not torch.equal(before[name], after[name])]
        if not changed or env._sim_step_counter != 2 * 24 * cfg.decimation:
            raise ValueError("actual actor update and all declared native steps required")
        checkpoint = a.output / "candidate.pt"
        runner.save(str(checkpoint))
        if (
            tracked_reference(root, a.reference_head) != pins
            or file_digest(a.protocol) != a.protocol_hash
        ):
            raise ValueError("reference or protocol changed during training")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="MJWARP_G1_STAGE1_PPO_TRAINING_SMOKE_ONLY",
                source_head=a.expected_head,
                reference_head=a.reference_head,
                declaration_hash=file_digest(a.output / "declaration.json"),
                checkpoint_hash=file_digest(checkpoint),
                actor_changed_parameters=changed,
                within_rollout_normalizer_drift=drift if a.measure_normalizer_drift else None,
                native_sim_steps=env._sim_step_counter,
                world_steps=16 * 2 * 24,
                hardware_authorized=False,
                promotion_authorized=False,
                kick_skill_proven=False,
                team_game_verified=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
