"""Four-rank, small SIM-only MJWarp G1 PPO synchronization smoke."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest, tracked_reference
from scripts.rsi_native_fresh_common import clean


def validate_ranks(environ):
    rank = int(environ.get("RANK", "-1"))
    local = int(environ.get("LOCAL_RANK", "-1"))
    if (
        (
            environ.get("WORLD_SIZE"),
            environ.get("CUDA_VISIBLE_DEVICES"),
            environ.get("CUDA_DEVICE_ORDER"),
        )
        != ("4", "0,1,2,3", "PCI_BUS_ID")
        or rank != local
        or rank not in range(4)
    ):
        raise ValueError("explicit four local ranks and ordered physical CUDA devices required")
    return rank, local


def validate_gpu_mapping(gpu_map, cuda_uuids):
    if set(gpu_map) != set(range(4)) or len(set(gpu_map.values())) != 4:
        raise ValueError("four uniquely identified local physical GPUs required")
    if cuda_uuids != [gpu_map[i] for i in range(4)]:
        raise ValueError("CUDA to physical GPU UUID mapping mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--reference-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synchronize-normalizers", action="store_true")
    a = p.parse_args()
    clean(a.expected_head)
    rank, local = validate_ranks(os.environ)
    root = a.reference_root.resolve()
    if (
        a.output.resolve().is_relative_to(Path.cwd())
        or a.output.resolve().is_relative_to(root)
        or file_digest(a.protocol) != a.protocol_hash
    ):
        raise ValueError("external output and exact predeclared protocol required")
    pins = tracked_reference(root, a.reference_head)
    motions = sorted((root / "src/assets/soccer/motions/shooter").glob("*.npz"))
    if len(motions) != 13 or any(str(path) not in pins for path in motions):
        raise ValueError("exact thirteen tracked reference motions required")
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    gpu_map = {int(line.split(",")[0].strip()): line.split(",")[1].strip() for line in gpu_lines}
    sys.path.insert(0, str(root))
    import torch
    import torch.distributed as dist
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg
    from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage1_env_cfg

    if torch.cuda.device_count() != 4:
        raise ValueError("four visible CUDA devices required")
    cuda_uuids = [f"GPU-{torch.cuda.get_device_properties(i).uuid}" for i in range(4)]
    validate_gpu_mapping(gpu_map, cuda_uuids)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "mjlab", "rsl-rl-lib", "warp-lang")
    }
    device = f"cuda:{local}"
    torch.cuda.set_device(local)
    torch.manual_seed(20261029 + rank)
    cfg = unitree_g1_stage1_env_cfg()
    cfg.scene.num_envs, cfg.seed = 16, 20261029 + rank
    cfg.commands["motion"].motion_dir = str(motions[0].parent)
    agent = unitree_g1_soccer_ppo_runner_cfg()
    agent.seed, agent.num_steps_per_env, agent.max_iterations = 20261029 + rank, 24, 2
    agent.logger = "tensorboard"
    agent.upload_model = False
    folder = a.output / f"rank-{rank}"
    folder.mkdir(parents=True, exist_ok=False)
    write_once(
        folder / "declaration.json",
        dict(
            rank=rank,
            local_rank=local,
            physical_gpu_index=local,
            gpu_uuid=gpu_map[local],
            cuda_uuids=cuda_uuids,
            versions=versions,
            source_head=a.expected_head,
            reference_head=a.reference_head,
            reference_files=pins,
            protocol_hash=a.protocol_hash,
            environments=16,
            steps_per_iteration=24,
            iterations=2,
            synchronize_normalizers=a.synchronize_normalizers,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        if env.action_manager.total_action_dim != 29:
            raise ValueError("actual 29-joint action surface required")
        runner = MjlabOnPolicyRunner(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions),
            asdict(agent),
            str(folder / "logs"),
            device=device,
        )
        if not dist.is_initialized() or dist.get_world_size() != 4 or dist.get_rank() != rank:
            raise ValueError("NCCL initialized with exact ranks required")
        if dist.get_backend() != "nccl":
            raise ValueError("NCCL backend required")
        if a.synchronize_normalizers:
            from scripts.rsi_sync_normalizer import install_four_rank_normalizers

            install_four_rank_normalizers(runner.alg)
        runner.alg.broadcast_parameters()
        policy = runner.alg.get_policy()
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        torch.cuda.synchronize()
        after = {name: value.detach() for name, value in policy.named_parameters()}
        if before.keys() != after.keys() or any(
            not torch.isfinite(v).all() for v in after.values()
        ):
            raise ValueError("finite actor parameters required")
        changed = [name for name in before if not torch.equal(before[name], after[name])]
        if not changed or env._sim_step_counter != 192:
            raise ValueError("actual actor update and exact physics steps required")
        vector = torch.cat([v.flatten() for v in after.values()])
        gathered = [torch.empty_like(vector) for _ in range(4)]
        dist.all_gather(gathered, vector)
        if any(not torch.equal(vector, other) for other in gathered):
            raise ValueError("post-update actor trainable parameters differ across ranks")
        if a.synchronize_normalizers:
            for model in (runner.alg.actor, runner.alg.critic):
                for buffer in model.obs_normalizer.state_dict().values():
                    copies = [torch.empty_like(buffer) for _ in range(4)]
                    dist.all_gather(copies, buffer.contiguous())
                    if any(not torch.equal(buffer, other) for other in copies):
                        raise ValueError("normalizer buffer differs across ranks")
        checkpoint = folder / "candidate.pt"
        runner.save(str(checkpoint))
        if (
            tracked_reference(root, a.reference_head) != pins
            or file_digest(a.protocol) != a.protocol_hash
        ):
            raise ValueError("reference or protocol changed during training")
        clean(a.expected_head)
        write_once(
            folder / "complete.json",
            dict(
                rank=rank,
                local_rank=local,
                gpu_uuid=gpu_map[local],
                declaration_hash=file_digest(folder / "declaration.json"),
                checkpoint_hash=file_digest(checkpoint),
                changed_parameters=changed,
                synchronized_actor_parameters=True,
                synchronized_normalizer_buffers=a.synchronize_normalizers,
                native_sim_steps=env._sim_step_counter,
                world_steps=16 * 24 * 2,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        dist.barrier()
        if rank == 0:
            receipts = [a.output / f"rank-{i}" / "complete.json" for i in range(4)]
            write_once(
                a.output / "complete.json",
                dict(
                    status="FOUR_GPU_MJWARP_G1_PPO_SMOKE_ONLY",
                    source_head=a.expected_head,
                    reference_head=a.reference_head,
                    rank_receipt_hashes=[file_digest(path) for path in receipts],
                    gpu_uuids=[gpu_map[i] for i in range(4)],
                    world_steps=4 * 16 * 24 * 2,
                    trainable_actor_synchronized=True,
                    normalizer_buffer_synchronization=(
                        "VERIFIED_ACTOR_AND_CRITIC" if a.synchronize_normalizers else "NOT_VERIFIED"
                    ),
                    kick_skill_proven=False,
                    team_game_verified=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
            )
    except Exception as error:
        write_once(
            folder / "failed.json",
            dict(
                rank=rank,
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
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
