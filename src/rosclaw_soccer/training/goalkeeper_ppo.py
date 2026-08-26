"""Distributed online PPO bootstrap for the causal Goalkeeper V2 actor.

The fast world is deliberately an analytical, vectorized goalkeeper task.  It
is a candidate generator, never promotion physics.  CPU MuJoCo remains the
strict authority and must re-run the frozen Coverage--Time suite before an
artifact can become Champion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.goalkeeper_v2.motion_library import (
    GoalkeeperMotionFamily,
    GoalkeeperMotionLibrary,
    load_goalkeeper_motion_library,
    load_motion_clip_frames,
)
from rosclaw_soccer.skills.goalkeeper_v2.observations import GoalkeeperObservationSpec
from rosclaw_soccer.skills.goalkeeper_v2.policy import (
    GoalkeeperActorArtifact,
    GoalkeeperDenseLayer,
    save_goalkeeper_actor_artifact,
)


@dataclass(frozen=True)
class GoalkeeperPPOConfig:
    environments_per_rank: int = 1024
    iterations: int = 80
    update_epochs: int = 3
    hidden_size: int = 64
    learning_rate: float = 3.0e-4
    clip_ratio: float = 0.20
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    motion_prior_coefficient: float = 0.02
    task_action_coefficient: float = 0.12
    maximum_lateral_speed_mps: float = 0.40
    maximum_joint_residual_scale: float = 0.25
    maximum_leg_residual_scale: float = 0.30
    operational_space_reach_enabled: bool = True
    paired_mirror_curriculum: bool = True
    hard_negative_fraction: float = 0.35
    mirror_consistency_coefficient: float = 0.10
    random_seed: int = 19
    schema_version: str = "rosclaw_soccer.goalkeeper_ppo_config.v1"

    def __post_init__(self) -> None:
        if not 64 <= self.environments_per_rank <= 16_384:
            raise ValueError("goalkeeper PPO environments_per_rank must be in [64, 16384]")
        if not 1 <= self.iterations <= 100_000 or not 1 <= self.update_epochs <= 16:
            raise ValueError("goalkeeper PPO iteration counts are invalid")
        if not 16 <= self.hidden_size <= 512:
            raise ValueError("goalkeeper PPO hidden size must be in [16, 512]")
        positive_values = (
            self.learning_rate,
            self.clip_ratio,
            self.entropy_coefficient,
            self.value_coefficient,
            self.motion_prior_coefficient,
            self.task_action_coefficient,
            self.maximum_lateral_speed_mps,
            self.maximum_joint_residual_scale,
            self.maximum_leg_residual_scale,
            self.mirror_consistency_coefficient,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError("goalkeeper PPO scalar settings must be finite and positive")
        if not math.isfinite(self.hard_negative_fraction):
            raise ValueError("goalkeeper hard-negative fraction must be finite")
        if self.maximum_joint_residual_scale > 1.0 or self.maximum_leg_residual_scale > 0.50:
            raise ValueError("goalkeeper PPO joint residual scale exceeds its safety ceiling")
        if not 0.0 <= self.hard_negative_fraction <= 0.75:
            raise ValueError("goalkeeper hard-negative fraction must be in [0, 0.75]")
        if not 0 <= self.random_seed < 2**31:
            raise ValueError("goalkeeper PPO seed must be a signed 32-bit integer")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class GoalkeeperResetCurriculum:
    phase_names: tuple[str, ...] = ("ready", "flight", "landing", "recovery")
    phase_probabilities: tuple[float, ...] = (0.15, 0.50, 0.15, 0.20)
    continuous_episode_sec: float = 3.0
    terminates_on_save: bool = False
    terminates_on_ball_contact: bool = False
    reset_from_motion_library: bool = True
    schema_version: str = "rosclaw_soccer.goalkeeper_reset_curriculum.v1"

    def __post_init__(self) -> None:
        if len(self.phase_names) != len(self.phase_probabilities) or len(
            set(self.phase_names)
        ) != len(self.phase_names):
            raise ValueError("goalkeeper reset curriculum phases are invalid")
        if any(value <= 0.0 or not math.isfinite(value) for value in self.phase_probabilities):
            raise ValueError("goalkeeper reset curriculum probabilities must be positive")
        if not math.isclose(sum(self.phase_probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("goalkeeper reset curriculum probabilities must sum to one")
        if self.continuous_episode_sec < 3.0:
            raise ValueError("goalkeeper episodes must include post-save recovery")
        if self.terminates_on_save or self.terminates_on_ball_contact:
            raise ValueError("goalkeeper learning cannot terminate at the first save/contact")

    @property
    def curriculum_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class GoalkeeperPPOTrainingReport:
    config_hash: str
    curriculum_hash: str
    actor_observation_contract_hash: str
    critic_observation_contract_hash: str
    motion_library_hash: str
    parent_policy_hash: str
    candidate_policy_hash: str
    body_hash: str
    world_size: int
    gpu_devices: tuple[str, ...]
    samples_seen: int
    initial_mean_reward: float
    final_mean_reward: float
    best_mean_reward: float
    final_save_proxy_rate: float
    final_recovery_proxy_rate: float
    phase_sample_counts: dict[str, int]
    synchronized_ddp: bool
    maximum_cross_rank_parameter_difference: float
    actor_critic_asymmetric: bool = True
    online_policy_updates: bool = True
    phase_conditioned_reset_sampling: bool = True
    multi_step_episode_training: bool = False
    continuous_reset_enabled: bool = False
    strict_physics_evaluation_completed: bool = False
    promotion_status: str = "CANDIDATE_PENDING_CPU_MUJOCO_EXAM"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_ppo_training_report.v2"

    @property
    def report_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["gpu_devices"] = list(self.gpu_devices)
        if include_hash:
            value["report_hash"] = self.report_hash
        return value


_JOINT_LIMITS = (
    0.12,
    0.12,
    0.10,
    0.15,
    0.08,
    0.08,
    0.12,
    0.12,
    0.10,
    0.15,
    0.08,
    0.08,
    0.18,
    0.16,
    0.14,
    0.28,
    0.30,
    0.25,
    0.24,
    0.12,
    0.10,
    0.10,
    0.28,
    0.30,
    0.25,
    0.24,
    0.12,
    0.10,
    0.10,
)


def run_distributed_goalkeeper_ppo(
    *,
    motion_library_path: Path,
    dataset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    body_hash: str,
    parent_policy_hash: str,
    config: GoalkeeperPPOConfig | None = None,
    curriculum: GoalkeeperResetCurriculum | None = None,
) -> GoalkeeperPPOTrainingReport | None:
    """Run one DDP job; non-zero ranks return ``None`` after synchronization."""

    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel

    active = config or GoalkeeperPPOConfig()
    resets = curriculum or GoalkeeperResetCurriculum()
    library = load_goalkeeper_motion_library(
        motion_library_path,
        dataset_root=dataset_root,
    )
    if library.body_hash != body_hash:
        raise ValueError("goalkeeper PPO motion library Body hash mismatch")
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("goalkeeper PPO artifacts must remain outside the source checkout")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("goalkeeper PPO requires one visible CUDA device per rank")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    seed = active.random_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    spec = GoalkeeperObservationSpec()
    actor_size = len(spec.actor_names)
    critic_size = len(spec.privileged_critic_names)

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(actor_size, active.hidden_size),
                nn.Tanh(),
                nn.Linear(active.hidden_size, 30),
                nn.Tanh(),
            )
            self.critic = nn.Sequential(
                nn.Linear(critic_size, active.hidden_size),
                nn.Tanh(),
                nn.Linear(active.hidden_size, 1),
            )
            self.log_std = nn.Parameter(torch.full((30,), -0.8))

        def forward(self, observation: Any, privileged: Any) -> tuple[Any, Any, Any]:
            return (
                self.actor(observation),
                self.critic(privileged).squeeze(-1),
                self.log_std,
            )

    model: Any = ActorCritic().to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=active.learning_rate)
    teacher = _motion_teacher_table(library, dataset_root)
    teacher_tensor = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in teacher.items()
    }
    phase_counts = torch.zeros(len(resets.phase_names), device=device, dtype=torch.long)
    initial_reward = 0.0
    best_reward = -math.inf
    final_reward = 0.0
    final_save = 0.0
    final_recovery = 0.0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for iteration in range(active.iterations):
        batch = _sample_fast_world(
            torch=torch,
            device=device,
            generator=generator,
            batch_size=active.environments_per_rank,
            spec=spec,
            curriculum=resets,
            teacher=teacher_tensor,
            config=active,
        )
        actor_obs, critic_obs, optimal_action, teacher_action, phases = batch
        phase_counts += torch.bincount(phases, minlength=len(resets.phase_names))
        with torch.no_grad():
            old_mean, old_value, old_log_std = model(actor_obs, critic_obs)
            old_std = torch.exp(old_log_std).expand_as(old_mean)
            old_distribution = torch.distributions.Normal(old_mean, old_std)
            action = torch.clamp(old_distribution.sample(), -1.0, 1.0)  # type: ignore[no-untyped-call]
            old_log_probability = old_distribution.log_prob(action).sum(-1)  # type: ignore[no-untyped-call]
            reward, save_proxy, recovery_proxy = _fast_world_reward(
                torch=torch,
                action=action,
                optimal_action=optimal_action,
                teacher_action=teacher_action,
                phases=phases,
            )
            returns = reward
            advantages = returns - old_value
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        for _ in range(active.update_epochs):
            mean, value, log_std = model(actor_obs, critic_obs)
            standard_deviation = torch.exp(log_std).expand_as(mean)
            distribution = torch.distributions.Normal(mean, standard_deviation)
            log_probability = distribution.log_prob(action).sum(-1)  # type: ignore[no-untyped-call]
            ratio = torch.exp(log_probability - old_log_probability)
            unclipped = ratio * advantages
            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - active.clip_ratio,
                    1.0 + active.clip_ratio,
                )
                * advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.mean((value - returns) ** 2)
            motion_loss = torch.mean((mean - teacher_action) ** 2)
            # The lateral intercept channel is one of 30 outputs.  A plain
            # vector MSE diluted it by 30x, allowing prettier joint motion to
            # improve the fast-world proxy while the keeper barely moved.
            # This auxiliary distillation is train-only; online PPO remains
            # the source of task returns and the deployed actor stays causal.
            task_loss = 0.60 * torch.mean((mean[:, 0] - optimal_action[:, 0]) ** 2)
            task_loss += 0.20 * torch.mean((mean[:, 1:13] - optimal_action[:, 1:13]) ** 2)
            task_loss += 0.08 * torch.mean((mean[:, 13:16] - optimal_action[:, 13:16]) ** 2)
            task_loss += 0.12 * torch.mean((mean[:, 16:] - optimal_action[:, 16:]) ** 2)
            mirror_loss = _paired_mirror_consistency_loss(
                torch=torch,
                action=mean,
                phases=phases,
            )
            entropy = distribution.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]
            loss = (
                policy_loss
                + active.value_coefficient * value_loss
                + active.motion_prior_coefficient * motion_loss
                + active.task_action_coefficient * task_loss
                + active.mirror_consistency_coefficient * mirror_loss
                - active.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = torch.stack((reward.mean(), save_proxy.mean(), recovery_proxy.mean()))
        if world_size > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
        mean_reward, mean_save, mean_recovery = (
            float(metrics[0].item()),
            float(metrics[1].item()),
            float(metrics[2].item()),
        )
        if iteration == 0:
            initial_reward = mean_reward
        final_reward = mean_reward
        final_save = mean_save
        final_recovery = mean_recovery
        best_reward = max(best_reward, mean_reward)

    if world_size > 1:
        dist.all_reduce(phase_counts, op=dist.ReduceOp.SUM)
    devices = _gather_device_names(
        torch=torch,
        dist=dist,
        world_size=world_size,
        local_device_name=torch.cuda.get_device_name(local_rank),
    )
    module = model.module if world_size > 1 else model
    cross_rank_difference = _maximum_cross_rank_parameter_difference(
        torch=torch,
        dist=dist,
        model=module,
        world_size=world_size,
    )
    if world_size > 1 and cross_rank_difference > 1e-7:
        raise RuntimeError("goalkeeper PPO DDP parameters diverged across ranks")
    artifact = _export_actor(
        model=module,
        config=active,
        body_hash=body_hash,
        parent_policy_hash=parent_policy_hash,
        motion_library_hash=library.library_hash,
        actor_contract_hash=spec.actor_contract_hash,
        run_hash=hash_json(
            {
                "config_hash": active.config_hash,
                "curriculum_hash": resets.curriculum_hash,
                "world_size": world_size,
                "seed": active.random_seed,
            }
        ),
    )
    report = GoalkeeperPPOTrainingReport(
        config_hash=active.config_hash,
        curriculum_hash=resets.curriculum_hash,
        actor_observation_contract_hash=spec.actor_contract_hash,
        critic_observation_contract_hash=spec.critic_contract_hash,
        motion_library_hash=library.library_hash,
        parent_policy_hash=parent_policy_hash,
        candidate_policy_hash=artifact.policy_hash,
        body_hash=body_hash,
        world_size=world_size,
        gpu_devices=devices,
        samples_seen=active.environments_per_rank * active.iterations * world_size,
        initial_mean_reward=initial_reward,
        final_mean_reward=final_reward,
        best_mean_reward=best_reward,
        final_save_proxy_rate=final_save,
        final_recovery_proxy_rate=final_recovery,
        phase_sample_counts={
            name: int(phase_counts[index].item()) for index, name in enumerate(resets.phase_names)
        },
        synchronized_ddp=world_size > 1,
        maximum_cross_rank_parameter_difference=cross_rank_difference,
    )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        save_goalkeeper_actor_artifact(
            artifact,
            output / "goalkeeper-v2-candidate.json",
            source_checkout=checkout,
        )
        (output / "training-report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "request.json").write_text(
            json.dumps(
                {
                    "schema_version": "rosclaw_soccer.goalkeeper_ppo_request.v1",
                    "config": asdict(active),
                    "curriculum": asdict(resets),
                    "motion_library_hash": library.library_hash,
                    "body_hash": body_hash,
                    "parent_policy_hash": parent_policy_hash,
                    "runtime": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "torch": torch.__version__,
                        "torch_cuda": str(torch.version.cuda),
                    },
                    "activation_ceiling": "SIM_ONLY",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return report if rank == 0 else None


def _sample_fast_world(
    *,
    torch: Any,
    device: Any,
    generator: Any,
    batch_size: int,
    spec: GoalkeeperObservationSpec,
    curriculum: GoalkeeperResetCurriculum,
    teacher: dict[GoalkeeperMotionFamily, Any],
    config: GoalkeeperPPOConfig,
) -> tuple[Any, Any, Any, Any, Any]:
    probabilities = torch.tensor(curriculum.phase_probabilities, dtype=torch.float32, device=device)
    phases = torch.multinomial(
        probabilities,
        batch_size,
        replacement=True,
        generator=generator,
    )
    if config.paired_mirror_curriculum:
        # Each adjacent pair is one physical scenario reflected across the
        # sagittal plane, including the same episode phase.
        phases[1 : 2 * (batch_size // 2) : 2] = phases[: 2 * (batch_size // 2) : 2]
    pair_count = batch_size // 2
    pair_y = torch.empty(pair_count, device=device).uniform_(0.20, 1.30, generator=generator)
    pair_z = torch.empty(pair_count, device=device).uniform_(0.25, 1.75, generator=generator)
    local_target_y = torch.repeat_interleave(pair_y, 2)
    local_target_y[1::2] *= -1.0
    target_z = torch.repeat_interleave(pair_z, 2)
    if batch_size % 2:
        local_target_y = torch.cat((local_target_y, torch.zeros(1, device=device)))
        target_z = torch.cat(
            (target_z, torch.empty(1, device=device).uniform_(0.25, 1.75, generator=generator))
        )
    hard_count = min(
        pair_count,
        int(round(pair_count * config.hard_negative_fraction))
        if config.paired_mirror_curriculum
        else 0,
    )
    if hard_count:
        hard_y = torch.empty(hard_count, device=device).uniform_(0.28, 0.58, generator=generator)
        hard_z = torch.empty(hard_count, device=device).uniform_(0.30, 1.30, generator=generator)
        local_target_y[: 2 * hard_count : 2] = hard_y
        local_target_y[1 : 2 * hard_count : 2] = -hard_y
        target_z[: 2 * hard_count] = torch.repeat_interleave(hard_z, 2)
    total_flight = torch.empty(batch_size, device=device).uniform_(0.38, 1.20, generator=generator)
    launch_x = torch.empty(batch_size, device=device).uniform_(3.0, 5.0, generator=generator)
    elapsed = torch.minimum(
        torch.empty(batch_size, device=device).uniform_(0.02, 0.35, generator=generator),
        0.65 * total_flight,
    )
    remaining = total_flight - elapsed
    velocity_x = -launch_x / total_flight
    velocity_y = local_target_y / total_flight
    launch_z = torch.full_like(target_z, 0.60)
    launch_velocity_z = (target_z - launch_z + 4.905 * total_flight**2) / total_flight
    history_age = (
        torch.arange(
            spec.ball_history_steps - 1,
            -1,
            -1,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    # A real observer pads the earliest visible sample until enough history
    # exists.  Clamp negative sample times to launch time to reproduce that
    # exact causal startup instead of leaking a pre-launch trajectory.
    sample_time = torch.clamp(elapsed[:, None] - history_age[None, :], min=0.0)
    history_x = launch_x[:, None] + velocity_x[:, None] * sample_time
    history_y = velocity_y[:, None] * sample_time
    history_z = (
        launch_z[:, None] + launch_velocity_z[:, None] * sample_time - 4.905 * sample_time**2
    )
    flight = phases == 1
    ready = phases == 0
    post_save = (phases == 2) | (phases == 3)
    history_x[ready] = launch_x[ready, None]
    history_y[ready] = 0.0
    history_z[ready] = launch_z[ready, None]
    history_x[post_save] = -0.30
    history_y[post_save] = local_target_y[post_save, None]
    history_z[post_save] = 0.11
    history = torch.stack((history_x, history_y, history_z), dim=-1).reshape(batch_size, -1)

    earliest_visible_time = torch.clamp(
        torch.floor(elapsed / 0.02) * 0.02 - (spec.ball_history_steps - 1) * 0.02,
        min=0.0,
    )
    visible_span = torch.clamp(elapsed - earliest_visible_time, min=0.02)
    current_x = launch_x + velocity_x * elapsed
    current_y = velocity_y * elapsed
    current_z = launch_z + launch_velocity_z * elapsed - 4.905 * elapsed**2
    earliest_x = launch_x + velocity_x * earliest_visible_time
    earliest_y = velocity_y * earliest_visible_time
    earliest_z = (
        launch_z + launch_velocity_z * earliest_visible_time - 4.905 * earliest_visible_time**2
    )
    estimated_velocity = torch.stack(
        (
            (current_x - earliest_x) / visible_span,
            (current_y - earliest_y) / visible_span,
            (current_z - earliest_z) / visible_span,
        ),
        dim=1,
    )
    estimated_velocity[~flight] = 0.0
    estimated_horizon = torch.clamp(-current_x / estimated_velocity[:, 0], 0.0, 3.0)
    estimated_y = current_y + estimated_velocity[:, 1] * estimated_horizon
    estimated_z = torch.clamp(
        current_z + estimated_velocity[:, 2] * estimated_horizon - 4.905 * estimated_horizon**2,
        min=0.0,
    )
    estimated_intercept = torch.stack((estimated_horizon, estimated_y, estimated_z), dim=1)
    estimated_intercept[~flight] = 0.0
    history_count = torch.clamp(torch.floor(elapsed / 0.02) + 1.0, 1.0, spec.ball_history_steps)
    confidence = ((history_count - 1.0) / (spec.ball_history_steps - 1.0)).unsqueeze(1)
    confidence[~flight] = 0.0
    estimated_region = _region_indices(torch, -estimated_y, estimated_z)
    estimated_region_one_hot = torch.nn.functional.one_hot(estimated_region, num_classes=6).float()
    estimated_region_one_hot[~flight] = 0.0
    estimated_region_one_hot[~flight, 5] = 1.0
    q = torch.zeros((batch_size, 29), device=device)
    dq = torch.zeros((batch_size, 29), device=device)
    previous = torch.zeros((batch_size, 29), device=device)
    landing = phases == 2
    recovering = phases == 3
    side = torch.sign(-local_target_y)
    q[landing, 13] = 0.12 * side[landing]
    q[landing, 3] = 0.25
    q[landing, 9] = 0.25
    q[recovering, 14] = 0.10
    dq[landing] = 0.20 * torch.randn(
        (int(landing.sum().item()), 29), generator=generator, device=device
    )
    gravity = torch.zeros((batch_size, 3), device=device)
    gravity[:, 2] = -1.0
    gravity[landing, 0] = 0.20 * side[landing]
    root_velocity = torch.zeros((batch_size, 3), device=device)
    angular = torch.zeros((batch_size, 3), device=device)
    angular[landing, 0] = 0.4 * side[landing]
    actor = torch.cat(
        (
            history,
            estimated_velocity,
            estimated_intercept,
            confidence,
            estimated_region_one_hot,
            gravity,
            root_velocity,
            angular,
            q,
            dq,
            previous,
        ),
        dim=1,
    )
    if actor.shape[1] != len(spec.actor_names):
        raise RuntimeError("fast-world actor observation violated its frozen contract")

    world_target_y = -local_target_y
    region = _region_indices(torch, world_target_y, target_z)
    region_one_hot = torch.nn.functional.one_hot(region, num_classes=6).float()
    contact = torch.zeros((batch_size, 1), device=device)
    hands = torch.zeros((batch_size, 6), device=device)
    privileged = torch.cat(
        (
            actor,
            torch.stack(
                (
                    velocity_x,
                    velocity_y,
                    launch_velocity_z - 9.81 * elapsed,
                ),
                1,
            ),
            torch.stack((remaining, world_target_y, target_z), 1),
            region_one_hot,
            contact,
            hands,
        ),
        dim=1,
    )
    if privileged.shape[1] != len(spec.privileged_critic_names):
        raise RuntimeError("fast-world critic observation violated its frozen contract")

    optimal = torch.zeros((batch_size, 30), device=device)
    # The keeper faces world -x; actor action +y is therefore world -y.
    optimal[:, 0] = torch.clamp(local_target_y / 0.9, -1.0, 1.0)
    magnitude = torch.clamp(torch.abs(world_target_y) / 0.9, 0.0, 1.0)
    high = target_z > 1.0
    low = target_z < 0.65
    # Output dimensions 1..29 correspond to canonical G1 joints 0..28.
    optimal[:, 1 + 13] = side * 0.5 * magnitude
    optimal[:, 1 + 12] = side * 0.25 * magnitude
    left = side < 0.0
    right = ~left
    optimal[left, 1 + 15] = torch.where(high[left], -0.9, -0.45) * magnitude[left]
    optimal[left, 1 + 16] = 0.8 * magnitude[left]
    optimal[left, 1 + 18] = 0.6 * magnitude[left]
    optimal[right, 1 + 22] = torch.where(high[right], -0.9, -0.45) * magnitude[right]
    optimal[right, 1 + 23] = -0.8 * magnitude[right]
    optimal[right, 1 + 25] = 0.6 * magnitude[right]
    # A high save is a temporal skill, not one frozen pose.  Early in flight
    # the keeper loads both legs; inside the final 0.42 s it extends against
    # the ground while the task-space controller owns the selected hand.  The
    # actor sees only its causal horizon estimate, so this remains deployable.
    high_loading = flight & high & (remaining > 0.42)
    high_extension = flight & high & ~high_loading
    optimal[high_loading, 1 + 0] = -0.25
    optimal[high_loading, 1 + 3] = 0.80
    optimal[high_loading, 1 + 6] = -0.25
    optimal[high_loading, 1 + 9] = 0.80
    optimal[high_extension, 1 + 0] = 0.35
    optimal[high_extension, 1 + 3] = -0.70
    optimal[high_extension, 1 + 6] = 0.35
    optimal[high_extension, 1 + 9] = -0.70
    optimal[low, 1 + 3] = 0.6
    optimal[low, 1 + 9] = 0.6
    optimal[recovering, 0] = 0.0
    optimal[recovering, 1:] *= 0.15
    optimal[phases == 0] *= 0.15

    teacher_action = torch.zeros_like(optimal)
    for family, value in teacher.items():
        mask = _family_mask(torch, family, phases, world_target_y, target_z)
        if torch.any(mask):
            teacher_action[mask, 1:] = value
    teacher_action[:, 0] = optimal[:, 0]
    return actor, privileged, optimal, teacher_action, phases


def _fast_world_reward(
    *,
    torch: Any,
    action: Any,
    optimal_action: Any,
    teacher_action: Any,
    phases: Any,
) -> tuple[Any, Any, Any]:
    lateral_error = (action[:, 0] - optimal_action[:, 0]) ** 2
    leg_error = torch.mean((action[:, 1:13] - optimal_action[:, 1:13]) ** 2, dim=1)
    waist_error = torch.mean((action[:, 13:16] - optimal_action[:, 13:16]) ** 2, dim=1)
    arm_error = torch.mean((action[:, 16:] - optimal_action[:, 16:]) ** 2, dim=1)
    style_error = torch.mean((action[:, 1:] - teacher_action[:, 1:]) ** 2, dim=1)
    save_proxy = torch.exp(
        -5.0 * (0.55 * lateral_error + 0.20 * leg_error + 0.08 * waist_error + 0.17 * arm_error)
    )
    recovery_mask = phases == 3
    recovery_proxy = torch.where(
        recovery_mask,
        torch.exp(-8.0 * torch.mean(action**2, dim=1)),
        torch.ones_like(save_proxy),
    )
    smoothness = torch.mean(action**2, dim=1)
    reward = 0.78 * save_proxy + 0.15 * recovery_proxy + 0.07 * torch.exp(-style_error)
    reward -= 0.015 * smoothness
    return reward, save_proxy, recovery_proxy


def _paired_mirror_consistency_loss(*, torch: Any, action: Any, phases: Any) -> Any:
    """Penalize left/right policy bias on adjacent mirrored observations."""

    pair_count = action.shape[0] // 2
    if pair_count == 0:
        return torch.zeros((), dtype=action.dtype, device=action.device)
    paired_flight = (phases[: 2 * pair_count : 2] == 1) & (phases[1 : 2 * pair_count : 2] == 1)
    if not bool(torch.any(paired_flight)):
        return torch.zeros((), dtype=action.dtype, device=action.device)
    left = action[: 2 * pair_count : 2][paired_flight]
    right = action[1 : 2 * pair_count : 2][paired_flight]
    # Lateral velocity and waist yaw/roll reverse sign under reflection;
    # sagittal waist pitch is invariant.  Arm targets swap anatomically.
    error = [(left[:, 0] + right[:, 0]) ** 2]
    error.append((left[:, 1 + 12] + right[:, 1 + 12]) ** 2)
    error.append((left[:, 1 + 13] + right[:, 1 + 13]) ** 2)
    error.append((left[:, 1 + 14] - right[:, 1 + 14]) ** 2)
    for left_joint, right_joint, sign in (
        (0, 6, 1.0),
        (1, 7, -1.0),
        (2, 8, -1.0),
        (3, 9, 1.0),
        (4, 10, 1.0),
        (5, 11, -1.0),
    ):
        error.append((left[:, 1 + left_joint] - sign * right[:, 1 + right_joint]) ** 2)
    for left_joint, right_joint in zip(range(15, 22), range(22, 29), strict=True):
        error.append((left[:, 1 + left_joint] - right[:, 1 + right_joint]) ** 2)
    return torch.mean(torch.stack(error, dim=1))


def _motion_teacher_table(
    library: GoalkeeperMotionLibrary,
    dataset_root: Path,
) -> dict[GoalkeeperMotionFamily, np.ndarray]:
    result: dict[GoalkeeperMotionFamily, np.ndarray] = {}
    limits = np.asarray(_JOINT_LIMITS, dtype=np.float64)
    for family in GoalkeeperMotionFamily:
        clip = next(item for item in library.clips if item.family is family)
        q, _ = load_motion_clip_frames(dataset_root=dataset_root, clip=clip)
        centered = q - q[:1]
        energy = np.sqrt(np.mean(np.square(centered), axis=1))
        frame = centered[int(np.argmax(energy))]
        result[family] = np.asarray(np.clip(frame / limits, -1.0, 1.0), dtype=np.float32)
    return result


def _region_indices(torch: Any, world_y: Any, z: Any) -> Any:
    region = torch.full_like(world_y, 4, dtype=torch.long)
    left = world_y > 0.25
    right = world_y < -0.25
    high = z >= 1.0
    region[left & high] = 0
    region[right & high] = 1
    region[left & ~high] = 2
    region[right & ~high] = 3
    return region


def _family_mask(torch: Any, family: GoalkeeperMotionFamily, phases: Any, y: Any, z: Any) -> Any:
    left = y > 0.25
    right = y < -0.25
    high = z >= 1.0
    low = z < 1.0
    flight = phases == 1
    mapping = {
        GoalkeeperMotionFamily.READY: phases == 0,
        GoalkeeperMotionFamily.SPLIT_STEP: flight & ~(left | right),
        GoalkeeperMotionFamily.SHUFFLE_LEFT: flight & left,
        GoalkeeperMotionFamily.SHUFFLE_RIGHT: flight & right,
        GoalkeeperMotionFamily.LOW_SAVE_LEFT: (flight | (phases == 2)) & left & low,
        GoalkeeperMotionFamily.LOW_SAVE_RIGHT: (flight | (phases == 2)) & right & low,
        GoalkeeperMotionFamily.HIGH_REACH_LEFT: (flight | (phases == 2)) & left & high,
        GoalkeeperMotionFamily.HIGH_REACH_RIGHT: (flight | (phases == 2)) & right & high,
        GoalkeeperMotionFamily.CENTER_BLOCK: flight & ~(left | right),
        GoalkeeperMotionFamily.RECOVERY: phases == 3,
    }
    return mapping[family]


def _export_actor(
    *,
    model: Any,
    config: GoalkeeperPPOConfig,
    body_hash: str,
    parent_policy_hash: str,
    motion_library_hash: str,
    actor_contract_hash: str,
    run_hash: str,
) -> GoalkeeperActorArtifact:
    linear = [module for module in model.actor if module.__class__.__name__ == "Linear"]
    if len(linear) != 2:
        raise RuntimeError("goalkeeper actor export expected exactly two linear layers")
    layers = tuple(
        GoalkeeperDenseLayer(
            weights=tuple(
                tuple(float(value) for value in row)
                for row in layer.weight.detach().cpu().numpy().T
            ),
            bias=tuple(float(value) for value in layer.bias.detach().cpu().numpy()),
            activation="tanh",
        )
        for layer in linear
    )
    return GoalkeeperActorArtifact(
        policy_id="soccer.goalkeeper.g1.v2",
        generation=1,
        parent_policy_hash=parent_policy_hash,
        body_hash=body_hash,
        actor_observation_contract_hash=actor_contract_hash,
        motion_library_hash=motion_library_hash,
        training_run_hash=run_hash,
        layers=layers,
        maximum_lateral_speed_mps=config.maximum_lateral_speed_mps,
        maximum_joint_residual_rad=tuple(
            value
            * (
                config.maximum_leg_residual_scale
                if index < 12
                else config.maximum_joint_residual_scale
            )
            for index, value in enumerate(_JOINT_LIMITS)
        ),
        operational_space_reach_enabled=config.operational_space_reach_enabled,
    )


def _gather_device_names(
    *,
    torch: Any,
    dist: Any,
    world_size: int,
    local_device_name: str,
) -> tuple[str, ...]:
    if world_size == 1:
        return (local_device_name,)
    encoded = local_device_name.encode()[:127]
    payload = torch.zeros(128, dtype=torch.uint8, device="cuda")
    payload[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device="cuda")
    gathered = [torch.zeros_like(payload) for _ in range(world_size)]
    dist.all_gather(gathered, payload)
    return tuple(
        bytes(int(value) for value in item.cpu().tolist()).rstrip(b"\0").decode()
        for item in gathered
    )


def _maximum_cross_rank_parameter_difference(
    *,
    torch: Any,
    dist: Any,
    model: Any,
    world_size: int,
) -> float:
    if world_size == 1:
        return 0.0
    flattened = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    minimum = flattened.clone()
    maximum = flattened.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(torch.max(torch.abs(maximum - minimum)).item())


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SIM_ONLY Goalkeeper V2 PPO candidate")
    parser.add_argument("--motion-library", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--body-hash", required=True)
    parser.add_argument("--parent-policy-hash", required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--environments-per-rank", type=int, default=1024)
    parser.add_argument("--maximum-joint-residual-scale", type=float, default=0.25)
    parser.add_argument("--maximum-leg-residual-scale", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    run_distributed_goalkeeper_ppo(
        motion_library_path=arguments.motion_library,
        dataset_root=arguments.dataset_root,
        output_dir=arguments.output_dir,
        source_checkout=arguments.source_checkout,
        body_hash=arguments.body_hash,
        parent_policy_hash=arguments.parent_policy_hash,
        config=GoalkeeperPPOConfig(
            iterations=arguments.iterations,
            environments_per_rank=arguments.environments_per_rank,
            maximum_joint_residual_scale=arguments.maximum_joint_residual_scale,
            maximum_leg_residual_scale=arguments.maximum_leg_residual_scale,
        ),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GoalkeeperPPOConfig",
    "GoalkeeperPPOTrainingReport",
    "GoalkeeperResetCurriculum",
    "run_distributed_goalkeeper_ppo",
]
