"""Collect true post-save G1 states for failure-driven recovery learning."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_agility import GoalkeeperAgilityConfig
from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    classify_recovery_posture,
    write_recovery_snapshot_corpus,
)


@dataclass(frozen=True)
class RecoverySnapshotCollectorConfig:
    """Bounded sampling contract around one frozen source policy."""

    environment_count: int = 64
    seeds: tuple[int, ...] = (49_001, 49_019, 49_037, 49_049)
    post_save_offsets_sec: tuple[float, ...] = (0.0, 0.20, 0.50, 1.0, 1.5, 2.0, 3.0)
    corpus_name: str = "goalkeeper-post-save-recovery"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.recovery_snapshot_collector_config.v1"

    def __post_init__(self) -> None:
        if (
            not 4 <= self.environment_count <= 4096
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(seed < 0 for seed in self.seeds)
            or not self.post_save_offsets_sec
            or self.post_save_offsets_sec[0] != 0.0
            or tuple(sorted(set(self.post_save_offsets_sec))) != self.post_save_offsets_sec
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 10.0
                for value in self.post_save_offsets_sec
            )
            or not self.corpus_name
            or len(self.corpus_name) > 64
            or not self.corpus_name.replace("-", "").isalnum()
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery snapshot collector config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _world_from_checkpoint(*, payload: dict[str, Any], environment_count: int) -> Any:
    world_payload = dict(payload)
    agility_payload = world_payload.get("agility")
    if not isinstance(agility_payload, dict):
        raise ValueError("source checkpoint goalkeeper agility contract is invalid")
    world_payload["agility"] = GoalkeeperAgilityConfig(**agility_payload)
    world = GoalkeeperMJWarpConfig(**world_payload)
    return replace(world, environment_count=environment_count)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def collect_goalkeeper_recovery_snapshots(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    source_checkpoint_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
    config: RecoverySnapshotCollectorConfig | None = None,
) -> dict[str, Any]:
    """Run a frozen actor and serialize its real post-save physics states."""

    import torch
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        GoalkeeperPhysicsPPOConfig,
        _build_actor_critic,
        _build_environment,
        _load_actor_critic_state,
    )

    active = config or RecoverySnapshotCollectorConfig()
    assets = asset_root.expanduser().resolve()
    checkpoint_path = source_checkpoint_path.expanduser().resolve()
    locomotion = locomotion_policy_path.expanduser().resolve()
    scene_path = assets / "g1_description" / "scene_with_ball.xml"
    if not checkpoint_path.is_file() or not locomotion.is_file() or not scene_path.is_file():
        raise FileNotFoundError("recovery snapshot collector inputs are incomplete")
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("recovery snapshot collector requires a CUDA physics device")
    torch.cuda.set_device(torch_device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    if (
        checkpoint.get("activation_ceiling") != "SIM_ONLY"
        or checkpoint.get("promotion_eligible") is True
    ):
        raise ValueError("source checkpoint safety boundary is invalid")
    training_payload = checkpoint.get("training_config")
    world_payload = checkpoint.get("world_config")
    if not isinstance(training_payload, dict) or not isinstance(world_payload, dict):
        raise ValueError("source checkpoint training contract is incomplete")
    training_config = GoalkeeperPhysicsPPOConfig(**training_payload)
    world_config = _world_from_checkpoint(
        payload=world_payload,
        environment_count=active.environment_count,
    )
    model = _build_actor_critic(
        torch,
        nn,
        int(checkpoint["observation_size"]),
        int(checkpoint["action_size"]),
        int(checkpoint["hidden_size"]),
    ).to(torch_device)
    _load_actor_critic_state(model, checkpoint["state_dict"])
    model.eval()
    environment = _build_environment(
        active=training_config,
        asset_root=assets,
        locomotion_policy_path=locomotion,
        device=torch_device,
        world_config=world_config,
    )
    if environment.count != active.environment_count:
        raise RuntimeError("recovery snapshot environment count binding failed")
    physics_device = str(environment.wp.get_device())
    if physics_device != str(torch_device):
        raise RuntimeError("recovery snapshot physics device binding failed")

    source_policy_hash = hash_bytes(checkpoint_path.read_bytes())
    body_hash = g1_body_hash(assets)
    scene_hash = hash_bytes(scene_path.read_bytes())
    source_config_hash = hash_json(
        {
            "training_config": asdict(training_config),
            "world_config": asdict(world_config),
            "collector_config": asdict(active),
        }
    )
    offset_steps = tuple(
        int(round(value / world_config.control_dt_sec)) for value in active.post_save_offsets_sec
    )
    snapshots: list[RecoverySnapshot] = []
    source_episodes = 0
    saved_episodes = 0
    hand_saved_episodes = 0
    failed_after_save = 0
    skipped_nonfinite = 0

    for seed in active.seeds:
        observation = environment.reset(seed=seed)
        save_step = torch.full(
            (environment.count,), -1, dtype=torch.long, device=torch_device
        )
        previous_save = torch.zeros(environment.count, dtype=torch.bool, device=torch_device)
        saw_unsupported = torch.zeros_like(previous_save)
        landing_recorded = torch.zeros_like(previous_save)
        terminal_recorded = torch.zeros_like(previous_save)
        recorded: set[tuple[int, int, str]] = set()

        for step in range(world_config.episode_steps):
            with torch.inference_mode():
                mean, _, _ = model(observation)
                action = torch.tanh(mean)
            observation, _, done, info = environment.step(action)
            qpos, qvel, left_foot, right_foot = environment.recovery_snapshot_state()
            current_save = environment.task.first_save.clone()
            newly_saved = current_save & ~previous_save
            save_step.copy_(torch.where(newly_saved, step, save_step))
            post_save = current_save & (save_step >= 0)
            any_foot = left_foot | right_foot
            saw_unsupported |= post_save & ~any_foot
            newly_landed = post_save & saw_unsupported & any_foot & ~landing_recorded
            failed = environment.task.phase == 7
            nonfinite = environment._nonfinite_quarantine_latched
            final_step = step == world_config.episode_steps - 1

            capture_indices = torch.nonzero(
                post_save & ~terminal_recorded, as_tuple=False
            ).flatten()
            for index_tensor in capture_indices:
                index = int(index_tensor)
                if bool(nonfinite[index]):
                    skipped_nonfinite += 1
                    terminal_recorded[index] = True
                    continue
                age_steps = step - int(save_step[index])
                offset_due = age_steps in offset_steps
                is_failure = bool(failed[index])
                is_terminal = final_step or bool(done[index])
                is_landing = bool(newly_landed[index])
                if not (
                    bool(newly_saved[index])
                    or offset_due
                    or is_landing
                    or is_failure
                    or is_terminal
                ):
                    continue
                if is_failure:
                    stage = "FAILURE_TERMINAL"
                elif is_terminal:
                    stage = "EPISODE_TERMINAL"
                elif bool(newly_saved[index]):
                    stage = "SAVE_EVENT"
                elif is_landing:
                    stage = "LANDING"
                else:
                    linear_speed = float(torch.linalg.vector_norm(qvel[index, :3]))
                    angular_speed = float(torch.linalg.vector_norm(qvel[index, 3:6]))
                    cluster = classify_recovery_posture(
                        root_quaternion_wxyz=qpos[index, 3:7].detach().cpu().numpy(),
                        pelvis_height_m=float(qpos[index, 2]),
                        root_linear_speed_mps=linear_speed,
                        root_angular_speed_rad_s=angular_speed,
                        left_foot_supported=bool(left_foot[index]),
                        right_foot_supported=bool(right_foot[index]),
                    )
                    stage = (
                        "POST_SAVE_FLIGHT"
                        if cluster == "AIRBORNE_OR_HIGH_MOMENTUM"
                        else "RECOVERY_ENTRY"
                    )
                key = (index, step, stage)
                if key in recorded:
                    continue
                linear_speed = float(torch.linalg.vector_norm(qvel[index, :3]))
                angular_speed = float(torch.linalg.vector_norm(qvel[index, 3:6]))
                cluster = classify_recovery_posture(
                    root_quaternion_wxyz=qpos[index, 3:7].detach().cpu().numpy(),
                    pelvis_height_m=float(qpos[index, 2]),
                    root_linear_speed_mps=linear_speed,
                    root_angular_speed_rad_s=angular_speed,
                    left_foot_supported=bool(left_foot[index]),
                    right_foot_supported=bool(right_foot[index]),
                )
                snapshots.append(
                    RecoverySnapshot(
                        episode_seed=seed,
                        environment_index=index,
                        control_step=step,
                        stage=stage,  # type: ignore[arg-type]
                        save_kind=(
                            "HAND" if bool(environment.task.first_hand_save[index]) else "BODY"
                        ),
                        posture_cluster=cluster,
                        qpos=qpos[index, :36].detach().cpu().numpy(),
                        qvel=qvel[index, :35].detach().cpu().numpy(),
                        applied_action=info["applied_action"][index].detach().cpu().numpy(),
                        ball_position_m=qpos[index, 36:39].detach().cpu().numpy(),
                        ball_velocity_mps=qvel[index, 35:38].detach().cpu().numpy(),
                        target_position_m=info["target_m"][index].detach().cpu().numpy(),
                        left_foot_supported=bool(left_foot[index]),
                        right_foot_supported=bool(right_foot[index]),
                        failed=is_failure,
                        body_hash=body_hash,
                        physics_scene_hash=scene_hash,
                        source_policy_hash=source_policy_hash,
                        source_config_hash=source_config_hash,
                    )
                )
                recorded.add(key)
                landing_recorded[index] |= is_landing
                terminal_recorded[index] |= is_failure or is_terminal
            previous_save.copy_(current_save)

        source_episodes += environment.count
        saved_episodes += int(environment.task.first_save.sum())
        hand_saved_episodes += int(environment.task.first_hand_save.sum())
        failed_after_save += int(
            (environment.task.first_save & (environment.task.phase == 7)).sum()
        )

    if not snapshots:
        raise RuntimeError("source policy produced no simulator-confirmed save snapshots")
    destination = output_dir.expanduser().resolve()
    manifest = write_recovery_snapshot_corpus(
        snapshots=snapshots,
        output_dir=destination,
        corpus_name=active.corpus_name,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_snapshot_collection_report.v1",
        "collector_config": asdict(active),
        "collector_config_hash": active.config_hash,
        "source_checkpoint": checkpoint_path.name,
        "source_policy_hash": source_policy_hash,
        "source_config_hash": source_config_hash,
        "body_hash": body_hash,
        "physics_scene_hash": scene_hash,
        "physics_backend": "mujoco_warp",
        "physics_device": physics_device,
        "source_episodes": source_episodes,
        "saved_episodes": saved_episodes,
        "hand_saved_episodes": hand_saved_episodes,
        "failed_after_save": failed_after_save,
        "save_rate": saved_episodes / source_episodes,
        "hand_save_rate": hand_saved_episodes / source_episodes,
        "post_save_failure_rate": failed_after_save / max(saved_episodes, 1),
        "skipped_nonfinite_states": skipped_nonfinite,
        "snapshot_count": len(snapshots),
        "corpus_manifest": f"{active.corpus_name}.json",
        "corpus_hash": manifest["corpus_hash"],
        "cluster_counts": manifest["cluster_counts"],
        "stage_counts": manifest["stage_counts"],
        "finite_state": environment.finite_state(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "collection-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment-count", type=int, default=64)
    parser.add_argument("--seeds", default="49001,49019,49037,49049")
    parser.add_argument("--corpus-name", default="goalkeeper-post-save-recovery")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    report = collect_goalkeeper_recovery_snapshots(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        source_checkpoint_path=args.source_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        config=RecoverySnapshotCollectorConfig(
            environment_count=args.environment_count,
            seeds=seeds,
            corpus_name=args.corpus_name,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoverySnapshotCollectorConfig",
    "collect_goalkeeper_recovery_snapshots",
]
