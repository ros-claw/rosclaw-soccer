"""Derive low-momentum recovery entries from true high-momentum failures."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    classify_recovery_posture,
    load_recovery_snapshot_corpus,
    write_recovery_snapshot_corpus,
)


@dataclass(frozen=True)
class RecoverySnapshotSettlerConfig:
    source_stages: tuple[str, ...] = ("FAILURE_TERMINAL",)
    settling_duration_sec: float = 1.0
    joint_damping_scale: float = 2.0
    maximum_snapshots: int = 64
    corpus_name: str = "settled-post-save-recovery"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.recovery_snapshot_settler_config.v1"

    def __post_init__(self) -> None:
        allowed = {
            "SAVE_EVENT",
            "POST_SAVE_FLIGHT",
            "LANDING",
            "RECOVERY_ENTRY",
            "FAILURE_TERMINAL",
            "EPISODE_TERMINAL",
        }
        if (
            not self.source_stages
            or not set(self.source_stages).issubset(allowed)
            or not math.isfinite(self.settling_duration_sec)
            or not 0.2 <= self.settling_duration_sec <= 3.0
            or not math.isfinite(self.joint_damping_scale)
            or not 0.5 <= self.joint_damping_scale <= 3.0
            or not 1 <= self.maximum_snapshots <= 64
            or not self.corpus_name
            or len(self.corpus_name) > 64
            or not self.corpus_name.replace("-", "").isalnum()
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery snapshot settler config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def settle_recovery_snapshots(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    getup_checkpoint_path: Path,
    getup_source_path: Path,
    getup_config_path: Path,
    snapshot_manifest_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
    config: RecoverySnapshotSettlerConfig | None = None,
) -> dict[str, Any]:
    """Apply only joint damping, then write the resulting physical states."""

    import torch

    from rosclaw_soccer.growth.mjlab_getup import load_mjlab_getup_torch
    from rosclaw_soccer.training.goalkeeper_mjwarp import (
        GoalkeeperMJWarpBatch,
        goalkeeper_world_config,
    )
    from rosclaw_soccer.training.joint_guard import project_joint_safe_torque_torch

    active = config or RecoverySnapshotSettlerConfig()
    corpus = load_recovery_snapshot_corpus(snapshot_manifest_path)
    selected = [item for item in corpus if item.stage in active.source_stages][
        : active.maximum_snapshots
    ]
    if not selected:
        raise ValueError("recovery snapshot settler selection is empty")
    assets = asset_root.expanduser().resolve()
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=len(selected),
        second_shot_probability=0.0,
        shot_intent_cue_enabled=True,
    )
    environment = GoalkeeperMJWarpBatch(
        asset_root=assets,
        locomotion_policy_path=locomotion_policy_path.expanduser().resolve(),
        device=torch_device,
        config=world,
    )
    physics_device = str(environment.wp.get_device())
    if physics_device != str(torch_device):
        raise RuntimeError("recovery snapshot settler physics device binding failed")
    environment.reset(seed=52_001)
    _, contract, _ = load_mjlab_getup_torch(
        checkpoint_path=getup_checkpoint_path,
        source_path=getup_source_path,
        config_path=getup_config_path,
        asset_root=assets,
        device=torch_device,
    )
    if any(
        item.body_hash != contract.body_hash
        or item.physics_scene_hash != contract.physics_scene_hash
        for item in selected
    ):
        raise ValueError("recovery snapshot settler source binding is invalid")
    environment.qpos[:, :36] = torch.as_tensor(
        [item.qpos.tolist() for item in selected], device=torch_device
    )
    environment.qvel[:, :35] = torch.as_tensor(
        [item.qvel.tolist() for item in selected], device=torch_device
    )
    environment._park_ball()
    environment.mjw.forward(environment.model, environment.data)
    damping = active.joint_damping_scale * torch.as_tensor(
        contract.joint_damping, device=torch_device
    )
    nonfinite = torch.zeros(len(selected), dtype=torch.bool, device=torch_device)
    steps = int(math.ceil(active.settling_duration_sec / world.control_dt_sec))
    for _ in range(steps):
        for _ in range(world.physics_substeps):
            torque = -damping * environment.qvel[:, 6:35]
            torque, _ = project_joint_safe_torque_torch(
                joint_position=environment.qpos[:, 7:36],
                joint_velocity=environment.qvel[:, 6:35],
                commanded_torque=torque,
                joint_ranges=environment._joint_ranges,
                limited=environment._joint_limited,
            )
            environment.ctrl.copy_(
                torch.clamp(torque, -environment._torque_limits, environment._torque_limits)
            )
            environment.mjw.step(environment.model, environment.data)
        finite = torch.all(torch.isfinite(environment.qpos), dim=1)
        finite &= torch.all(torch.isfinite(environment.qvel), dim=1)
        nonfinite |= ~finite
        if bool(nonfinite.any()):
            raise FloatingPointError("recovery snapshot settling produced a non-finite state")
    left_foot, right_foot = environment._foot_contact_state()
    settled: list[RecoverySnapshot] = []
    rows: list[dict[str, Any]] = []
    for index, parent in enumerate(selected):
        qpos = environment.qpos[index, :36].detach().cpu().numpy()
        qvel = environment.qvel[index, :35].detach().cpu().numpy()
        linear_speed = float(np.linalg.norm(qvel[:3]))
        angular_speed = float(np.linalg.norm(qvel[3:6]))
        cluster = classify_recovery_posture(
            root_quaternion_wxyz=qpos[3:7],
            pelvis_height_m=float(qpos[2]),
            root_linear_speed_mps=linear_speed,
            root_angular_speed_rad_s=angular_speed,
            left_foot_supported=bool(left_foot[index]),
            right_foot_supported=bool(right_foot[index]),
        )
        source_config_hash = hash_json(
            {
                "parent_source_config_hash": parent.source_config_hash,
                "settler_config_hash": active.config_hash,
                "getup_contract_hash": contract.contract_hash,
            }
        )
        snapshot = RecoverySnapshot(
            episode_seed=parent.episode_seed,
            environment_index=parent.environment_index,
            control_step=parent.control_step + steps,
            stage="RECOVERY_ENTRY",
            save_kind=parent.save_kind,
            posture_cluster=cluster,
            qpos=qpos,
            qvel=qvel,
            applied_action=np.zeros_like(parent.applied_action),
            ball_position_m=parent.ball_position_m,
            ball_velocity_mps=np.zeros(3, dtype=np.float64),
            target_position_m=parent.target_position_m,
            left_foot_supported=bool(left_foot[index]),
            right_foot_supported=bool(right_foot[index]),
            failed=True,
            body_hash=parent.body_hash,
            physics_scene_hash=parent.physics_scene_hash,
            source_policy_hash=parent.source_policy_hash,
            source_config_hash=source_config_hash,
        )
        settled.append(snapshot)
        rows.append(
            {
                "parent_snapshot_hash": parent.snapshot_hash,
                "settled_snapshot_hash": snapshot.snapshot_hash,
                "parent_cluster": parent.posture_cluster,
                "settled_cluster": cluster,
                "parent_root_linear_speed_mps": float(np.linalg.norm(parent.qvel[:3])),
                "settled_root_linear_speed_mps": linear_speed,
                "parent_root_angular_speed_rad_s": float(np.linalg.norm(parent.qvel[3:6])),
                "settled_root_angular_speed_rad_s": angular_speed,
                "settled_pelvis_height_m": float(qpos[2]),
            }
        )
    destination = output_dir.expanduser().resolve()
    manifest = write_recovery_snapshot_corpus(
        snapshots=settled,
        output_dir=destination,
        corpus_name=active.corpus_name,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_snapshot_settling_report.v1",
        "settler_config": asdict(active),
        "settler_config_hash": active.config_hash,
        "parent_snapshot_manifest": snapshot_manifest_path.expanduser().resolve().name,
        "parent_snapshot_count": len(selected),
        "settled_snapshot_count": len(settled),
        "settled_cluster_counts": manifest["cluster_counts"],
        "settled_stage_counts": manifest["stage_counts"],
        "corpus_manifest": f"{active.corpus_name}.json",
        "corpus_hash": manifest["corpus_hash"],
        "getup_contract_hash": contract.contract_hash,
        "physics_backend": "mujoco_warp_no_auto_reset",
        "physics_device": physics_device,
        "rows": rows,
        "nonfinite_count": int(nonfinite.sum()),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "settling-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--getup-checkpoint", type=Path, required=True)
    parser.add_argument("--getup-source", type=Path, required=True)
    parser.add_argument("--getup-config", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settling-duration-sec", type=float, default=1.0)
    parser.add_argument("--joint-damping-scale", type=float, default=2.0)
    parser.add_argument("--corpus-name", default="settled-post-save-recovery")
    args = parser.parse_args()
    report = settle_recovery_snapshots(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        getup_checkpoint_path=args.getup_checkpoint,
        getup_source_path=args.getup_source,
        getup_config_path=args.getup_config,
        snapshot_manifest_path=args.snapshot_manifest,
        output_dir=args.output_dir,
        device=args.device,
        config=RecoverySnapshotSettlerConfig(
            settling_duration_sec=args.settling_duration_sec,
            joint_damping_scale=args.joint_damping_scale,
            corpus_name=args.corpus_name,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RecoverySnapshotSettlerConfig", "settle_recovery_snapshots"]
