"""Test a frozen neural get-up expert on true post-skill recovery states."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus


@dataclass(frozen=True)
class RecoverySnapshotExamConfig:
    stages: tuple[str, ...] = ("FAILURE_TERMINAL",)
    maximum_snapshots: int = 64
    duration_sec: float = 20.0
    minimum_final_stable_sec: float = 2.0
    impact_absorption_sec: float = 0.0
    impact_joint_damping_scale: float = 1.0
    align_getup_reference_phase: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.recovery_snapshot_exam_config.v3"

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
            not self.stages
            or not set(self.stages).issubset(allowed)
            or not 1 <= self.maximum_snapshots <= 64
            or not math.isfinite(self.duration_sec)
            or not 12.0 <= self.duration_sec <= 30.0
            or not math.isfinite(self.minimum_final_stable_sec)
            or not 1.0 <= self.minimum_final_stable_sec <= 5.0
            or not math.isfinite(self.impact_absorption_sec)
            or not 0.0 <= self.impact_absorption_sec <= 2.0
            or not math.isfinite(self.impact_joint_damping_scale)
            or not 0.50 <= self.impact_joint_damping_scale <= 3.0
            or not isinstance(self.align_getup_reference_phase, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery snapshot exam config is invalid")

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


def run_recovery_snapshot_exam(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    getup_checkpoint_path: Path,
    getup_source_path: Path,
    getup_config_path: Path,
    snapshot_manifest_path: Path,
    output_path: Path,
    device: str = "cuda:0",
    config: RecoverySnapshotExamConfig | None = None,
) -> dict[str, Any]:
    """Run one no-reset physics exam from each selected corpus row."""

    import torch

    from rosclaw_soccer.growth.mjlab_getup import (
        MJLabGetUpTorchController,
        MJLabRecoveryHandoff,
        MJLabRecoveryHandoffConfig,
        estimate_mjlab_getup_reference_frame_torch,
        load_mjlab_getup_torch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import (
        GoalkeeperMJWarpBatch,
        goalkeeper_world_config,
    )
    from rosclaw_soccer.training.joint_guard import project_joint_safe_torque_torch

    active_config = config or RecoverySnapshotExamConfig()
    corpus = load_recovery_snapshot_corpus(snapshot_manifest_path)
    selected = [item for item in corpus if item.stage in active_config.stages][
        : active_config.maximum_snapshots
    ]
    if not selected:
        raise ValueError("recovery snapshot exam selection is empty")
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
        raise RuntimeError("recovery snapshot exam physics device binding failed")
    environment.reset(seed=51_001)
    policy, contract, references = load_mjlab_getup_torch(
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
        raise ValueError("recovery snapshots do not match the get-up body and scene")
    controller = MJLabGetUpTorchController(
        policy=policy,
        contract=contract,
        references=references,
        environment_count=len(selected),
        device=torch_device,
    )
    handoff_config = MJLabRecoveryHandoffConfig(control_dt_sec=world.control_dt_sec)
    handoff = MJLabRecoveryHandoff(
        config=handoff_config,
        environment_count=len(selected),
        device=torch_device,
    )
    environment.qpos[:, :36] = torch.as_tensor(
        [item.qpos.tolist() for item in selected], device=torch_device
    )
    environment.qvel[:, :35] = torch.as_tensor(
        [item.qvel.tolist() for item in selected], device=torch_device
    )
    environment._park_ball()
    environment.mjw.forward(environment.model, environment.data)
    torso_body = int(environment.cpu_model.body("torso_link").id)
    kp = torch.as_tensor(contract.joint_stiffness, device=torch_device)
    kd = torch.as_tensor(contract.joint_damping, device=torch_device)
    active = torch.ones(len(selected), dtype=torch.bool, device=torch_device)
    nonfinite = torch.zeros_like(active)
    final_stable_streak = torch.zeros(len(selected), dtype=torch.long, device=torch_device)
    maximum_stable_streak = torch.zeros_like(final_stable_streak)
    maximum_root_angular_speed = torch.linalg.vector_norm(
        environment.qvel[:, 3:6], dim=1
    )
    post_absorption_linear_speed = torch.full(
        (len(selected),), float("nan"), device=torch_device
    )
    post_absorption_angular_speed = torch.full_like(
        post_absorption_linear_speed, float("nan")
    )
    phase_estimator_weights: dict[str, float] | None = None
    trace: list[dict[str, Any]] = []
    steps = int(round(active_config.duration_sec / world.control_dt_sec)) + 1
    for step in range(steps):
        time = torch.full(
            (len(selected),), step * world.control_dt_sec, device=torch_device
        )
        quaternion = environment.qpos[:, 3:7]
        upright = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
        linear_speed = torch.linalg.vector_norm(environment.qvel[:, :3], dim=1)
        angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
        left_foot, right_foot = environment._foot_contact_state()
        absorbing = active & (time < active_config.impact_absorption_sec)
        expert_active = active & ~absorbing
        absorption_finished = (
            active
            & (time >= active_config.impact_absorption_sec)
            & torch.isnan(post_absorption_angular_speed)
        )
        post_absorption_linear_speed.copy_(
            torch.where(absorption_finished, linear_speed, post_absorption_linear_speed)
        )
        post_absorption_angular_speed.copy_(
            torch.where(absorption_finished, angular_speed, post_absorption_angular_speed)
        )
        if active_config.align_getup_reference_phase and bool(absorption_finished.any()):
            estimated_frame, phase_estimator_weights = (
                estimate_mjlab_getup_reference_frame_torch(
                    references=references,
                    canonical_joint_position=environment.qpos[:, 7:36],
                    canonical_joint_velocity=environment.qvel[:, 6:35],
                    pelvis_height_m=environment.qpos[:, 2],
                    pelvis_quaternion_wxyz=environment.qpos[:, 3:7],
                )
            )
            controller.set_initial_reference_frame_before_start(
                estimated_frame,
                mask=absorption_finished,
            )
        signals = handoff.update(
            pelvis_height_m=environment.qpos[:, 2],
            upright_projection=upright,
            root_linear_speed_mps=linear_speed,
            root_angular_speed_rad_s=angular_speed,
            left_foot_supported=left_foot,
            right_foot_supported=right_foot,
            active=expert_active,
        )
        expert_target, _ = controller.target(
            canonical_joint_position=environment.qpos[:, 7:36],
            canonical_joint_velocity=environment.qvel[:, 6:35],
            torso_quaternion_wxyz=environment.xquat[:, torso_body],
            base_angular_velocity_body_rad_s=environment.qvel[:, 3:6],
            relative_time_sec=torch.clamp(
                time - active_config.impact_absorption_sec, min=0.0
            )
            * handoff_config.expert_time_scale,
            active=expert_active,
        )
        reset = signals.reset_locomotion
        environment._loco_hidden[:, reset] = 0.0
        environment._loco_cell[:, reset] = 0.0
        environment._loco_action[reset] = 0.0
        not_warming = ~signals.warm_locomotion
        environment._loco_hidden[:, not_warming] = 0.0
        environment._loco_cell[:, not_warming] = 0.0
        environment._loco_action[not_warming] = 0.0
        ready_target = environment._locomotion_target(
            torch.zeros(len(selected), device=torch_device)
        )
        environment._loco_hidden[:, not_warming] = 0.0
        environment._loco_cell[:, not_warming] = 0.0
        environment._loco_action[not_warming] = 0.0
        blend = signals.blend_fraction[:, None]
        target = expert_target + blend * (ready_target - expert_target)
        control_kp = kp + blend * (environment._kp - kp)
        control_kd = kd + blend * (environment._kd - kd)
        target = torch.where(absorbing[:, None], environment.qpos[:, 7:36], target)
        control_kp = torch.where(absorbing[:, None], torch.zeros_like(control_kp), control_kp)
        control_kd = torch.where(
            absorbing[:, None],
            active_config.impact_joint_damping_scale * kd,
            control_kd,
        )
        for _ in range(world.physics_substeps):
            torque = control_kp * (target - environment.qpos[:, 7:36])
            torque -= control_kd * environment.qvel[:, 6:35]
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
        newly_nonfinite = active & ~finite
        nonfinite |= newly_nonfinite
        active &= finite
        if bool(newly_nonfinite.any()):
            environment._quarantined |= newly_nonfinite
            environment._restore_quarantined_worlds()
        quaternion = environment.qpos[:, 3:7]
        upright = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
        linear_speed = torch.linalg.vector_norm(environment.qvel[:, :3], dim=1)
        angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
        maximum_root_angular_speed = torch.maximum(maximum_root_angular_speed, angular_speed)
        left_foot, right_foot = environment._foot_contact_state()
        stable_now = (
            active
            & (environment.qpos[:, 2] >= 0.62)
            & (upright >= 0.80)
            & (linear_speed <= 0.50)
            & (angular_speed <= 1.50)
            & left_foot
            & right_foot
            & (signals.blend_fraction >= 1.0)
        )
        final_stable_streak = torch.where(
            stable_now,
            final_stable_streak + 1,
            torch.zeros_like(final_stable_streak),
        )
        maximum_stable_streak = torch.maximum(maximum_stable_streak, final_stable_streak)
        if step % 50 == 0 or step == steps - 1:
            trace.append(
                {
                    "time_sec": round((step + 1) * world.control_dt_sec, 3),
                    "active_count": int(active.sum()),
                    "mean_pelvis_height_m": float(environment.qpos[active, 2].mean())
                    if bool(active.any())
                    else None,
                    "handoff_started_count": int(signals.handoff_started.sum()),
                    "handoff_completed_count": int((signals.blend_fraction >= 1.0).sum()),
                    "stable_count": int(stable_now.sum()),
                }
            )
    required_steps = int(
        math.ceil(active_config.minimum_final_stable_sec / world.control_dt_sec)
    )
    final_stable = final_stable_streak >= required_steps
    rows = []
    for index, snapshot in enumerate(selected):
        rows.append(
            {
                "snapshot_hash": snapshot.snapshot_hash,
                "source_stage": snapshot.stage,
                "source_cluster": snapshot.posture_cluster,
                "source_pelvis_height_m": float(snapshot.qpos[2]),
                "source_root_linear_speed_mps": float(
                    math.sqrt(sum(float(value) ** 2 for value in snapshot.qvel[:3]))
                ),
                "source_root_angular_speed_rad_s": float(
                    math.sqrt(sum(float(value) ** 2 for value in snapshot.qvel[3:6]))
                ),
                "post_absorption_root_linear_speed_mps": (
                    float(post_absorption_linear_speed[index])
                    if math.isfinite(float(post_absorption_linear_speed[index]))
                    else None
                ),
                "post_absorption_root_angular_speed_rad_s": (
                    float(post_absorption_angular_speed[index])
                    if math.isfinite(float(post_absorption_angular_speed[index]))
                    else None
                ),
                "initial_reference_frame": int(controller.initial_reference_frame[index]),
                "handoff_started": bool(handoff.handoff_step[index] >= 0),
                "handoff_completed": bool(signals.blend_fraction[index] >= 1.0),
                "final_stable": bool(final_stable[index]),
                "final_continuous_stable_sec": round(
                    int(final_stable_streak[index]) * world.control_dt_sec, 3
                ),
                "maximum_post_handoff_stable_sec": round(
                    int(maximum_stable_streak[index]) * world.control_dt_sec, 3
                ),
                "maximum_root_angular_speed_rad_s": float(
                    maximum_root_angular_speed[index]
                ),
                "nonfinite": bool(nonfinite[index]),
            }
        )
    final_count = int(final_stable.sum())
    report: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_snapshot_exam_report.v1",
        "exam_config": asdict(active_config),
        "exam_config_hash": active_config.config_hash,
        "snapshot_corpus_hash": hash_json([item.snapshot_hash for item in selected]),
        "getup_contract_hash": contract.contract_hash,
        "getup_checkpoint_hash": contract.checkpoint_hash,
        "getup_reference_phase_alignment": active_config.align_getup_reference_phase,
        "getup_reference_phase_estimator_weights": phase_estimator_weights,
        "initial_reference_frames": [
            int(value) for value in controller.initial_reference_frame.tolist()
        ],
        "body_hash": contract.body_hash,
        "physics_scene_hash": contract.physics_scene_hash,
        "physics_backend": "mujoco_warp_no_auto_reset",
        "physics_device": physics_device,
        "snapshot_count": len(selected),
        "handoff_started_count": int((handoff.handoff_step >= 0).sum()),
        "handoff_completed_count": int((signals.blend_fraction >= 1.0).sum()),
        "final_stable_count": final_count,
        "final_stable_rate": final_count / len(selected),
        "nonfinite_count": int(nonfinite.sum()),
        "rows": rows,
        "trace": trace,
        "promotion_status": (
            "PASS_SIM_SNAPSHOT_COMPONENT"
            if final_count == len(selected) and not bool(nonfinite.any())
            else "REJECTED_OUT_OF_DISTRIBUTION_RECOVERY_STATES"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--getup-checkpoint", type=Path, required=True)
    parser.add_argument("--getup-source", type=Path, required=True)
    parser.add_argument("--getup-config", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stages", default="FAILURE_TERMINAL")
    parser.add_argument("--maximum-snapshots", type=int, default=64)
    parser.add_argument("--impact-absorption-sec", type=float, default=0.0)
    parser.add_argument("--impact-joint-damping-scale", type=float, default=1.0)
    parser.add_argument("--align-getup-reference-phase", action="store_true")
    args = parser.parse_args()
    report = run_recovery_snapshot_exam(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        getup_checkpoint_path=args.getup_checkpoint,
        getup_source_path=args.getup_source,
        getup_config_path=args.getup_config,
        snapshot_manifest_path=args.snapshot_manifest,
        output_path=args.output,
        device=args.device,
        config=RecoverySnapshotExamConfig(
            stages=tuple(item for item in args.stages.split(",") if item),
            maximum_snapshots=args.maximum_snapshots,
            impact_absorption_sec=args.impact_absorption_sec,
            impact_joint_damping_scale=args.impact_joint_damping_scale,
            align_getup_reference_phase=args.align_getup_reference_phase,
        ),
    )
    printable = {key: value for key, value in report.items() if key != "trace"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RecoverySnapshotExamConfig", "run_recovery_snapshot_exam"]
