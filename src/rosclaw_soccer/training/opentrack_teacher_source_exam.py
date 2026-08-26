"""Replay CPU-proven OpenTrack recovery routes in the source football scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx_routes import (
    validate_recovery_mjx_route_manifest,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def validate_opentrack_teacher_source_exam_report(path: Path) -> dict[str, Any]:
    """Load source-scene evidence and fail closed on integrity or claim drift."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OpenTrack source exam report must be an object")
    declared_hash = payload.pop("report_hash", None)
    rows = payload.get("rows")
    count = payload.get("environment_count")
    compiled = payload.get("compiled_model_contract")
    if not isinstance(compiled, dict):
        raise ValueError("OpenTrack source exam compiled model contract is absent")
    compiled_without_hash = dict(compiled)
    compiled_hash = compiled_without_hash.pop("model_hash", None)
    if (
        payload.get("schema_version") != "rosclaw.opentrack_teacher_source_exam.v1"
        or declared_hash != hash_json(payload)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(rows, list)
        or len(rows) != count
        or compiled_hash != hash_json(compiled_without_hash)
        or payload.get("source_scene_hash") != compiled_hash
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
    ):
        raise ValueError("OpenTrack source exam report integrity failed")
    if [item.get("snapshot_index") for item in rows if isinstance(item, dict)] != list(
        range(count)
    ) or len({item.get("snapshot_hash") for item in rows if isinstance(item, dict)}) != count:
        raise ValueError("OpenTrack source exam state coverage is invalid")
    stable_count = sum(bool(item.get("final_stable")) for item in rows)
    ready_count = sum(bool(item.get("ready_reached")) for item in rows)
    nonfinite_count = sum(bool(item.get("nonfinite")) for item in rows)
    if (
        payload.get("final_stable_count") != stable_count
        or payload.get("ready_count") != ready_count
        or payload.get("nonfinite_count") != nonfinite_count
        or not math.isclose(float(payload.get("final_stable_rate", -1.0)), stable_count / count)
    ):
        raise ValueError("OpenTrack source exam summary is inconsistent")
    expected_decision = (
        "SOURCE_SCENE_TEACHER_REACHABILITY_SUPPORTED"
        if stable_count == count and nonfinite_count == 0
        else "SOURCE_SCENE_TEACHER_TRANSFER_INCOMPLETE"
    )
    if payload.get("decision") != expected_decision:
        raise ValueError("OpenTrack source exam decision is inconsistent")
    if expected_decision == "SOURCE_SCENE_TEACHER_REACHABILITY_SUPPORTED" and (
        payload.get("expert_chain")
        != "OPENTRACK_MULTIPOSTURE_GETUP_TO_PHASE_ALIGNED_CAPTURE_TO_LOCOMOTION"
        or any(item.get("capture_entry_step") is None for item in rows)
        or any(item.get("capture_handoff_completed") is not True for item in rows)
        or any(
            not (
                item.get("capture_entry_left_foot_supported") is True
                or item.get("capture_entry_right_foot_supported") is True
            )
            for item in rows
        )
        or any(float(item.get("final_continuous_stable_sec", 0.0)) < 2.0 for item in rows)
    ):
        raise ValueError("OpenTrack source exam supported claim lacks full MoE evidence")
    payload["report_hash"] = declared_hash
    return payload


def run_opentrack_teacher_source_exam(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    teacher_policy_path: Path,
    teacher_config_path: Path,
    motion_path: Path,
    snapshot_manifest_path: Path,
    route_manifest_path: Path,
    output_path: Path,
    device: str = "cuda:0",
    maximum_duration_sec: float = 40.0,
    capture_checkpoint_path: Path | None = None,
    capture_source_path: Path | None = None,
    capture_config_path: Path | None = None,
    trajectory_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the selected routes without auto-reset in the source stadium."""

    import torch

    from rosclaw_soccer.growth.mjlab_getup import (
        MJLabGetUpTorchController,
        MJLabRecoveryHandoff,
        MJLabRecoveryHandoffConfig,
        estimate_mjlab_getup_reference_frame_torch,
        load_mjlab_getup_torch,
    )
    from rosclaw_soccer.growth.opentrack_tracking import (
        OPENTRACK_JOINT_DAMPING,
        OPENTRACK_JOINT_STIFFNESS,
        load_opentrack_tracking_torch,
        opentrack_tracking_observation_torch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import (
        GoalkeeperMJWarpBatch,
        goalkeeper_world_config,
    )
    from rosclaw_soccer.training.recovery_mjx import compiled_mujoco_model_contract

    if not math.isfinite(maximum_duration_sec) or not 20.0 <= maximum_duration_sec <= 60.0:
        raise ValueError("OpenTrack source exam duration is invalid")
    capture_paths = (
        capture_checkpoint_path,
        capture_source_path,
        capture_config_path,
    )
    capture_enabled = all(path is not None for path in capture_paths)
    if any(path is not None for path in capture_paths) != capture_enabled:
        raise ValueError("OpenTrack source exam capture assets must be all present or absent")
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    route_path = route_manifest_path.expanduser().resolve()
    route_manifest = validate_recovery_mjx_route_manifest(route_path)
    snapshots = load_recovery_snapshot_corpus(snapshot_path)
    routes = sorted(route_manifest["routes"], key=lambda item: int(item["snapshot_index"]))
    if (
        route_manifest["snapshot_manifest_hash"] != hash_bytes(snapshot_path.read_bytes())
        or len(routes) != len(snapshots)
        or [int(item["snapshot_index"]) for item in routes] != list(range(len(snapshots)))
        or [str(item["snapshot_hash"]) for item in routes]
        != [item.snapshot_hash for item in snapshots]
    ):
        raise ValueError("OpenTrack source exam route/snapshot binding is invalid")
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    policy, contract, references = load_opentrack_tracking_torch(
        policy_path=teacher_policy_path,
        config_path=teacher_config_path,
        motion_path=motion_path,
        device=torch_device,
    )
    if (
        contract.policy_hash != route_manifest["teacher_policy_hash"]
        or contract.config_hash != route_manifest["teacher_config_hash"]
        or any(contract.motion_hash != item["motion_source_hash"] for item in routes)
    ):
        raise ValueError("OpenTrack source exam teacher binding is invalid")
    count = len(snapshots)
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=count,
        second_shot_probability=0.0,
        shot_intent_cue_enabled=True,
    )
    environment = GoalkeeperMJWarpBatch(
        asset_root=asset_root.expanduser().resolve(),
        locomotion_policy_path=locomotion_policy_path.expanduser().resolve(),
        device=torch_device,
        config=world,
    )
    if str(environment.wp.get_device()) != str(torch_device):
        raise RuntimeError("OpenTrack source exam physics device binding failed")
    environment.reset(seed=76_101)
    environment.qpos[:, :36] = torch.as_tensor(
        [item.qpos.tolist() for item in snapshots], device=torch_device
    )
    environment.qvel[:, :35] = torch.as_tensor(
        [item.qvel.tolist() for item in snapshots], device=torch_device
    )
    environment._park_ball()
    environment.mjw.forward(environment.model, environment.data)
    kp = torch.as_tensor(OPENTRACK_JOINT_STIFFNESS, device=torch_device)
    kd = torch.as_tensor(OPENTRACK_JOINT_DAMPING, device=torch_device)
    entry_frame = torch.as_tensor(
        [int(item["entry_frame"]) for item in routes], dtype=torch.long, device=torch_device
    )
    time_dilation = torch.as_tensor(
        [int(item["time_dilation"]) for item in routes],
        dtype=torch.long,
        device=torch_device,
    )
    maximum_reference_frame = references["qpos"].shape[0] - 1
    if bool((entry_frame < 0).any()) or bool((entry_frame >= maximum_reference_frame).any()):
        raise ValueError("OpenTrack source exam route frame is invalid")
    previous_target = environment.qpos[:, 7:36].clone()
    capture_contract = None
    capture_references = None
    capture_controller = None
    capture_handoff = None
    capture_handoff_config = None
    capture_phase_weights: dict[str, float] | None = None
    capture_entry_step = torch.full((count,), -1, dtype=torch.long, device=torch_device)
    if capture_enabled:
        checkpoint_file = capture_checkpoint_path
        source_file = capture_source_path
        config_file = capture_config_path
        if checkpoint_file is None or source_file is None or config_file is None:
            raise RuntimeError("capture asset validation lost its invariant")
        capture_policy, capture_contract, capture_references = load_mjlab_getup_torch(
            checkpoint_path=checkpoint_file,
            source_path=source_file,
            config_path=config_file,
            asset_root=asset_root,
            device=torch_device,
        )
        capture_controller = MJLabGetUpTorchController(
            policy=capture_policy,
            contract=capture_contract,
            references=capture_references,
            environment_count=count,
            device=torch_device,
        )
        capture_handoff_config = MJLabRecoveryHandoffConfig(control_dt_sec=world.control_dt_sec)
        capture_handoff = MJLabRecoveryHandoff(
            config=capture_handoff_config,
            environment_count=count,
            device=torch_device,
        )
    torso_body = int(environment.cpu_model.body("torso_link").id)
    frozen_frame = torch.full((count,), -1, dtype=torch.long, device=torch_device)
    ready_pose_streak = torch.zeros(count, dtype=torch.long, device=torch_device)
    ever_ready_pose = torch.zeros(count, dtype=torch.bool, device=torch_device)
    capture_entry_linear_speed = torch.full((count,), float("nan"), device=torch_device)
    capture_entry_angular_speed = torch.full_like(capture_entry_linear_speed, float("nan"))
    capture_entry_left_foot = torch.zeros(count, dtype=torch.bool, device=torch_device)
    capture_entry_right_foot = torch.zeros_like(capture_entry_left_foot)
    final_stable_streak = torch.zeros_like(ready_pose_streak)
    maximum_stable_streak = torch.zeros_like(ready_pose_streak)
    nonfinite = torch.zeros(count, dtype=torch.bool, device=torch_device)
    peak_angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
    trace: list[dict[str, Any]] = []
    trajectory_qpos = [environment.qpos.detach().cpu().numpy().copy()]
    trajectory_time = [0.0]
    steps = int(round(maximum_duration_sec / world.control_dt_sec))
    for step in range(steps):
        capture_active = capture_entry_step >= 0
        progressing_frame = (
            entry_frame
            + 1
            + torch.div(torch.full_like(entry_frame, step), time_dilation, rounding_mode="floor")
        )
        progressing_frame = torch.clamp(progressing_frame, max=maximum_reference_frame)
        frame = torch.where(frozen_frame >= 0, frozen_frame, progressing_frame)
        reference_qpos = references["qpos"][frame]
        reference_qvel = references["qvel"][frame]
        observation = opentrack_tracking_observation_torch(
            canonical_joint_position=environment.qpos[:, 7:36],
            canonical_joint_velocity=environment.qvel[:, 6:35],
            pelvis_quaternion_wxyz=environment.qpos[:, 3:7],
            root_angular_velocity_body_rad_s=environment.qvel[:, 3:6],
            previous_motor_target=previous_target,
            reference_joint_position=reference_qpos[:, 7:36],
            reference_joint_velocity=reference_qvel[:, 6:35],
            reference_feet_height_m=references["feet_height"][frame],
            reference_root_height_m=reference_qpos[:, 2],
        )
        with torch.no_grad():
            action = policy(observation)
        opentrack_target = reference_qpos[:, 7:36] + action
        previous_target.copy_(opentrack_target)
        target = opentrack_target
        control_kp = kp
        control_kd = kd
        capture_blend = torch.zeros(count, device=torch_device)
        if capture_enabled:
            if (
                capture_controller is None
                or capture_handoff is None
                or capture_handoff_config is None
                or capture_contract is None
            ):
                raise RuntimeError("capture runtime is incomplete")
            quaternion = environment.qpos[:, 3:7]
            upright_before = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
            linear_before = torch.linalg.vector_norm(environment.qvel[:, :3], dim=1)
            angular_before = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
            left_foot_before, right_foot_before = environment._foot_contact_state()
            capture_signals = capture_handoff.update(
                pelvis_height_m=environment.qpos[:, 2],
                upright_projection=upright_before,
                root_linear_speed_mps=linear_before,
                root_angular_speed_rad_s=angular_before,
                left_foot_supported=left_foot_before,
                right_foot_supported=right_foot_before,
                active=capture_active,
            )
            capture_time = torch.clamp(
                (step - capture_entry_step).to(torch.float32) * world.control_dt_sec,
                min=0.0,
            )
            capture_target, _ = capture_controller.target(
                canonical_joint_position=environment.qpos[:, 7:36],
                canonical_joint_velocity=environment.qvel[:, 6:35],
                torso_quaternion_wxyz=environment.xquat[:, torso_body],
                base_angular_velocity_body_rad_s=environment.qvel[:, 3:6],
                relative_time_sec=(capture_time * capture_handoff_config.expert_time_scale),
                active=capture_active,
            )
            reset = capture_signals.reset_locomotion
            environment._loco_hidden[:, reset] = 0.0
            environment._loco_cell[:, reset] = 0.0
            environment._loco_action[reset] = 0.0
            not_warming = capture_active & ~capture_signals.warm_locomotion
            environment._loco_hidden[:, not_warming] = 0.0
            environment._loco_cell[:, not_warming] = 0.0
            environment._loco_action[not_warming] = 0.0
            ready_target = environment._locomotion_target(torch.zeros(count, device=torch_device))
            environment._loco_hidden[:, not_warming] = 0.0
            environment._loco_cell[:, not_warming] = 0.0
            environment._loco_action[not_warming] = 0.0
            capture_blend = capture_signals.blend_fraction
            blend = capture_blend[:, None]
            capture_target = capture_target + blend * (ready_target - capture_target)
            capture_kp = torch.as_tensor(capture_contract.joint_stiffness, device=torch_device)
            capture_kd = torch.as_tensor(capture_contract.joint_damping, device=torch_device)
            capture_kp = capture_kp + blend * (environment._kp - capture_kp)
            capture_kd = capture_kd + blend * (environment._kd - capture_kd)
            target = torch.where(capture_active[:, None], capture_target, target)
            control_kp = torch.where(capture_active[:, None], capture_kp, kp)
            control_kd = torch.where(capture_active[:, None], capture_kd, kd)
        for _ in range(world.physics_substeps):
            torque = control_kp * (target - environment.qpos[:, 7:36])
            torque -= control_kd * environment.qvel[:, 6:35]
            environment.ctrl.copy_(
                torch.clamp(torque, -environment._torque_limits, environment._torque_limits)
            )
            environment.mjw.step(environment.model, environment.data)
        finite = torch.all(torch.isfinite(environment.qpos), dim=1)
        finite &= torch.all(torch.isfinite(environment.qvel), dim=1)
        nonfinite |= ~finite
        quaternion = environment.qpos[:, 3:7]
        upright = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
        linear_speed = torch.linalg.vector_norm(environment.qvel[:, :3], dim=1)
        angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
        peak_angular_speed = torch.maximum(peak_angular_speed, angular_speed)
        ready_pose = finite & (environment.qpos[:, 2] >= 0.62) & (upright >= 0.75)
        ever_ready_pose |= ready_pose
        left_foot, right_foot = environment._foot_contact_state()
        # Keep the observable ready signal independent from the stricter
        # transition gate; in-place masking an alias would erase diagnostics.
        entry_candidate = ready_pose.clone()
        if capture_enabled:
            entry_candidate &= linear_speed <= 0.75
            entry_candidate &= angular_speed <= 2.0
            entry_candidate &= left_foot | right_foot
        ready_pose_streak = torch.where(
            entry_candidate,
            ready_pose_streak + 1,
            torch.zeros_like(ready_pose_streak),
        )
        newly_frozen = (frozen_frame < 0) & (ready_pose_streak >= 10)
        frozen_frame.copy_(torch.where(newly_frozen, frame, frozen_frame))
        if capture_enabled and bool(newly_frozen.any()):
            if capture_controller is None or capture_references is None:
                raise RuntimeError("capture phase adapter is incomplete")
            estimated_frame, capture_phase_weights = estimate_mjlab_getup_reference_frame_torch(
                references=capture_references,
                canonical_joint_position=environment.qpos[:, 7:36],
                canonical_joint_velocity=environment.qvel[:, 6:35],
                pelvis_height_m=environment.qpos[:, 2],
                pelvis_quaternion_wxyz=environment.qpos[:, 3:7],
            )
            capture_controller.set_initial_reference_frame_before_start(
                estimated_frame,
                mask=newly_frozen,
            )
            capture_entry_step.copy_(
                torch.where(
                    newly_frozen,
                    torch.full_like(capture_entry_step, step + 1),
                    capture_entry_step,
                )
            )
            capture_entry_linear_speed.copy_(
                torch.where(newly_frozen, linear_speed, capture_entry_linear_speed)
            )
            capture_entry_angular_speed.copy_(
                torch.where(newly_frozen, angular_speed, capture_entry_angular_speed)
            )
            capture_entry_left_foot |= newly_frozen & left_foot
            capture_entry_right_foot |= newly_frozen & right_foot
        capture_completed = (
            capture_blend >= 1.0
            if capture_enabled
            else torch.ones(count, dtype=torch.bool, device=torch_device)
        )
        stable = ready_pose & (linear_speed <= 0.50) & (angular_speed <= 1.50)
        stable &= capture_completed
        final_stable_streak = torch.where(
            stable,
            final_stable_streak + 1,
            torch.zeros_like(final_stable_streak),
        )
        maximum_stable_streak = torch.maximum(maximum_stable_streak, final_stable_streak)
        if step % 50 == 0 or step == steps - 1:
            trace.append(
                {
                    "time_sec": round((step + 1) * world.control_dt_sec, 3),
                    "ready_count": int(ready_pose.sum()),
                    "capture_entry_candidate_count": int(entry_candidate.sum()),
                    "reference_frozen_count": int((frozen_frame >= 0).sum()),
                    "capture_active_count": int((capture_entry_step >= 0).sum()),
                    "capture_handoff_completed_count": int(capture_completed.sum()),
                    "stable_two_sec_count": int((final_stable_streak >= 100).sum()),
                    "mean_pelvis_height_m": float(environment.qpos[:, 2].mean()),
                }
            )
        if (step + 1) % 2 == 0:
            trajectory_qpos.append(environment.qpos.detach().cpu().numpy().copy())
            trajectory_time.append((step + 1) * world.control_dt_sec)
    success = final_stable_streak >= 100
    rows = [
        {
            "snapshot_index": index,
            "snapshot_hash": snapshots[index].snapshot_hash,
            "posture_cluster": snapshots[index].posture_cluster,
            "entry_frame": int(entry_frame[index]),
            "time_dilation": int(time_dilation[index]),
            "frozen_reference_frame": (
                int(frozen_frame[index]) if int(frozen_frame[index]) >= 0 else None
            ),
            "ready_reached": bool(ever_ready_pose[index]),
            "capture_entry_step": (
                int(capture_entry_step[index]) if int(capture_entry_step[index]) >= 0 else None
            ),
            "capture_initial_reference_frame": (
                int(capture_controller.initial_reference_frame[index])
                if capture_controller is not None and int(capture_entry_step[index]) >= 0
                else None
            ),
            "capture_handoff_completed": bool(capture_blend[index] >= 1.0)
            if capture_enabled
            else None,
            "capture_entry_root_linear_speed_mps": (
                float(capture_entry_linear_speed[index])
                if math.isfinite(float(capture_entry_linear_speed[index]))
                else None
            ),
            "capture_entry_root_angular_speed_rad_s": (
                float(capture_entry_angular_speed[index])
                if math.isfinite(float(capture_entry_angular_speed[index]))
                else None
            ),
            "capture_entry_left_foot_supported": (
                bool(capture_entry_left_foot[index]) if capture_enabled else None
            ),
            "capture_entry_right_foot_supported": (
                bool(capture_entry_right_foot[index]) if capture_enabled else None
            ),
            "final_stable": bool(success[index]),
            "final_continuous_stable_sec": round(
                int(final_stable_streak[index]) * world.control_dt_sec, 3
            ),
            "maximum_stable_sec": round(
                int(maximum_stable_streak[index]) * world.control_dt_sec, 3
            ),
            "peak_root_angular_speed_rad_s": float(peak_angular_speed[index]),
            "nonfinite": bool(nonfinite[index]),
        }
        for index in range(count)
    ]
    success_count = int(success.sum())
    compiled_model_contract = compiled_mujoco_model_contract(environment.cpu_model)
    trajectory_archive = None
    if trajectory_output_path is not None:
        import numpy as np

        trajectory_file = trajectory_output_path.expanduser().resolve()
        if trajectory_file.suffix != ".npz" or trajectory_file.exists():
            raise ValueError("OpenTrack source exam trajectory requires a new NPZ path")
        trajectory_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_file,
            time=np.asarray(trajectory_time, dtype=np.float64),
            qpos=np.asarray(trajectory_qpos, dtype=np.float32),
        )
        trajectory_archive = {
            "path": trajectory_file.name,
            "hash": hash_bytes(trajectory_file.read_bytes()),
            "frame_count": len(trajectory_time),
            "frame_rate_hz": 25.0,
            "contains_scored_qpos_only": True,
        }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.opentrack_teacher_source_exam.v1",
        "teacher_contract_hash": contract.contract_hash,
        "teacher_policy_hash": contract.policy_hash,
        "teacher_config_hash": contract.config_hash,
        "motion_hash": contract.motion_hash,
        "route_manifest_hash": route_manifest["report_hash"],
        "snapshot_manifest_hash": hash_bytes(snapshot_path.read_bytes()),
        "source_scene_hash": compiled_model_contract["model_hash"],
        "compiled_model_contract": compiled_model_contract,
        "physics_backend": "mujoco_warp_source_stadium_no_auto_reset",
        "physics_device": str(environment.wp.get_device()),
        "environment_count": count,
        "random_seed": 76_101,
        "duration_sec": maximum_duration_sec,
        "ready_count": int(ever_ready_pose.sum()),
        "final_stable_count": success_count,
        "final_stable_rate": success_count / count,
        "nonfinite_count": int(nonfinite.sum()),
        "rows": rows,
        "trace": trace,
        "trajectory_archive": trajectory_archive,
        "teacher_role": "PRIVILEGED_REFERENCE_CONDITIONED_SOURCE_SCENE_ORACLE",
        "expert_chain": (
            "OPENTRACK_MULTIPOSTURE_GETUP_TO_PHASE_ALIGNED_CAPTURE_TO_LOCOMOTION"
            if capture_enabled
            else "OPENTRACK_MULTIPOSTURE_GETUP_ONLY"
        ),
        "capture_contract_hash": (
            capture_contract.contract_hash if capture_contract is not None else None
        ),
        "capture_reference_phase_estimator_weights": capture_phase_weights,
        "capture_transition_gate": (
            {
                "minimum_pelvis_height_m": 0.62,
                "minimum_upright_projection": 0.75,
                "maximum_root_linear_speed_mps": 0.75,
                "maximum_root_angular_speed_rad_s": 2.0,
                "require_any_foot_support": True,
                "continuous_hold_sec": 10 * world.control_dt_sec,
                "feature_contract": "ENTRY_PROPRIOCEPTION_AND_CONTACTS_ONLY",
            }
            if capture_enabled
            else None
        ),
        "reference_phase_reads_during_control": steps * count,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "SOURCE_SCENE_REACHABILITY_ORACLE_NOT_DEPLOYABLE_ACTOR",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["decision"] = (
        "SOURCE_SCENE_TEACHER_REACHABILITY_SUPPORTED"
        if success_count == count and not bool(nonfinite.any())
        else "SOURCE_SCENE_TEACHER_TRANSFER_INCOMPLETE"
    )
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--teacher-policy", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-duration-sec", type=float, default=40.0)
    parser.add_argument("--capture-checkpoint", type=Path)
    parser.add_argument("--capture-source", type=Path)
    parser.add_argument("--capture-config", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
    args = parser.parse_args()
    report = run_opentrack_teacher_source_exam(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        teacher_policy_path=args.teacher_policy,
        teacher_config_path=args.teacher_config,
        motion_path=args.motion,
        snapshot_manifest_path=args.snapshot_manifest,
        route_manifest_path=args.route_manifest,
        output_path=args.output,
        device=args.device,
        maximum_duration_sec=args.maximum_duration_sec,
        capture_checkpoint_path=args.capture_checkpoint,
        capture_source_path=args.capture_source,
        capture_config_path=args.capture_config,
        trajectory_output_path=args.trajectory_output,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "final_stable_count": report["final_stable_count"],
                "environment_count": report["environment_count"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_opentrack_teacher_source_exam",
    "validate_opentrack_teacher_source_exam_report",
]
