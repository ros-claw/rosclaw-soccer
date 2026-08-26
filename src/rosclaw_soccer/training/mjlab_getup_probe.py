"""SIM_ONLY MuJoCo-Warp exam for the MuJoCo-native neural get-up expert."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx import (
    validate_recovery_mjx_failure_state_manifest,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _quaternion_multiply(torch: Any, left: Any, right: Any) -> Any:
    lw, lx, ly, lz = left.unbind(dim=1)
    rw, rx, ry, rz = right.unbind(dim=1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=1,
    )


def _small_euler_quaternion(torch: Any, angles: Any) -> Any:
    half = 0.5 * angles
    cr, cp, cy = torch.cos(half).unbind(dim=1)
    sr, sp, sy = torch.sin(half).unbind(dim=1)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=1,
    )


def route_recovery_entry_torch(
    *,
    torch: Any,
    pelvis_height_m: Any,
    upright_projection: Any,
    root_angular_speed_rad_s: Any,
    route_upright_to_locomotion: bool,
    capture_router_enabled: bool,
    capture_maximum_pelvis_height_m: float,
    capture_minimum_root_angular_speed_rad_s: float,
) -> tuple[Any, Any, Any]:
    """Return mutually exclusive Athlete, Capture and Get-up entry routes."""

    count = int(pelvis_height_m.shape[0])
    if (
        tuple(pelvis_height_m.shape) != (count,)
        or tuple(upright_projection.shape) != (count,)
        or tuple(root_angular_speed_rad_s.shape) != (count,)
    ):
        raise ValueError("recovery entry-router input shape is invalid")
    athlete_candidate = (
        (pelvis_height_m >= 0.68) & (upright_projection >= 0.85)
        if route_upright_to_locomotion
        else torch.zeros(count, dtype=torch.bool, device=pelvis_height_m.device)
    )
    capture_route = (
        athlete_candidate
        & (pelvis_height_m <= capture_maximum_pelvis_height_m)
        & (root_angular_speed_rad_s >= capture_minimum_root_angular_speed_rad_s)
        if capture_router_enabled
        else torch.zeros_like(athlete_candidate)
    )
    athlete_route = athlete_candidate & ~capture_route
    getup_route = ~athlete_candidate
    coverage = athlete_route.to(torch.int8)
    coverage += capture_route.to(torch.int8)
    coverage += getup_route.to(torch.int8)
    if not bool((coverage == 1).all()):
        raise RuntimeError("recovery entry routes are not mutually exclusive and complete")
    return athlete_route, capture_route, getup_route


def run_mjlab_getup_physics_probe(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    checkpoint_path: Path,
    source_path: Path,
    config_path: Path,
    output_path: Path,
    environment_count: int = 4,
    seed: int = 47_201,
    device: str = "cuda:0",
    initial_perturbation_scale: float = 0.0,
    failure_state_manifest_path: Path | None = None,
    failure_state_start_index: int = 0,
    route_upright_to_locomotion: bool = False,
    align_getup_reference_phase: bool = False,
    capture_router_max_pelvis_height_m: float | None = None,
    capture_router_min_root_angular_speed_rad_s: float | None = None,
) -> dict[str, Any]:
    """Execute get-up or a seed get-up/Athlete MoE without auto-reset."""

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

    if not 1 <= environment_count <= 64:
        raise ValueError("MJLab get-up probe environment count must be in [1, 64]")
    if not 0.0 <= initial_perturbation_scale <= 0.15:
        raise ValueError("MJLab get-up initial perturbation scale must be in [0, 0.15]")
    capture_router_enabled = (
        capture_router_max_pelvis_height_m is not None
        and capture_router_min_root_angular_speed_rad_s is not None
    )
    capture_maximum_height = (
        float(capture_router_max_pelvis_height_m)
        if capture_router_max_pelvis_height_m is not None
        else 0.0
    )
    capture_minimum_angular_speed = (
        float(capture_router_min_root_angular_speed_rad_s)
        if capture_router_min_root_angular_speed_rad_s is not None
        else 0.0
    )
    if (
        failure_state_start_index < 0
        or not isinstance(route_upright_to_locomotion, bool)
        or not isinstance(align_getup_reference_phase, bool)
        or (
            failure_state_manifest_path is None
            and (
                failure_state_start_index != 0
                or route_upright_to_locomotion
                or align_getup_reference_phase
            )
        )
        or (failure_state_manifest_path is not None and initial_perturbation_scale != 0.0)
        or (capture_router_max_pelvis_height_m is None)
        != (capture_router_min_root_angular_speed_rad_s is None)
        or (
            capture_router_enabled
            and (
                not route_upright_to_locomotion
                or not align_getup_reference_phase
                or not 0.65 <= capture_maximum_height <= 0.75
                or not 0.50 <= capture_minimum_angular_speed <= 3.0
            )
        )
    ):
        raise ValueError("MJLab reachability initial-state contract is invalid")
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=environment_count,
        second_shot_probability=0.0,
        shot_intent_cue_enabled=True,
    )
    environment = GoalkeeperMJWarpBatch(
        asset_root=asset_root.expanduser().resolve(),
        locomotion_policy_path=locomotion_policy_path.expanduser().resolve(),
        device=torch_device,
        config=world,
    )
    physics_device = str(environment.wp.get_device())
    if physics_device != str(torch_device):
        raise RuntimeError("MJLab get-up probe physics device binding is invalid")
    environment.reset(seed=seed)
    policy, contract, references = load_mjlab_getup_torch(
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        config_path=config_path,
        asset_root=asset_root,
        device=torch_device,
    )
    failure_manifest: dict[str, Any] | None = None
    selected_failure_rows: list[dict[str, Any]] = []
    if failure_state_manifest_path is None:
        environment.qpos[:, 2] = references["pelvis_position"][0, 2]
        environment.qpos[:, 3:7] = references["pelvis_quaternion"][0]
        environment.qpos[:, 7:36] = references["joint_position"][0]
        environment.qvel.zero_()
    else:
        failure_manifest_file = failure_state_manifest_path.expanduser().resolve()
        failure_manifest = validate_recovery_mjx_failure_state_manifest(failure_manifest_file)
        stop = failure_state_start_index + environment_count
        if stop > int(failure_manifest["collected_state_count"]):
            raise ValueError("MJLab reachability state slice exceeds the bound bank")
        archive_path = failure_manifest_file.parent / str(failure_manifest["state_archive"])
        with np.load(archive_path, allow_pickle=False) as archive:
            qpos = np.array(archive["qpos"][failure_state_start_index:stop], copy=True)
            qvel = np.array(archive["qvel"][failure_state_start_index:stop], copy=True)
        if qpos.shape != (environment_count, 36) or qvel.shape != (environment_count, 35):
            raise ValueError("MJX failure bank does not contain canonical G1 29-DoF state")
        if environment.qpos.shape[1] < 36 or environment.qvel.shape[1] < 35:
            raise ValueError("MJWarp target scene cannot contain the canonical G1 state")
        # The source recovery scene contains only G1 (36 qpos / 35 qvel).
        # The stadium appends a free football.  Replay the canonical robot
        # prefix exactly while retaining the target scene's reset ball state.
        environment.qpos[:, :36].copy_(torch.as_tensor(qpos, device=torch_device))
        environment.qvel[:, :35].copy_(torch.as_tensor(qvel, device=torch_device))
        rows = failure_manifest.get("selection_rows")
        if isinstance(rows, list) and len(rows) == int(failure_manifest["collected_state_count"]):
            selected_failure_rows = [dict(row) for row in rows[failure_state_start_index:stop]]
    if initial_perturbation_scale > 0.0:
        generator = torch.Generator(device=torch_device)
        generator.manual_seed(seed + 97)

        def symmetric(shape: tuple[int, ...]) -> Any:
            return 2.0 * torch.rand(shape, generator=generator, device=torch_device) - 1.0

        environment.qpos[:, :2] += 0.02 * symmetric((environment_count, 2))
        perturbation = _small_euler_quaternion(
            torch,
            initial_perturbation_scale * symmetric((environment_count, 3)),
        )
        environment.qpos[:, 3:7] = _quaternion_multiply(
            torch,
            perturbation,
            environment.qpos[:, 3:7],
        )
        joint_noise = initial_perturbation_scale * symmetric((environment_count, 29))
        environment.qpos[:, 7:36] = torch.clamp(
            environment.qpos[:, 7:36] + joint_noise,
            environment._joint_ranges[:, 0],
            environment._joint_ranges[:, 1],
        )
        environment.qvel[:, :3] = initial_perturbation_scale * symmetric((environment_count, 3))
        environment.qvel[:, 3:6] = (
            2.0 * initial_perturbation_scale * symmetric((environment_count, 3))
        )
        environment.qvel[:, 6:35] = (
            2.0 * initial_perturbation_scale * symmetric((environment_count, 29))
        )
    environment.mjw.forward(environment.model, environment.data)
    torso_body = int(environment.cpu_model.body("torso_link").id)
    kp = torch.as_tensor(contract.joint_stiffness, device=torch_device)
    kd = torch.as_tensor(contract.joint_damping, device=torch_device)
    initial_quaternion = environment.qpos[:, 3:7]
    initial_upright = (
        2.0 * (initial_quaternion[:, 0].square() + initial_quaternion[:, 3].square()) - 1.0
    )
    initial_root_angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
    athlete_route, capture_route, getup_route = route_recovery_entry_torch(
        torch=torch,
        pelvis_height_m=environment.qpos[:, 2],
        upright_projection=initial_upright,
        root_angular_speed_rad_s=initial_root_angular_speed,
        route_upright_to_locomotion=route_upright_to_locomotion,
        capture_router_enabled=capture_router_enabled,
        capture_maximum_pelvis_height_m=capture_maximum_height,
        capture_minimum_root_angular_speed_rad_s=capture_minimum_angular_speed,
    )
    recovery_route = capture_route | getup_route
    initial_reference_frame = torch.zeros(environment_count, dtype=torch.long, device=torch_device)
    phase_estimator_weights: dict[str, float] | None = None
    if align_getup_reference_phase:
        estimated_frame, phase_estimator_weights = estimate_mjlab_getup_reference_frame_torch(
            references=references,
            canonical_joint_position=environment.qpos[:, 7:36],
            canonical_joint_velocity=environment.qvel[:, 6:35],
            pelvis_height_m=environment.qpos[:, 2],
            pelvis_quaternion_wxyz=environment.qpos[:, 3:7],
        )
        initial_reference_frame = torch.where(
            recovery_route, estimated_frame, initial_reference_frame
        )
    controller = MJLabGetUpTorchController(
        policy=policy,
        contract=contract,
        references=references,
        environment_count=environment_count,
        device=torch_device,
        initial_reference_frame=initial_reference_frame,
    )
    handoff_config = MJLabRecoveryHandoffConfig(control_dt_sec=world.control_dt_sec)
    handoff = MJLabRecoveryHandoff(
        config=handoff_config,
        environment_count=environment_count,
        device=torch_device,
    )
    final_stable_streak = torch.zeros(environment_count, dtype=torch.long, device=torch_device)
    maximum_post_handoff_stable_streak = torch.zeros_like(final_stable_streak)
    trace: list[dict[str, Any]] = []
    # The motion command ends around nine seconds.  The policy must remain in
    # closed loop at its terminal reference long enough to dissipate momentum,
    # warm locomotion and prove a two-second post-handoff standing interval.
    steps = int(round(20.0 / world.control_dt_sec)) + 1
    for step in range(steps):
        time = torch.full((environment_count,), step * world.control_dt_sec, device=torch_device)
        quaternion = environment.qpos[:, 3:7]
        upright = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
        root_linear_speed = torch.linalg.vector_norm(environment.qvel[:, :3], dim=1)
        root_angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
        left_foot, right_foot = environment._foot_contact_state()
        signals = handoff.update(
            pelvis_height_m=environment.qpos[:, 2],
            upright_projection=upright,
            root_linear_speed_mps=root_linear_speed,
            root_angular_speed_rad_s=root_angular_speed,
            left_foot_supported=left_foot,
            right_foot_supported=right_foot,
            active=recovery_route,
        )
        expert_target, _ = controller.target(
            canonical_joint_position=environment.qpos[:, 7:36],
            canonical_joint_velocity=environment.qvel[:, 6:35],
            torso_quaternion_wxyz=environment.xquat[:, torso_body],
            base_angular_velocity_body_rad_s=environment.qvel[:, 3:6],
            relative_time_sec=time * handoff_config.expert_time_scale,
            # Do not terminate the expert on the first height crossing.  Its
            # capped final reference is a learned terminal stabilizer.
            active=recovery_route,
        )
        # A recurrent locomotion controller cannot safely receive a dynamic
        # body as a cold start.  Reset it at entry to the stable envelope, feed
        # it every causal state during the hold, and withhold authority until
        # the continuous gate completes.
        reset = signals.reset_locomotion
        environment._loco_hidden[:, reset] = 0.0
        environment._loco_cell[:, reset] = 0.0
        environment._loco_action[reset] = 0.0
        not_warming = recovery_route & ~signals.warm_locomotion
        environment._loco_hidden[:, not_warming] = 0.0
        environment._loco_cell[:, not_warming] = 0.0
        environment._loco_action[not_warming] = 0.0
        ready_target = environment._locomotion_target(
            torch.zeros(environment_count, device=torch_device)
        )
        environment._loco_hidden[:, not_warming] = 0.0
        environment._loco_cell[:, not_warming] = 0.0
        environment._loco_action[not_warming] = 0.0
        effective_blend = torch.where(
            athlete_route,
            torch.ones(environment_count, device=torch_device),
            signals.blend_fraction,
        )
        blend = effective_blend[:, None]
        target = expert_target + blend * (ready_target - expert_target)
        control_kp = kp + blend * (environment._kp - kp)
        control_kd = kd + blend * (environment._kd - kd)
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
        quaternion = environment.qpos[:, 3:7]
        upright = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
        root_linear_speed = torch.linalg.vector_norm(environment.qvel[:, :3], dim=1)
        root_angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
        left_foot, right_foot = environment._foot_contact_state()
        blend_completed = effective_blend >= 1.0
        stable_now = (
            (environment.qpos[:, 2] >= 0.62)
            & (upright >= 0.80)
            & (root_linear_speed <= 0.50)
            & (root_angular_speed <= 1.50)
            & left_foot
            & right_foot
            & blend_completed
        )
        final_stable_streak = torch.where(
            stable_now,
            final_stable_streak + 1,
            torch.zeros_like(final_stable_streak),
        )
        maximum_post_handoff_stable_streak = torch.maximum(
            maximum_post_handoff_stable_streak,
            final_stable_streak,
        )
        if step % 25 == 0 or step == steps - 1:
            trace.append(
                {
                    "time_sec": round((step + 1) * world.control_dt_sec, 3),
                    "mean_pelvis_height_m": float(environment.qpos[:, 2].mean()),
                    "mean_upright_projection": float(upright.mean()),
                    "bilateral_foot_support_count": int((left_foot & right_foot).sum()),
                    "handoff_started_count": int((athlete_route | signals.handoff_started).sum()),
                    "handoff_completed_count": int(blend_completed.sum()),
                    "mean_blend_fraction": float(effective_blend.mean()),
                    "maximum_root_angular_speed_rad_s": float(
                        torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1).max()
                    ),
                }
            )
    required_final_stable_steps = int(round(2.0 / world.control_dt_sec))
    final_stable = final_stable_streak >= required_final_stable_steps
    handoff_started = athlete_route | (handoff.handoff_step >= 0)
    handoff_completed = athlete_route | (signals.blend_fraction >= 1.0)
    handoff_started_count = int(handoff_started.sum())
    handoff_completed_count = int(handoff_completed.sum())
    final_stable_count = int(final_stable.sum())
    failure_manifest_file_hash = None
    if failure_state_manifest_path is not None:
        failure_manifest_file_hash = hash_bytes(
            failure_state_manifest_path.expanduser().resolve().read_bytes()
        )
    payload = {
        "schema_version": (
            "rosclaw_soccer.mjlab_recovery_moe_reachability_probe.v1"
            if failure_manifest is not None
            else "rosclaw_soccer.mjlab_getup_physics_probe.v3"
        ),
        "contract_hash": contract.contract_hash,
        "checkpoint_hash": contract.checkpoint_hash,
        "source_hash": contract.source_hash,
        "body_hash": contract.body_hash,
        "physics_scene_hash": contract.physics_scene_hash,
        "environment_count": environment_count,
        "random_seed": seed,
        "physics_device": physics_device,
        "initial_perturbation_scale": initial_perturbation_scale,
        "failure_state_manifest_hash": (
            failure_manifest.get("report_hash") if failure_manifest else None
        ),
        "failure_state_manifest_file_hash": failure_manifest_file_hash,
        "failure_state_start_index": failure_state_start_index,
        "failure_state_stop_index": failure_state_start_index + environment_count,
        "source_compiled_model_contract_hash": (
            hash_json(failure_manifest.get("compiled_model_contract")) if failure_manifest else None
        ),
        "cross_scene_state_replay": failure_manifest is not None,
        "cross_scene_boundary": (
            "SAME_G1_29DOF_QPOS_QVEL_DIFFERENT_COLLISION_AND_ACTUATOR_SCENE"
            if failure_manifest
            else None
        ),
        "routing": (
            "RISK_ROUTED_ATHLETE_CAPTURE_GETUP"
            if capture_router_enabled
            else (
                "UPRIGHT_TO_ATHLETE_OTHERWISE_MJLAB_GETUP"
                if route_upright_to_locomotion
                else "MJLAB_GETUP_ONLY"
            )
        ),
        "capture_router_config": (
            {
                "maximum_pelvis_height_m": capture_router_max_pelvis_height_m,
                "minimum_root_angular_speed_rad_s": (capture_router_min_root_angular_speed_rad_s),
                "feature_contract": "ENTRY_PROPRIOCEPTION_ONLY",
            }
            if capture_router_enabled
            else None
        ),
        "athlete_route_count": int(athlete_route.sum()),
        "capture_route_count": int(capture_route.sum()),
        "getup_route_count": int(getup_route.sum()),
        "getup_reference_phase_alignment": align_getup_reference_phase,
        "getup_reference_phase_estimator_weights": phase_estimator_weights,
        "initial_reference_frames": [int(value) for value in initial_reference_frame.tolist()],
        "handoff_config_hash": handoff_config.config_hash,
        "handoff_config": handoff_config.__dict__,
        "handoff_started_count": handoff_started_count,
        "handoff_completed_count": handoff_completed_count,
        "transient_recovery_rate": handoff_started_count / environment_count,
        "final_stable_recovery_count": final_stable_count,
        "final_stable_recovery_rate": final_stable_count / environment_count,
        "minimum_required_final_stable_sec": 2.0,
        "final_continuous_stable_sec": [
            round(float(value) * world.control_dt_sec, 3) for value in final_stable_streak.tolist()
        ],
        "maximum_post_handoff_stable_sec": [
            round(float(value) * world.control_dt_sec, 3)
            for value in maximum_post_handoff_stable_streak.tolist()
        ],
        "state_results": [
            {
                "local_environment_index": index,
                "failure_state_index": failure_state_start_index + index,
                "state_identity": (
                    selected_failure_rows[index].get("state_identity")
                    if selected_failure_rows
                    else None
                ),
                "route": (
                    "ATHLETE"
                    if bool(athlete_route[index])
                    else ("CAPTURE" if bool(capture_route[index]) else "GET_UP")
                ),
                "initial_reference_frame": int(initial_reference_frame[index]),
                "handoff_started": bool(handoff_started[index]),
                "handoff_completed": bool(handoff_completed[index]),
                "final_stable_recovery": bool(final_stable[index]),
                "final_continuous_stable_sec": round(
                    float(final_stable_streak[index]) * world.control_dt_sec, 3
                ),
                "maximum_post_handoff_stable_sec": round(
                    float(maximum_post_handoff_stable_streak[index]) * world.control_dt_sec,
                    3,
                ),
            }
            for index in range(environment_count)
        ],
        "termination_semantics": (
            "BILATERAL_LOW_MOMENTUM_HOLD_THEN_WARM_RECURRENT_LOCOMOTION_BLEND"
        ),
        "duration_sec": controller.duration_sec,
        "claim_boundary": (
            "INDEPENDENT_MODERN_MUJOCO_EXPERT_CROSS_SCENE_REACHABILITY_DIAGNOSTIC"
            if failure_manifest
            else "LOCAL_SOURCE_POSTURE_COMPONENT_EXAM"
        ),
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "trace": trace,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["report_hash"] = hash_json(payload)
    _atomic_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47_201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-perturbation-scale", type=float, default=0.0)
    parser.add_argument("--failure-state-manifest", type=Path)
    parser.add_argument("--failure-state-start-index", type=int, default=0)
    parser.add_argument("--route-upright-to-locomotion", action="store_true")
    parser.add_argument("--align-getup-reference-phase", action="store_true")
    parser.add_argument("--capture-router-max-pelvis-height-m", type=float)
    parser.add_argument("--capture-router-min-root-angular-speed-rad-s", type=float)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    result = run_mjlab_getup_physics_probe(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        checkpoint_path=args.checkpoint,
        source_path=args.source,
        config_path=args.config,
        output_path=args.output,
        environment_count=args.environment_count,
        seed=args.seed,
        device=args.device,
        initial_perturbation_scale=args.initial_perturbation_scale,
        failure_state_manifest_path=args.failure_state_manifest,
        failure_state_start_index=args.failure_state_start_index,
        route_upright_to_locomotion=args.route_upright_to_locomotion,
        align_getup_reference_phase=args.align_getup_reference_phase,
        capture_router_max_pelvis_height_m=args.capture_router_max_pelvis_height_m,
        capture_router_min_root_angular_speed_rad_s=(
            args.capture_router_min_root_angular_speed_rad_s
        ),
    )
    printable = (
        {key: value for key, value in result.items() if key != "trace"}
        if args.summary_only
        else result
    )
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
