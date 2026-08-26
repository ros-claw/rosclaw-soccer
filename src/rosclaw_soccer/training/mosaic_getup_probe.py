"""Reproducible SIM_ONLY MuJoCo-Warp exam for one MOSAIC G1 get-up skill."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def run_mosaic_getup_physics_probe(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    targeted_dive_checkpoint: Path,
    mosaic_gmt_model_path: Path,
    mosaic_gmt_getup_skill_path: Path,
    mosaic_source_path: Path,
    output_path: Path,
    environment_count: int = 4,
    seed: int = 47_101,
    blend: float = 1.0,
    reference_feedforward_blend: float = 0.0,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Start from the bound down pose and score recovery before any reset."""

    import numpy as np
    import torch

    from rosclaw_soccer.growth.mosaic_g1_contract import (
        MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import goalkeeper_world_config
    from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
        GoalkeeperTargetedDiveMJWarpBatch,
        GoalkeeperTargetedDiveRLConfig,
    )

    if not 1 <= environment_count <= 64:
        raise ValueError("MOSAIC get-up probe environment count must be in [1, 64]")
    source = mosaic_source_path.expanduser().resolve()
    skill_path = mosaic_gmt_getup_skill_path.expanduser().resolve()
    if not source.is_file() or source.suffix != ".npz":
        raise ValueError("MOSAIC get-up probe source must be an NPZ")
    world = replace(
        goalkeeper_world_config(
            difficulty_profile="match",
            environment_count=environment_count,
            second_shot_probability=0.0,
            shot_intent_cue_enabled=True,
        ),
        episode_duration_sec=7.0,
    )
    recovery = GoalkeeperTargetedDiveRLConfig(
        actor_recovery_plasticity_sec=5.0,
        post_save_counterstep_enabled=True,
        post_save_counterstep_duration_sec=1.5,
        post_save_fall_recovery_enabled=True,
        post_save_fall_recovery_duration_sec=5.0,
        runtime_contact_support_side_enabled=True,
        actor_contact_support_side_enabled=True,
        actor_recovery_context_enabled=True,
        mosaic_gmt_model_path=str(mosaic_gmt_model_path.expanduser().resolve()),
        mosaic_gmt_getup_skill_path=str(skill_path),
        mosaic_gmt_getup_blend=blend,
        mosaic_gmt_getup_reference_feedforward_blend=reference_feedforward_blend,
    )
    environment = GoalkeeperTargetedDiveMJWarpBatch(
        asset_root=asset_root.expanduser().resolve(),
        locomotion_policy_path=locomotion_policy_path.expanduser().resolve(),
        targeted_dive_checkpoint=targeted_dive_checkpoint.expanduser().resolve(),
        device=torch.device(device),
        config=world,
        dive_config=recovery,
    )
    environment.reset(seed=seed)
    with np.load(source, allow_pickle=False) as data:
        required = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w"}
        if not required.issubset(data.files):
            raise ValueError("MOSAIC get-up probe source arrays are incomplete")
        raw_position = np.asarray(data["joint_pos"][0], dtype=np.float32)
        raw_velocity = np.asarray(data["joint_vel"][0], dtype=np.float32)
        root_height = float(data["body_pos_w"][0, 0, 2])
        root_quaternion = np.asarray(data["body_quat_w"][0, 0], dtype=np.float32)
    order = np.asarray(MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF, dtype=np.int64)
    canonical_position = torch.as_tensor(raw_position[order], device=device)
    canonical_velocity = torch.as_tensor(raw_velocity[order], device=device)
    environment.qpos[:, 2] = root_height
    environment.qpos[:, 3:7] = torch.as_tensor(root_quaternion, device=device)
    environment.qpos[:, 7:36] = canonical_position
    environment.qvel.zero_()
    environment.qvel[:, 6:35] = canonical_velocity
    environment.task.first_save.fill_(True)
    environment._target.copy_(canonical_position.unsqueeze(0))
    environment._previous_option_lower_body_delta.copy_(
        (canonical_position[:12] - environment._runtime_reach_ready[:12]).unsqueeze(0)
    )
    environment._previous_option_waist_target.copy_(canonical_position[12:15].unsqueeze(0))
    environment._previous_option_arm_target.copy_(canonical_position[15:29].unsqueeze(0))
    environment.mjw.forward(environment.model, environment.data)

    action = torch.zeros((environment_count, environment.action_size), device=device)
    done_ever = torch.zeros(environment_count, dtype=torch.bool, device=device)
    recovered_before_reset = torch.zeros_like(done_ever)
    trace: list[dict[str, Any]] = []
    skill = environment._mosaic_gmt_getup_skill
    contract = environment._mosaic_gmt_contract
    if skill is None or contract is None:
        raise RuntimeError("MOSAIC get-up probe failed to bind its controller artifacts")
    steps = int(round((skill.duration_sec + 0.50) / world.control_dt_sec))
    for step in range(steps):
        _, _, done, _ = environment.step(action)
        quaternion = environment.qpos[:, 3:7]
        upright = 2.0 * (quaternion[:, 0].square() + quaternion[:, 3].square()) - 1.0
        recovered = (environment.qpos[:, 2] >= 0.62) & (upright >= 0.75)
        # ``step`` auto-resets terminal worlds before returning.  Excluding
        # the current ``done`` mask prevents the reset standing pose from
        # being mislabeled as a successful get-up.
        recovered_before_reset |= recovered & ~done_ever & ~done
        done_ever |= done
        if step % 10 == 0 or step == steps - 1:
            trace.append(
                {
                    "time_sec": round((step + 1) * world.control_dt_sec, 3),
                    "mean_pelvis_height_m": float(environment.qpos[:, 2].mean()),
                    "mean_upright_projection": float(upright.mean()),
                    "done_ever_count": int(done_ever.sum()),
                    "recovered_before_reset_count": int(recovered_before_reset.sum()),
                }
            )
    recovered_count = int(recovered_before_reset.sum())
    payload = {
        "schema_version": "rosclaw_soccer.mosaic_getup_physics_probe.v1",
        "source": str(source),
        "source_hash": hash_bytes(source.read_bytes()),
        "skill": str(skill_path),
        "skill_hash": skill.skill_hash,
        "checkpoint_hash": contract.checkpoint_hash,
        "body_hash": skill.body_hash,
        "physics_scene_hash": skill.physics_scene_hash,
        "environment_count": environment_count,
        "recovered_before_reset_count": recovered_count,
        "recovery_rate": recovered_count / environment_count,
        "done_ever_count": int(done_ever.sum()),
        "maximum_applied_blend": float(environment._maximum_applied_mosaic_gmt_getup_blend.max()),
        "reference_feedforward_blend": reference_feedforward_blend,
        "observation_contract": "MOSAIC_BODY_FRAME_BASE_ANGULAR_VELOCITY",
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
    parser.add_argument("--targeted-dive-checkpoint", type=Path, required=True)
    parser.add_argument("--mosaic-gmt-model", type=Path, required=True)
    parser.add_argument("--mosaic-gmt-getup-skill", type=Path, required=True)
    parser.add_argument("--mosaic-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47_101)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--reference-feedforward-blend", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_mosaic_getup_physics_probe(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        targeted_dive_checkpoint=args.targeted_dive_checkpoint,
        mosaic_gmt_model_path=args.mosaic_gmt_model,
        mosaic_gmt_getup_skill_path=args.mosaic_gmt_getup_skill,
        mosaic_source_path=args.mosaic_source,
        output_path=args.output,
        environment_count=args.environment_count,
        seed=args.seed,
        blend=args.blend,
        reference_feedforward_blend=args.reference_feedforward_blend,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
