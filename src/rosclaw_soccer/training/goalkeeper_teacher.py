"""Causal goalkeeper teacher used only to warm-start the neural actor.

This compact teacher expresses reusable task-space intent: shuffle toward the
visible intercept, lead with the anatomically matching arm, close the save
with the support arm, shape both wrists, and return to ready posture when no
shot is active.  It never commands MuJoCo or hardware.  PPO remains responsible
for learning contact, diving reach, and recovery physics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

_UPPER_BODY_RESIDUAL_LIMITS_RAD = np.asarray(
    (
        0.16,
        0.14,
        0.12,
        0.70,
        0.75,
        0.55,
        0.55,
        0.18,
        0.16,
        0.16,
        0.70,
        0.75,
        0.55,
        0.55,
        0.18,
        0.16,
        0.16,
    ),
    dtype=np.float64,
)


def goalkeeper_teacher_action(observation: Any) -> Any:
    """Map the public observation to the bounded 18-D residual contract."""

    import torch

    if observation.ndim != 2 or observation.shape[1] not in {74, 77}:
        raise ValueError("goalkeeper teacher expects a batched 74-D or 77-D observation")
    if not bool(torch.all(torch.isfinite(observation))):
        raise ValueError("goalkeeper teacher observation must be finite")
    target_y = observation[:, 7] * 2.0
    # Observation index 8 is intercept z *relative to the pelvis*, scaled by
    # 0.5.  The qualified locomotion prior keeps the pelvis near 0.77 m.  The
    # previous teacher treated this as world height, silently collapsing its
    # high/low reach branches toward the middle.
    target_z = observation[:, 8] * 2.0 + 0.77
    flight_active = observation[:, -3] + observation[:, -2]
    ready = flight_active < 0.5
    lateral = torch.clamp(-target_y / 0.95, -1.0, 1.0)
    high = torch.clamp((target_z - 0.72) / 0.70, 0.0, 1.0)
    low = torch.clamp((0.72 - target_z) / 0.50, 0.0, 1.0)
    # A central high shot previously produced only the fixed 0.20 reach floor,
    # so even a perfect imitation kept both hands near the torso.  Height now
    # contributes to reach amplitude while the same [-1, 1] residual boundary
    # and downstream angular guard remain in force.
    magnitude = torch.clamp(
        torch.abs(target_y) / 0.95 + 0.22 + 0.44 * high + 0.16 * low,
        0.0,
        1.0,
    )
    action = torch.zeros((observation.shape[0], 18), device=observation.device)
    action[:, 0] = lateral
    # Waist roll and pitch provide a small whole-body reach without taking
    # control away from the frozen locomotion expert.
    action[:, 2] = torch.clamp(-target_y / 1.40, -0.55, 0.55)
    action[:, 3] = torch.clamp(0.25 * low - 0.12 * high, -0.25, 0.35)
    world_negative_y = target_y < 0.0
    left_primary = world_negative_y.to(observation.dtype)
    right_primary = (~world_negative_y).to(observation.dtype)
    # Central/high shots should be closed with two hands.  A far-corner save
    # still leads with the matching arm, while the opposite arm remains active
    # as a counterbalance instead of hanging rigidly at the torso.
    centrality = torch.clamp(1.0 - torch.abs(target_y) / 0.72, 0.0, 1.0)
    support = torch.clamp(0.18 + 0.62 * centrality + 0.20 * high, 0.0, 0.90)
    left = magnitude * torch.where(left_primary > 0.5, torch.ones_like(support), support)
    right = magnitude * torch.where(right_primary > 0.5, torch.ones_like(support), support)
    # For a far-left intercept (negative MuJoCo y while facing -x), negative
    # shoulder-roll/yaw moves the left glove outward; a far-right intercept
    # needs the mirrored positive motion on the right.  The old fixed signs
    # drove both primary hands inward according to the compiled wrist Jacobian.
    left_outward = torch.where(
        world_negative_y,
        -torch.ones_like(lateral),
        torch.ones_like(lateral),
    )
    right_outward = torch.where(
        world_negative_y,
        -torch.ones_like(lateral),
        torch.ones_like(lateral),
    )
    # Residual order: waist(3), left arm(7), right arm(7).
    action[:, 4] = left * (-0.45 - 0.40 * high + 0.20 * low)
    action[:, 5] = left * left_outward * (0.25 + 0.65 * high)
    action[:, 6] = left * left_outward * (0.55 + 0.20 * high)
    # Negative elbow motion extends the G1 palm toward incoming -x and lifts
    # it in +z.  The previous positive sign flexed the elbow backward/down,
    # which made the policy look active while moving the glove away from the
    # intercept (confirmed against the compiled G1 wrist Jacobian).
    action[:, 7] = left * (-0.42 - 0.18 * high + 0.12 * low)
    action[:, 8] = left * (0.12 + 0.18 * high)
    action[:, 9] = left * (-0.10 + 0.18 * low)
    action[:, 10] = left * 0.12 * (1.0 - low)
    action[:, 11] = right * (-0.45 - 0.40 * high + 0.20 * low)
    action[:, 12] = right * right_outward * (0.25 + 0.65 * high)
    action[:, 13] = right * right_outward * (0.55 + 0.20 * high)
    action[:, 14] = right * (-0.42 - 0.18 * high + 0.12 * low)
    action[:, 15] = right * (-0.12 - 0.18 * high)
    action[:, 16] = right * (-0.10 + 0.18 * low)
    action[:, 17] = right * -0.12 * (1.0 - low)
    # Keep a compact goalkeeper-ready pocket between shots instead of letting
    # both arms hang straight down.  Negative shoulder pitch/elbow lifts the
    # palms toward incoming -x/+z; mirrored roll opens the elbows.  This gives
    # the second save a short, symmetric start pose while the frozen lower-body
    # prior remains fully authoritative.
    action[ready] = 0.0
    action[ready, 4] = -0.22
    action[ready, 5] = -0.16
    action[ready, 7] = -0.18
    action[ready, 11] = -0.22
    action[ready, 12] = 0.16
    action[ready, 14] = -0.18
    return torch.clamp(action, -1.0, 1.0)


def pretrain_goalkeeper_actor(
    model: Any,
    *,
    observation_size: int,
    device: Any,
    samples: int = 32_768,
    epochs: int = 20,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    parent_replay_coefficient: float = 0.0,
    motion_library_path: Path | None = None,
    motion_dataset_root: Path | None = None,
    expected_body_hash: str | None = None,
    motion_prior_blend: float = 0.08,
    reach_model: Any | None = None,
    task_space_reach_blend: float = 0.0,
    arm_only_update: bool = False,
    seed: int = 4711,
) -> dict[str, Any]:
    """Distill the causal teacher into the actor before online physics PPO."""

    import torch

    if observation_size not in {74, 77} or samples < batch_size or samples % batch_size:
        raise ValueError("goalkeeper teacher pretraining dimensions are invalid")
    if (
        not 1 <= epochs <= 200
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not math.isfinite(parent_replay_coefficient)
        or not 0.0 <= parent_replay_coefficient <= 2.0
        or not math.isfinite(motion_prior_blend)
        or not 0.0 <= motion_prior_blend <= 0.20
        or not math.isfinite(task_space_reach_blend)
        or not 0.0 <= task_space_reach_blend <= 0.85
    ):
        raise ValueError("goalkeeper teacher pretraining settings are invalid")
    if not isinstance(arm_only_update, bool):
        raise ValueError("goalkeeper teacher arm-only update flag must be boolean")
    if (reach_model is None) != (task_space_reach_blend == 0.0):
        raise ValueError("goalkeeper task-space reach requires model and positive blend together")
    motion_inputs = (motion_library_path, motion_dataset_root, expected_body_hash)
    if any(value is not None for value in motion_inputs) and not all(
        value is not None for value in motion_inputs
    ):
        raise ValueError("goalkeeper motion prior requires library, dataset, and Body hash")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    observation = torch.zeros((samples, observation_size), device=device)
    # Match the live observation normalization: relative x is small, target
    # y/z occupy indices 7/8, and the two flight bits occupy 71/72.
    observation[:, 0] = torch.empty(samples, device=device).uniform_(
        -1.2, -0.4, generator=generator
    )
    # The causal goal-line intercept occupies 6:9 and is scaled by 0.5.
    # Omitting x made the historical reach teacher learn toward the pelvis
    # plane instead of the physical glove/ball contact plane 0.08 m ahead.
    observation[:, 6] = -0.04
    observation[:, 7] = torch.empty(samples, device=device).uniform_(
        -0.60, 0.60, generator=generator
    )
    # Live channel 8 is ``(target_z - pelvis_z) * 0.5``.  For the qualified
    # 0.793 m ready pelvis and declared 0.10--1.65 m elite curriculum this is
    # roughly [-0.35, 0.43].  This also covers the narrower standard course.
    observation[:, 8] = torch.empty(samples, device=device).uniform_(
        -0.35, 0.43, generator=generator
    )
    high_shot = torch.rand(samples, device=device, generator=generator) > 0.5
    ready_sample = torch.rand(samples, device=device, generator=generator) < 0.25
    observation[:, -3] = (high_shot & ~ready_sample).to(torch.float32)
    observation[:, -2] = (~high_shot & ~ready_sample).to(torch.float32)
    with torch.no_grad():
        parent_target = torch.tanh(model(observation)[0]).detach()
    target = goalkeeper_teacher_action(observation)
    motion_report: dict[str, Any] | None = None
    if motion_library_path is not None:
        if motion_dataset_root is None or expected_body_hash is None:
            raise AssertionError("validated goalkeeper motion prior inputs disappeared")
        motion_table, motion_report = build_motiondecode_upper_body_teacher(
            motion_library_path=motion_library_path,
            dataset_root=motion_dataset_root,
            expected_body_hash=expected_body_hash,
        )
        motion_target = _motiondecode_teacher_action(
            observation,
            motion_table=motion_table,
        )
        # Preserve the task-space teacher's lateral command.  MotionDecode is
        # an unlabeled proxy and may only add a small, bounded waist/arm style
        # residual; it can never replace the causal reach target or the frozen
        # lower-body locomotion expert.
        target[:, 1:] = (1.0 - motion_prior_blend) * target[:, 1:]
        target[:, 1:] += motion_prior_blend * motion_target[:, 1:]
    reach_report: dict[str, Any] | None = None
    if reach_model is not None:
        from rosclaw_soccer.training.goalkeeper_reach import (
            reach_model_payload,
            task_space_reach_teacher_action,
        )

        reach_target = task_space_reach_teacher_action(observation, model=reach_model)
        # The task-space layer is an arm-only teacher.  It cannot change the
        # actor's lateral command or waist counter-rotation and never executes
        # in the physics loop after pretraining.
        target[:, 4:] = (1.0 - task_space_reach_blend) * target[:, 4:]
        target[:, 4:] += task_space_reach_blend * reach_target[:, 4:]
        reach_report = reach_model_payload(reach_model)
    if arm_only_update:
        # A specialized arm lesson must not silently rewrite the learned
        # representation, lateral command, or waist stabilizer.  Actor rows
        # 4:18 remain plastic and consume the frozen parent's latent features.
        target[:, :4] = parent_target[:, :4]
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if arm_only_update and (name.startswith("trunk.") or name in {"actor.weight", "actor.bias"})
    }
    optimizer = torch.optim.Adam(
        [*model.trunk.parameters(), *model.actor.parameters()], lr=learning_rate
    )
    initial_loss = 0.0
    final_loss = 0.0
    initial_parent_replay_loss = float(
        torch.mean(torch.square(torch.tanh(model(observation)[0]) - parent_target)).detach()
    )
    final_parent_replay_loss = 0.0
    for epoch in range(epochs):
        permutation = torch.randperm(samples, device=device, generator=generator)
        losses: list[float] = []
        for start in range(0, samples, batch_size):
            indices = permutation[start : start + batch_size]
            mean, _, _ = model(observation[indices])
            prediction = torch.tanh(mean)
            teacher_loss = torch.mean(torch.square(prediction - target[indices]))
            parent_loss = torch.mean(torch.square(prediction - parent_target[indices]))
            loss = teacher_loss + parent_replay_coefficient * parent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if arm_only_update:
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    if name.startswith("trunk."):
                        parameter.grad.zero_()
                    elif name in {"actor.weight", "actor.bias"}:
                        parameter.grad[:4].zero_()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = sum(losses) / len(losses)
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss
    with torch.no_grad():
        final_parent_replay_loss = float(
            torch.mean(torch.square(torch.tanh(model(observation)[0]) - parent_target))
        )
        frozen_differences: list[float] = []
        for name, parameter in model.named_parameters():
            if name not in frozen_before:
                continue
            before = frozen_before[name]
            if name in {"actor.weight", "actor.bias"}:
                frozen_differences.append(float(torch.max(torch.abs(parameter[:4] - before[:4]))))
            else:
                frozen_differences.append(float(torch.max(torch.abs(parameter - before))))
        maximum_frozen_parameter_difference = max(frozen_differences, default=0.0)
    return {
        "schema_version": "rosclaw_soccer.goalkeeper_teacher_pretraining.v3",
        "samples": samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_mean_squared_error": initial_loss,
        "final_mean_squared_error": final_loss,
        "parent_replay_coefficient": parent_replay_coefficient,
        "ready_posture_sample_fraction": float(ready_sample.to(torch.float32).mean()),
        "initial_parent_replay_mean_squared_error": initial_parent_replay_loss,
        "final_parent_replay_mean_squared_error": final_parent_replay_loss,
        "improved": final_loss < initial_loss,
        "teacher_authority": "PRETRAINING_TARGET_ONLY",
        "motion_prior": motion_report,
        "motion_prior_blend": motion_prior_blend if motion_report is not None else 0.0,
        "task_space_reach": reach_report,
        "task_space_reach_blend": (task_space_reach_blend if reach_report is not None else 0.0),
        "arm_only_update": arm_only_update,
        "maximum_frozen_parameter_difference": maximum_frozen_parameter_difference,
        "physics_authority": False,
        "activation_ceiling": "SIM_ONLY",
    }


def build_motiondecode_upper_body_teacher(
    *,
    motion_library_path: Path,
    dataset_root: Path,
    expected_body_hash: str,
    residual_scale: float = 0.70,
) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    """Distil train-only waist/arm proxy poses with content-bound provenance."""

    from rosclaw_soccer.skills.goalkeeper_v2.motion_library import (
        GoalkeeperMotionFamily,
        load_goalkeeper_motion_library,
        load_motion_clip_frames,
    )

    if not math.isfinite(residual_scale) or not 0.25 <= residual_scale <= 1.0:
        raise ValueError("goalkeeper motion prior residual scale is outside [0.25, 1]")
    library = load_goalkeeper_motion_library(
        motion_library_path,
        dataset_root=dataset_root,
    )
    if library.body_hash != expected_body_hash:
        raise ValueError("goalkeeper motion prior Body hash mismatch")
    if not library.contains_only_proxy_motion or library.human_goalkeeper_claim_allowed:
        raise ValueError("MotionDecode teacher must preserve its proxy-only claim boundary")
    limits = _UPPER_BODY_RESIDUAL_LIMITS_RAD * residual_scale
    table: dict[str, tuple[float, ...]] = {}
    source_hashes: dict[str, str] = {}
    for family in GoalkeeperMotionFamily:
        clip = library.clips_for(
            task=(
                "ready"
                if family is GoalkeeperMotionFamily.READY
                else "recovery"
                if family is GoalkeeperMotionFamily.RECOVERY
                else "shuffle"
                if family
                in {
                    GoalkeeperMotionFamily.SPLIT_STEP,
                    GoalkeeperMotionFamily.SHUFFLE_LEFT,
                    GoalkeeperMotionFamily.SHUFFLE_RIGHT,
                }
                else "save"
            ),
            region=(
                "upper_left"
                if family
                in {
                    GoalkeeperMotionFamily.SHUFFLE_LEFT,
                    GoalkeeperMotionFamily.HIGH_REACH_LEFT,
                }
                else "upper_right"
                if family
                in {
                    GoalkeeperMotionFamily.SHUFFLE_RIGHT,
                    GoalkeeperMotionFamily.HIGH_REACH_RIGHT,
                }
                else "lower_left"
                if family is GoalkeeperMotionFamily.LOW_SAVE_LEFT
                else "lower_right"
                if family is GoalkeeperMotionFamily.LOW_SAVE_RIGHT
                else "center"
            ),
        )[0]
        q, _ = load_motion_clip_frames(dataset_root=dataset_root, clip=clip)
        upper = q[:, 12:] - q[:1, 12:]
        # Choose one coherent whole-body instant instead of independently
        # mixing per-joint extrema.  This retains human joint correlations.
        arm_energy = np.linalg.norm(upper[:, 3:], axis=1)
        frame = upper[int(np.argmax(arm_energy))]
        normalized = np.clip(frame / limits, -1.0, 1.0)
        table[family.value] = tuple(float(value) for value in normalized)
        source_hashes[family.value] = clip.source_hash
    return table, {
        "schema_version": "rosclaw_soccer.motiondecode_upper_body_teacher.v1",
        "library_hash": library.library_hash,
        "body_hash": library.body_hash,
        "source_hashes": source_hashes,
        "joint_authority": "WAIST_AND_ARMS_STYLE_TARGET_ONLY",
        "lower_body_authority": "NONE",
        "contains_only_proxy_motion": True,
        "human_goalkeeper_claim_allowed": False,
        "commercial_use_allowed": False,
        "activation_ceiling": "SIM_ONLY",
    }


def _motiondecode_teacher_action(
    observation: Any,
    *,
    motion_table: dict[str, tuple[float, ...]],
) -> Any:
    """Select a proxy pose causally from the same public shot observation."""

    import torch

    target_y = observation[:, 7] * 2.0
    target_z = observation[:, 8] * 2.0 + 0.77
    active = observation[:, -3] + observation[:, -2] >= 0.5
    high = target_z >= 0.86
    far = torch.abs(target_y) >= 0.22
    # The G1 faces -x in this world, so negative world-y is its anatomical
    # left.  This matches the causal teacher and compiled wrist Jacobian.
    left = target_y < 0.0
    family = ["ready"] * observation.shape[0]
    for index in range(observation.shape[0]):
        if not bool(active[index]):
            family[index] = "ready"
        elif not bool(far[index]):
            family[index] = "center_block" if bool(high[index]) else "split_step"
        elif bool(left[index]):
            family[index] = "high_reach_left" if bool(high[index]) else "low_save_left"
        else:
            family[index] = "high_reach_right" if bool(high[index]) else "low_save_right"
    result = torch.zeros((observation.shape[0], 18), device=observation.device)
    for name in set(family):
        indices = torch.tensor(
            [item == name for item in family],
            dtype=torch.bool,
            device=observation.device,
        )
        result[indices, 1:] = torch.as_tensor(
            motion_table[name],
            dtype=observation.dtype,
            device=observation.device,
        )
    return result


__all__ = [
    "build_motiondecode_upper_body_teacher",
    "goalkeeper_teacher_action",
    "pretrain_goalkeeper_actor",
]
