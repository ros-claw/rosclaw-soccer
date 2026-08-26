"""Stability-plasticity consolidation for a physics-trained goalkeeper.

The fast actor may discover useful reaching while also becoming too violent.
This module creates bounded interpolation candidates between the immutable
locomotion baseline and a learned actor.  It never chooses a candidate from
training reward: every scale remains pending the same GPU holdout and strict
CPU MuJoCo exam as any newly trained policy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def consolidate_goalkeeper_checkpoint(
    *,
    source_checkpoint: Path,
    output_checkpoint: Path,
    actor_scale: float,
) -> dict[str, Any]:
    """Scale only actor logits and exploration variance, preserving the critic."""

    if not math.isfinite(actor_scale) or not 0.0 < actor_scale <= 1.0:
        raise ValueError("goalkeeper consolidation actor scale must be in (0, 1]")
    source = source_checkpoint.expanduser().resolve()
    output = output_checkpoint.expanduser().resolve()
    if source == output or output.exists():
        raise ValueError("goalkeeper consolidation requires a new output checkpoint")
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not {"actor.weight", "actor.bias", "log_std"} <= set(state):
        raise ValueError("goalkeeper consolidation source checkpoint is invalid")
    consolidated = dict(payload)
    consolidated_state = {
        name: value.detach().clone() if hasattr(value, "detach") else value
        for name, value in state.items()
    }
    consolidated_state["actor.weight"] *= actor_scale
    consolidated_state["actor.bias"] *= actor_scale
    # Evaluation uses the deterministic mean, but preserving the same scale
    # for log standard deviation makes any later continuation honest.
    consolidated_state["log_std"] += math.log(actor_scale)
    consolidated["state_dict"] = consolidated_state
    consolidated["consolidation"] = {
        "schema_version": "rosclaw_soccer.goalkeeper_consolidation.v1",
        "source_checkpoint": source.name,
        "actor_scale": actor_scale,
        "selection_authority": False,
        "evaluation_required": "GPU_HOLDOUT_THEN_CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
    }
    consolidated["promotion_status"] = "CANDIDATE_PENDING_EVALUATION"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(consolidated, output)
    return dict(consolidated["consolidation"])


def consolidate_goalkeeper_hierarchical_checkpoint(
    *,
    parent_checkpoint: Path,
    candidate_checkpoint: Path,
    output_checkpoint: Path,
    trunk_plasticity: float,
    core_action_plasticity: float,
    arm_action_plasticity: float,
) -> dict[str, Any]:
    """Merge reusable representation, stable core, and plastic arms separately."""

    scales = (trunk_plasticity, core_action_plasticity, arm_action_plasticity)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scales):
        raise ValueError("goalkeeper hierarchical consolidation scales must be in [0, 1]")
    parent = parent_checkpoint.expanduser().resolve()
    candidate = candidate_checkpoint.expanduser().resolve()
    output = output_checkpoint.expanduser().resolve()
    if len({parent, candidate, output}) != 3 or output.exists():
        raise ValueError("goalkeeper hierarchical consolidation requires three distinct paths")
    import torch

    parent_payload = torch.load(parent, map_location="cpu", weights_only=True)
    candidate_payload = torch.load(candidate, map_location="cpu", weights_only=True)
    parent_state = parent_payload.get("state_dict")
    candidate_state = candidate_payload.get("state_dict")
    if (
        not isinstance(parent_state, dict)
        or not isinstance(candidate_state, dict)
        or set(parent_state) != set(candidate_state)
        or not {"actor.weight", "actor.bias", "log_std"} <= set(parent_state)
    ):
        raise ValueError("goalkeeper hierarchical consolidation checkpoints are incompatible")

    def interpolate(name: str, scale: float) -> Any:
        parent_value = parent_state[name]
        candidate_value = candidate_state[name]
        if not hasattr(parent_value, "shape") or parent_value.shape != candidate_value.shape:
            raise ValueError("goalkeeper hierarchical consolidation tensor mismatch")
        return parent_value.detach().clone() + scale * (
            candidate_value.detach() - parent_value.detach()
        )

    merged_state: dict[str, Any] = {}
    for name in parent_state:
        if name.startswith("trunk."):
            merged_state[name] = interpolate(name, trunk_plasticity)
        elif name in {"actor.weight", "actor.bias", "log_std"}:
            merged_state[name] = parent_state[name].detach().clone()
        else:
            # The critic has no inference authority. Keep the candidate value
            # so later continuation retains its value-function warm start.
            merged_state[name] = candidate_state[name].detach().clone()
    for name in ("actor.weight", "actor.bias", "log_std"):
        merged_state[name][:4] = interpolate(name, core_action_plasticity)[:4]
        merged_state[name][4:] = interpolate(name, arm_action_plasticity)[4:]
    merged = dict(candidate_payload)
    merged["state_dict"] = merged_state
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_hierarchical_consolidation.v1",
        "parent_checkpoint_hash": hash_bytes(parent.read_bytes()),
        "candidate_checkpoint_hash": hash_bytes(candidate.read_bytes()),
        "trunk_plasticity": trunk_plasticity,
        "core_action_plasticity": core_action_plasticity,
        "arm_action_plasticity": arm_action_plasticity,
        "stable_action_channels": "LATERAL_AND_WAIST_0_TO_3",
        "plastic_action_channels": "LEFT_AND_RIGHT_ARMS_4_TO_17",
        "selection_authority": False,
        "evaluation_required": "PAIRED_PARENT_GPU_HOLDOUT_THEN_CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    merged["consolidation"] = report
    merged["promotion_status"] = "CANDIDATE_PENDING_EVALUATION"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    return report


def consolidate_goalkeeper_action_channels(
    *,
    source_checkpoint: Path,
    output_checkpoint: Path,
    core_action_scale: float,
    arm_action_scale: float,
) -> dict[str, Any]:
    """Attenuate lateral/waist and arm logits independently after learning."""

    scales = (core_action_scale, arm_action_scale)
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in scales):
        raise ValueError("goalkeeper action-channel scales must be in (0, 1]")
    source = source_checkpoint.expanduser().resolve()
    output = output_checkpoint.expanduser().resolve()
    if source == output or output.exists():
        raise ValueError("goalkeeper action-channel consolidation requires a new output")
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not {"actor.weight", "actor.bias", "log_std"} <= set(state):
        raise ValueError("goalkeeper action-channel source checkpoint is invalid")
    merged = dict(payload)
    merged_state = {
        name: value.detach().clone() if hasattr(value, "detach") else value
        for name, value in state.items()
    }
    for name in ("actor.weight", "actor.bias"):
        merged_state[name][:4] *= core_action_scale
        merged_state[name][4:] *= arm_action_scale
    merged_state["log_std"][:4] += math.log(core_action_scale)
    merged_state["log_std"][4:] += math.log(arm_action_scale)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_action_channel_consolidation.v1",
        "source_checkpoint_hash": hash_bytes(source.read_bytes()),
        "core_action_scale": core_action_scale,
        "arm_action_scale": arm_action_scale,
        "stable_action_channels": "LATERAL_AND_WAIST_0_TO_3",
        "plastic_action_channels": "LEFT_AND_RIGHT_ARMS_4_TO_17",
        "selection_authority": False,
        "evaluation_required": "PAIRED_PARENT_GPU_HOLDOUT_THEN_CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    merged["state_dict"] = merged_state
    merged["consolidation"] = report
    merged["promotion_status"] = "CANDIDATE_PENDING_EVALUATION"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    return report


__all__ = [
    "consolidate_goalkeeper_checkpoint",
    "consolidate_goalkeeper_hierarchical_checkpoint",
    "consolidate_goalkeeper_action_channels",
]
