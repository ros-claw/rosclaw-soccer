"""Bounded offline neural-head experiment over frozen WHOLE learned memory.

Head parameters are new. Their optimizer starts at zero, while old112/48
state stays immutable. This is not PPO, a physical qualification or promotion.
"""

import random
from typing import Any

import torch

from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
from rosclaw_soccer.sim.contracts import hash_json

KIND = "consolidated-adapter-v1"
STATUS = "CONSOLIDATED_ADAPTER_HEAD_TRAINED_PENDING_CPU_OWN_AND_PHYSICAL_EXAM"
FEATURE_STATUS = "CONSOLIDATED_LEARNED_MEMORY_OWN274_ZERO_HEAD_IDENTITY_NOT_PHYSICS"
SCHEMA = "soccer.consolidated_learned_motor_adapter.v1"


def adapter_config(seed: int, device: str) -> dict[str, Any]:
    if type(seed) is not int or seed not in (105023, 105024, 105025):
        raise ValueError("three explicit independent neural-head seeds required")
    if device not in ("cpu", "cuda:0"):
        raise ValueError("explicit bounded training device required")
    return dict(
        protocol=SCHEMA,
        seed=seed,
        device=device,
        head_steps=2048,
        head_lr=0.0003,
        head_bound_rad=0.2,
        original_total_bound_rad=0.2,
        head_shape=[288, 128, 29],
        head_parameters=4,
        old_memory_actor_steps=112,
        old_memory_critic_steps=48,
        initial_head_optimizer_steps=0,
        head_optimizer_inherited=False,
        old_memory_frozen=True,
        motor_loss_unit_rad=0.02,
        old_weight=4.0,
        baseline_pass_weight=1.0,
        teachers_per_step=2,
        old_anchors_per_step=4,
        baseline_passes_per_step=1,
        grad_norm_limit=1.0,
        learning_phase="offline_supervised_new_head_NOT_PPO",
        own_zero_state_features=True,
        actor_case_lookup=False,
        final_only=True,
        PPO_steps_this_stage=0,
        DEV_fit_rows=0,
        Fresh_fit_rows=0,
    )


def adapter_fit_plan(records: list[dict[str, Any]]) -> dict[str, Any]:
    original = single_mode_pass_curriculum(records)
    anchors = [r["index"] for r in records if r["role"] == "retained_neural17"]
    if len(anchors) != 17 or any(not records[i]["task_success"] for i in anchors):
        raise ValueError("all17 ORIGINAL slow-memory anchors required, not later warm17")
    plan = dict(
        schema="soccer.consolidated_adapter_fit_plan.v1",
        source_records_hash=hash_json(records),
        coherent_pass_selection_hash=original["selection_hash"],
        old_anchor_indices=anchors,
        teacher_indices=original["selected_teacher_indices"],
        baseline_pass_indices=original["selected_autonomous_indices"],
        anchor_target="exact_zero_added_delta_to_entire_learned_memory",
        teacher_target="actual_qualified_action_minus_own_frozen_memory",
        task_labels_modified=False,
        records_removed=0,
        promotion_authorized=False,
    )
    plan["hash"] = hash_json(plan)
    return plan


def adapter_schedule(plan: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    cfg = adapter_config(seed, "cpu")
    rng = random.Random(seed)
    pools = [
        list(plan[k]) for k in ("teacher_indices", "old_anchor_indices", "baseline_pass_indices")
    ]
    if any(not pool or len(pool) != len(set(pool)) for pool in pools):
        raise ValueError("nonempty unique actual-history pools required")
    if len(set(sum(pools, []))) != sum(map(len, pools)):
        raise ValueError("disjoint teacher and protected slow-memory histories required")
    for pool in pools:
        rng.shuffle(pool)
    teachers, old, baseline = pools
    return [
        dict(
            step=i + 1,
            teacher_indices=[teachers[(2 * i + j) % len(teachers)] for j in range(2)],
            old_anchor_indices=[old[(4 * i + j) % len(old)] for j in range(4)],
            baseline_pass_index=baseline[i % len(baseline)],
        )
        for i in range(cfg["head_steps"])
    ]


def validate_head_checkpoint(
    checkpoint: Any,
    expected: dict[str, torch.Tensor],
    *,
    config: dict[str, Any],
    commitments: dict[str, str],
) -> None:
    """Safe tensor-only shape/identity/Adam admission, not gradient reexecution."""
    keys = {
        "schema",
        "config",
        "head",
        "initial_head",
        "head_optimizer",
        "head_optimizer_steps",
        "feature_hash",
        "data_hash",
        "fit_plan_hash",
        "memory_training_hash",
        "memory_checkpoint_hash",
        "parent_lineage_hash",
    }
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != keys
        or checkpoint["schema"] != SCHEMA
        or checkpoint["config"] != config
        or config != adapter_config(config["seed"], config["device"])
        or checkpoint["head_optimizer_steps"] != config["head_steps"]
        or any(checkpoint[k] != v for k, v in commitments.items())
    ):
        raise ValueError("exact separately versioned new-head checkpoint and ancestry required")
    for field in ("head", "initial_head"):
        state = checkpoint[field]
        if type(state) is not dict or set(state) != set(expected):
            raise ValueError("all and only new-head tensors required")
        for k, v in state.items():
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != expected[k].shape
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite exact-shape CPU head tensor required")
    if (
        checkpoint["initial_head"]["2.weight"].any()
        or checkpoint["initial_head"]["2.bias"].any()
        or any(torch.equal(checkpoint["head"][k], checkpoint["initial_head"][k]) for k in expected)
    ):
        raise ValueError("zero initial output and actual update of ALL4 new-head tensors required")
    opt = checkpoint["head_optimizer"]
    if type(opt) is not dict or set(opt) != {"param_groups", "state"}:
        raise ValueError("explicit head-only Adam required")
    groups = opt["param_groups"]
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or groups[0]["params"] != [0, 1, 2, 3]
        or groups[0]["lr"] != config["head_lr"]
        or groups[0]["betas"] != (0.9, 0.999)
        or groups[0]["weight_decay"] != 0
        or set(opt["state"]) != {0, 1, 2, 3}
    ):
        raise ValueError("all and only4 new-parameter Adam slots required")
    for i, (_, shape) in enumerate(expected.items()):
        slot = opt["state"][i]
        if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("exact new-head Adam slot required")
        step = slot["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != ()
            or step.device.type != "cpu"
            or step.dtype != torch.float32
            or float(step) != config["head_steps"]
        ):
            raise ValueError("truthful separate new-head update count required")
        for name in ("exp_avg", "exp_avg_sq"):
            v = slot[name]
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != shape.shape
                or not torch.isfinite(v).all()
                or (name == "exp_avg_sq" and (v < 0).any())
            ):
                raise ValueError("finite own-parameter head optimizer moments required")
