"""Versioned offline new-head experiment; never authorize physical deployment."""

import random
from typing import Any

import torch

from rosclaw_soccer.sim.contracts import hash_json

KIND = "consolidated-refinement-v1"
SCHEMA = "soccer.whole_acquired_motor_refinement.v1"
STATUS = "WHOLE_MOTOR_REFINEMENT_HEAD_TRAINED_PENDING_CPU_AND_PHYSICS"
FEATURE_STATUS = "PASS_REFINEMENT_OWN_FEATURES_ZERO_IDENTITY_NOT_PHYSICS"
COMMITMENTS = {
    "feature_hash",
    "bank_hash",
    "parent_training_hash",
    "parent_checkpoint_hash",
    "parent_lineage_hash",
    "schedule_hash",
}


def refinement_config(seed: int, device: str) -> dict[str, Any]:
    if (
        type(seed) is not int
        or seed not in (115023, 115024, 115025)
        or device not in ("cpu", "cuda:0")
    ):
        raise ValueError("one bounded three-seed refinement family and explicit device required")
    return dict(
        schema=SCHEMA,
        seed=seed,
        device=device,
        new_head_steps=2048,
        lr=0.0003,
        initial_new_adam_steps=0,
        optimizer_inherited=False,
        acquired_head_steps_unchanged=2048,
        memory_actor_steps_unchanged=112,
        memory_critic_steps_unchanged=48,
        cumulative_steps_across_two_heads=4096,
        parent_whole_frozen=True,
        original_total_bound_rad=0.2,
        head_shape=[288, 128, 29],
        head_parameters=4,
        old_per_step=4,
        current_per_step=4,
        teachers_per_step=2,
        old_weight=4.0,
        current_weight=1.0,
        teacher_weight=1.0,
        loss_unit_rad=0.02,
        grad_norm_limit=1.0,
        whole_histories=True,
        phase="offline_supervised_NOT_PPO",
        PPO_steps=0,
        actor_case_lookup=False,
        Fresh_fit_rows=0,
        DEV_fit_rows=0,
    )


def refinement_schedule(records: list[dict[str, Any]], seed: int) -> list[list[int]]:
    refinement_config(seed, "cpu")
    pools = [
        [r["index"] for r in records if r["role"] == role]
        for role in ("retained_original17", "retained_current41", "qualified_rescue")
    ]
    if (
        len(pools[0]) != 17
        or len(pools[1]) != 41
        or len(pools[2]) < 2
        or [r["index"] for r in records] != list(range(len(records)))
        or len(set(sum(pools, []))) != len(records)
    ):
        raise ValueError("all disjoint qualified17/41/rescue whole-history pools required")
    rng = random.Random(seed)
    for pool in pools:
        rng.shuffle(pool)
    return [
        [pools[0][(4 * i + j) % len(pools[0])] for j in range(4)]
        + [pools[1][(4 * i + j) % len(pools[1])] for j in range(4)]
        + [pools[2][(2 * i + j) % len(pools[2])] for j in range(2)]
        for i in range(2048)
    ]


def schedule_hash(records: list[dict[str, Any]], seed: int) -> str:
    return str(hash_json(refinement_schedule(records, seed)))


def validate_refinement_checkpoint(
    checkpoint: Any,
    expected: dict[str, torch.Tensor],
    *,
    config: dict[str, Any],
    commitments: dict[str, str],
) -> None:
    if (
        set(commitments) != COMMITMENTS
        or type(checkpoint) is not dict
        or set(checkpoint)
        != {"schema", "config", "head", "initial_head", "head_optimizer", "new_head_steps"}
        | COMMITMENTS
        or checkpoint["schema"] != SCHEMA
        or checkpoint["config"] != config
        or config != refinement_config(config["seed"], config["device"])
        or checkpoint["new_head_steps"] != 2048
        or any(checkpoint[k] != v for k, v in commitments.items())
    ):
        raise ValueError("exact versioned head-only training and ancestry required")
    for field in ("head", "initial_head"):
        values = checkpoint[field]
        if type(values) is not dict or set(values) != set(expected):
            raise ValueError("all and only four exact head tensors required")
        for key, v in values.items():
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != expected[key].shape
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite shape-bound CPU head tensors required")
    if (
        checkpoint["initial_head"]["2.weight"].any()
        or checkpoint["initial_head"]["2.bias"].any()
        or any(torch.equal(checkpoint["head"][k], checkpoint["initial_head"][k]) for k in expected)
    ):
        raise ValueError(
            "zero initial output and actual changes to all four new head tensors required"
        )
    opt = checkpoint["head_optimizer"]
    if (
        type(opt) is not dict
        or set(opt) != {"state", "param_groups"}
        or len(opt["param_groups"]) != 1
    ):
        raise ValueError("one new-head-only Adam optimizer required")
    group = opt["param_groups"][0]
    if (
        group["params"] != list(range(4))
        or group["lr"] != config["lr"]
        or group["weight_decay"] != 0
        or group["amsgrad"] is not False
        or group["maximize"] is not False
        or set(opt["state"]) != set(range(4))
    ):
        raise ValueError("exact four-parameter unregularized Adam group required")
    for i, key in enumerate(expected):
        value = opt["state"][i]
        if type(value) is not dict or set(value) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("complete new Adam state required")
        step = value["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.ndim != 0
            or step.item() != 2048
        ):
            raise ValueError("actual2048 new Adam steps required, not inherited4096")
        for field in ("exp_avg", "exp_avg_sq"):
            v = value[field]
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != expected[key].shape
                or not torch.isfinite(v).all()
                or (field == "exp_avg_sq" and (v < 0).any())
            ):
                raise ValueError("finite exact-shape new optimizer moments required")
