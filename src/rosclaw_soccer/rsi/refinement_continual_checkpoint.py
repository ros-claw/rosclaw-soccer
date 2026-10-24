"""One bounded generation1 actor/critic/Adam continuation; research only."""

import copy
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.refinement_continual_batch import continual_config
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
from rosclaw_soccer.rsi.refinement_onpolicy_learning import tensor_state_hash, validate_tensor_map

KIND = "refinement-actor-critic-continual-gen1-v3"
SCHEMA = "soccer.refinement.continual_gen1_checkpoint.v3"
STATUS = "REFINEMENT_CONTINUAL_GEN1_UPDATED_PENDING_PHYSICAL_VALIDATION"


def same_nested(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_nested(a[k], b[k]) for k in a)
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return bool(a.dtype == b.dtype and a.device == b.device and torch.equal(a, b))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_nested(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def rebased_adam(prior: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(prior)
    if len(result["param_groups"]) != 2:
        raise ValueError("two exact actor/critic optimizer groups required")
    for group, rate in zip(
        result["param_groups"], (config["actor_lr"], config["critic_lr"]), strict=True
    ):
        group["lr"] = rate
    return result


def validate_continual_checkpoint(
    state: dict[str, Any],
    behavior: RefinementMotorAdapter,
    prior_critic: MotorCritic,
    prior_optimizer: dict[str, Any],
    *,
    config: dict[str, Any],
    commitments: dict[str, str],
    steps: int,
) -> None:
    if (
        config != continual_config(config["state_id"])
        or state.get("schema") != SCHEMA
        or state.get("kind") != KIND
        or state.get("config") != config
        or state.get("commitments") != commitments
        or set(commitments) != {"prior_training_hash", "prepared_hash", "frozen_parent_hash"}
        or type(steps) is not int
        or not 1 <= steps <= 4
        or state.get("new_actor_steps") != steps
        or state.get("new_critic_steps") != steps
        or state.get("actor_steps") != 4 + steps
        or state.get("critic_steps") != 4 + steps
        or tensor_state_hash(dict(behavior.parent.state_dict()))
        != commitments["frozen_parent_hash"]
    ):
        raise ValueError("exact genuine generation1 metadata and frozen parent required")
    head, value = dict(behavior.head.state_dict()), dict(prior_critic.state_dict())
    for k, expected in (
        ("initial_head", head),
        ("head", head),
        ("initial_critic", value),
        ("critic", value),
    ):
        validate_tensor_map(state.get(k), expected)
    if not same_nested(state["initial_head"], head) or not same_nested(
        state["initial_critic"], value
    ):
        raise ValueError("generation1 must start from ACTUAL learned actor and trained critic")
    if same_nested(state["head"], head) or same_nested(state["critic"], value):
        raise ValueError("unchanged weights are not an actor/critic continuation")
    expected_initial = rebased_adam(prior_optimizer, config)
    if set(prior_optimizer["state"]) != set(range(8)) or any(
        float(s["step"]) != 4 for s in prior_optimizer["state"].values()
    ):
        raise ValueError("actual prior generation four-step Adam state required")
    if not same_nested(state.get("initial_optimizer"), expected_initial):
        raise ValueError(
            "all prior Adam moments/counters must carry; only declared LR rebase allowed"
        )
    optimizer = state.get("optimizer")
    if type(optimizer) is not dict or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("complete continued Adam state required")
    params = [*head.values(), *value.values()]
    if set(optimizer["state"]) != set(range(8)) or len(optimizer["param_groups"]) != 2:
        raise ValueError("all8 continued actor/critic parameter states required")
    for i, p in enumerate(params):
        s = optimizer["state"][i]
        if (
            set(s) != {"step", "exp_avg", "exp_avg_sq"}
            or not isinstance(s["step"], torch.Tensor)
            or s["step"].numel() != 1
            or float(s["step"]) != 4 + steps
        ):
            raise ValueError("continued Adam counters must advance from4, not reset")
        for k in ("exp_avg", "exp_avg_sq"):
            t = s[k]
            if (
                not isinstance(t, torch.Tensor)
                or t.shape != p.shape
                or t.dtype != p.dtype
                or t.device.type != "cpu"
                or not torch.isfinite(t).all()
                or (k == "exp_avg_sq" and (t < 0).any())
            ):
                raise ValueError("finite complete continued Adam moments required")
    for actual, initial in zip(
        optimizer["param_groups"], expected_initial["param_groups"], strict=True
    ):
        if (
            not same_nested(
                {k: v for k, v in actual.items() if k != "lr"},
                {k: v for k, v in initial.items() if k != "lr"},
            )
            or type(actual["lr"]) not in (int, float)
            or not 0 < actual["lr"] <= initial["lr"]
        ):
            raise ValueError("optimizer ownership/hyperparameters cannot silently change")
    rng = state.get("rng_state")
    if (
        not isinstance(rng, torch.Tensor)
        or rng.device.type != "cpu"
        or rng.dtype != torch.uint8
        or rng.shape != torch.get_rng_state().shape
    ):
        raise ValueError("valid continued CPU RNG state required")


def load_continual_refinement(
    behavior: RefinementMotorAdapter,
    prior_critic: MotorCritic,
    prior_optimizer: dict[str, Any],
    *,
    prior_training_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[RefinementMotorAdapter, dict[str, Any]]:
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or (root / "failed.json").exists()
        or file_hash(rp) != expected_receipt_hash
    ):
        raise ValueError("exact complete generation1 training required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    cfg = continual_config(r["config"]["state_id"])
    if (
        r["status"] != STATUS
        or r["checkpoint_kind"] != KIND
        or r["config"] != cfg
        or d["config"] != cfg
        or r["prior_training_hash"] != prior_training_hash
        or r["declaration_hash"] != file_hash(dp)
        or any(
            r[k] != d[k]
            for k in (
                "input_file_hashes",
                "prior_training_hash",
                "prepared_hash",
                "frozen_parent_hash",
                "source_head",
            )
        )
        or r["new_actor_steps"] != r["new_critic_steps"]
        or r["cumulative_actor_steps"] != 4 + r["new_actor_steps"]
        or r["cumulative_critic_steps"] != 4 + r["new_critic_steps"]
        or r["old_arm_rows_used"] != 0
        or r["new_policy_rollout_worlds"] != 24
        or r["physical_steps_executed"] != 0
        or any(
            r[k] is not True
            for k in (
                "trained_prior_critic_used",
                "prior_adam_moments_carried",
                "learning_rate_rebased",
                "two_generation_parameter_continuity_verified",
            )
        )
        or any(
            r[k] is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "continuous_learning_proven",
                "physical_retention_proven",
            )
        )
        or len(r["history"]) != 4
        or sum(h["update"]["applied"] for h in r["history"]) != r["new_actor_steps"]
    ):
        raise ValueError("exact new-policy generation1 continuity contract required")
    for i, h in enumerate(r["history"], start=1):
        if (
            h["epoch"] != i
            or type(h["update"]["applied"]) is not bool
            or not 1 <= len(h["update"]["attempts"]) <= 8
        ):
            raise ValueError("all bounded actual generation1 update history required")
        if h["update"]["applied"] and (
            not h["update"]["attempts"][-1]["accepted"]
            or not 0 <= h["update"]["attempts"][-1]["kl"] <= 0.015
        ):
            raise ValueError("all accepted continuation steps must satisfy trust region")
    paths = [
        Path(p)
        for p, v in r["input_file_hashes"].items()
        if v == r["prepared_hash"] and p.endswith("/complete.json")
    ]
    if len(paths) != 1 or file_hash(paths[0]) != r["prepared_hash"]:
        raise ValueError("exact unique independently prepared learned-arm batch required")
    prepared = json.loads(paths[0].read_text())
    if (
        prepared["status"] != "CONTINUAL_NEW_LEARNED_ARM24_AND_TRAINED_CRITIC_PREPARED_NOT_UPDATED"
        or prepared["config"] != cfg
        or prepared["prior_training_hash"] != prior_training_hash
        or prepared["selected_indices"] != list(range(1, 48, 2))
        or prepared["old_arm_rows_used"] != 0
    ):
        raise ValueError("only actual new learned-arm experience can train a successor")
    cp = Path(r["checkpoint_path"])
    if (
        cp.resolve().parent != root.resolve()
        or file_hash(cp, maximum=16 * 1024 * 1024) != r["checkpoint_hash"]
    ):
        raise ValueError("confined exact tensor-only generation1 checkpoint required")
    state = torch.load(cp, map_location="cpu", weights_only=True)
    validate_continual_checkpoint(
        state,
        behavior,
        prior_critic,
        prior_optimizer,
        config=cfg,
        commitments={
            k: r[k] for k in ("prior_training_hash", "prepared_hash", "frozen_parent_hash")
        },
        steps=r["new_actor_steps"],
    )
    result = copy.deepcopy(behavior).eval().requires_grad_(False)
    result.head.load_state_dict(state["head"], strict=True)
    return result, r
