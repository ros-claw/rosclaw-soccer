"""One bounded generation1 actor/critic/Adam continuation; research only."""

import copy
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
